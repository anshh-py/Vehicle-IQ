from collections.abc import Generator
from datetime import date
from uuid import UUID, uuid4

import pytest
import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import OCRResult, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.schemas.vehicle_lookup import VehicleRegistrationDetails
from app.vehicle_lookup.provider import get_vehicle_registration_provider
from app.vehicle_lookup.provider import VehicleDetailsHTTPProvider
from app.vehicle_lookup.vehicle_services import ChallanHTTPProvider, FastagHTTPProvider
from app.vehicle_lookup.vehicle_services import get_challan_provider, get_fastag_provider


class FakeVehicleProvider:
    provider_name = "test-vehicle-provider"

    def __init__(self, response=None, error=None):
        self.response = response or VehicleRegistrationDetails(
            registration_number="RJ45CA1879",
            make="Maruti Suzuki",
            model="Swift",
            fuel_type="Petrol",
            registration_date=date(2021, 5, 12),
            vehicle_class="Motor Car",
            registration_valid_until=date(2036, 5, 11),
        )
        self.error = error
        self.received = []

    def lookup(self, registration_number):
        self.received.append(registration_number)
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def lookup_api() -> Generator[tuple[TestClient, object, str, str], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assessment = VehicleAssessment()
        other_assessment = VehicleAssessment()
        session.add_all([assessment, other_assessment])
        session.flush()
        ocr = OCRResult(
            assessment_id=assessment.id,
            image_id=None,
            result_type="image_ocr",
            status="completed",
            provider="test-ocr",
            raw_text="Registration: RJ45 CA 1879",
        )
        session.add(ocr)
        session.commit()
        assessment_id = str(assessment.id)
        other_assessment_id = str(other_assessment.id)
        ocr_id = str(ocr.id)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: None
    try:
        with TestClient(app) as client:
            yield client, engine, assessment_id, ocr_id
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_ocr_registration_lookup_normalizes_persists_and_appears_in_report(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api
    provider = FakeVehicleProvider()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )

    assert response.status_code == 201
    assert provider.received == ["RJ45CA1879"]
    body = response.json()
    assert body["details"]["registration_number"] == "RJ45CA1879"
    assert body["details"]["make"] == "Maruti Suzuki"
    assert body["details"]["registration_date"] == "2021-05-12"
    with Session(engine) as session:
        rows = session.scalars(
            select(OCRResult).where(OCRResult.result_type == "registration_lookup")
        ).all()
        assert len(rows) == 1
        assert rows[0].assessment_id == UUID(assessment_id)
        assert rows[0].raw_text == "RJ45CA1879"
        assert rows[0].result_data["make"] == "Maruti Suzuki"

    report = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report.status_code == 200
    report_data = report.json()["report_data"]
    assert report_data["registration_lookup_results"][0]["details"]["model"] == "Swift"
    assert report_data["stages"]["ocr"]["result_count"] == 1


def test_partial_lookup_response_keeps_unknown_fields_unavailable(lookup_api):
    client, _engine, assessment_id, ocr_id = lookup_api
    provider = FakeVehicleProvider(
        VehicleRegistrationDetails(registration_number="RJ45CA1879", make="Maruti Suzuki")
    )
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )

    assert response.status_code == 201
    assert response.json()["details"]["model"] is None
    assert response.json()["details"]["fuel_type"] is None
    assert response.json()["details"]["registration_valid_until"] is None

    report = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report.status_code == 200
    assert report.json()["report_data"]["vehicle_details_lookup"]["status"] == "partial"


@pytest.mark.parametrize("raw_text", [None, "", "No plate found", "A123"])
def test_invalid_or_missing_ocr_registration_is_rejected(lookup_api, raw_text):
    client, engine, assessment_id, ocr_id = lookup_api
    with Session(engine) as session:
        ocr = session.get(OCRResult, UUID(ocr_id))
        ocr.raw_text = raw_text
        session.commit()
    provider = FakeVehicleProvider()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )

    assert response.status_code == 422
    assert provider.received == []


def test_unconfigured_and_failed_provider_are_not_reported_as_success(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api
    endpoint = f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"

    unavailable = client.post(endpoint)
    assert unavailable.status_code == 503

    app.dependency_overrides[get_vehicle_registration_provider] = lambda: FakeVehicleProvider(
        error=TimeoutError("sensitive provider payload")
    )
    failed = client.post(endpoint)
    assert failed.status_code == 502
    assert "sensitive" not in failed.text
    with Session(engine) as session:
        assert session.scalar(
            select(OCRResult).where(OCRResult.result_type == "registration_lookup")
        ) is None


def test_lookup_enforces_assessment_association_and_provider_response_number(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api
    provider = FakeVehicleProvider(
        VehicleRegistrationDetails(registration_number="KA01AB1234", make="Other")
    )
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    cross_assessment = client.post(
        f"/api/v1/assessments/{uuid4()}/ocr-results/{ocr_id}/registration-lookup"
    )
    mismatch = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )

    assert cross_assessment.status_code == 404
    assert mismatch.status_code == 502
    with Session(engine) as session:
        assert session.scalar(
            select(OCRResult).where(OCRResult.result_type == "registration_lookup")
        ) is None


def test_lookup_rejects_ocr_row_linked_to_another_assessments_image(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api
    with Session(engine) as session:
        other_assessment = session.scalar(
            select(VehicleAssessment).where(VehicleAssessment.id != UUID(assessment_id))
        )
        image = UploadedImage(
            assessment_id=other_assessment.id,
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key="other-assessment-image",
            original_filename="other.png",
            content_type="image/png",
            byte_size=12,
            sha256="digest",
        )
        session.add(image)
        session.flush()
        ocr = session.get(OCRResult, UUID(ocr_id))
        ocr.image_id = image.id
        session.commit()
    provider = FakeVehicleProvider()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )

    assert response.status_code == 404
    assert provider.received == []


def test_candidate_http_provider_integrates_with_lookup_and_report(lookup_api):
    client, _engine, assessment_id, ocr_id = lookup_api
    provider = VehicleDetailsHTTPProvider(
        "https://vehiclex.profilework239.workers.dev/",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "data": {
                        "vehicle": {
                            "make": "Maruti Suzuki",
                            "model": "Swift",
                            "fuel": "Petrol",
                            "rto": "Jaipur",
                            "owner_name": "Must not persist",
                        }
                    }
                },
            )
        ),
    )
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider

    lookup = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/registration-lookup"
    )
    report = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert lookup.status_code == 201
    assert lookup.json()["details"]["rto"] == "Jaipur"
    assert report.status_code == 200
    reported = report.json()["report_data"]["registration_lookup_results"][0]["details"]
    assert reported["registration_number"] == "RJ45CA1879"
    assert reported["rto"] == "Jaipur"
    assert "owner_name" not in reported


def test_challan_and_fastag_are_independent_and_report_partial_success(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api
    challan = ChallanHTTPProvider(
        "https://challan-info-new-abhigyan.vercel.app/challaninfo",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "status": "success",
                    "data": {
                        "challans": [
                            {
                                "status": "pending",
                                "description": "Speed limit violation",
                                "amount": 500,
                                "date": "2025-03-01",
                                "payment_status": "unpaid",
                            }
                        ]
                    },
                },
            )
        ),
    )
    fastag = FastagHTTPProvider(
        "https://abhigyan-fastag-info.vercel.app/vehtofastaginfo",
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )
    app.dependency_overrides[get_challan_provider] = lambda: challan
    app.dependency_overrides[get_fastag_provider] = lambda: fastag

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/vehicle-services"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["challan"]["status"] == "available"
    assert payload["challan"]["details"]["records"][0]["amount"] == "500"
    assert payload["fastag"]["status"] == "unavailable"
    assert payload["fastag"]["error_code"] == "provider_unavailable"
    with Session(engine) as session:
        service_rows = session.scalars(
            select(OCRResult).where(OCRResult.result_type.in_(["challan_lookup", "fastag_lookup"]))
        ).all()
        assert {row.result_type for row in service_rows} == {"challan_lookup", "fastag_lookup"}
        assert all("owner_name" not in (row.result_data or {}) for row in service_rows)

    report = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report.status_code == 200
    data = report.json()["report_data"]
    assert data["challan_lookup"]["status"] == "available"
    assert data["challan_lookup"]["details"]["records"][0]["violation_description"] == "Speed limit violation"
    assert data["fastag_lookup"]["status"] == "unavailable"


def test_disabled_challan_and_fastag_return_unavailable_without_provider_calls(lookup_api):
    client, engine, assessment_id, ocr_id = lookup_api

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/vehicle-services"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "unavailable"
    assert response.json()["challan"]["status"] == "unavailable"
    assert response.json()["fastag"]["status"] == "unavailable"
    with Session(engine) as session:
        assert session.scalar(
            select(OCRResult).where(OCRResult.result_type.in_(["challan_lookup", "fastag_lookup"]))
        ) is None
    report = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report.status_code == 200
    assert report.json()["report_data"]["challan_lookup"]["status"] == "unavailable"
    assert report.json()["report_data"]["fastag_lookup"]["status"] == "unavailable"


def test_fastag_success_survives_challan_provider_failure(lookup_api):
    client, _engine, assessment_id, ocr_id = lookup_api
    challan = ChallanHTTPProvider(
        "https://challan-info-new-abhigyan.vercel.app/challaninfo",
        transport=httpx.MockTransport(lambda _request: httpx.Response(503)),
    )
    fastag = FastagHTTPProvider(
        "https://abhigyan-fastag-info.vercel.app/vehtofastaginfo",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"data": {"status": "active", "issuing_bank": "Example Bank", "vehicle_class": "4W"}},
            )
        ),
    )
    app.dependency_overrides[get_challan_provider] = lambda: challan
    app.dependency_overrides[get_fastag_provider] = lambda: fastag

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_id}/vehicle-services"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["challan"]["status"] == "unavailable"
    assert body["fastag"]["status"] == "partial"
    assert body["fastag"]["details"]["issuing_bank"] == "Example Bank"
