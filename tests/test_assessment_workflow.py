from collections.abc import Generator
from datetime import date
from decimal import Decimal
from io import BytesIO
import json
from uuid import UUID

import httpx
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.v1.damage import get_damage_provider
from app.api.v1.ocr import get_ocr_client
from app.api.v1.vehicle_lookup import get_challan_provider, get_fastag_provider, get_vehicle_registration_provider
from app.api.v1.reports import get_market_listing_provider
from app.db.base import Base
from app.db.models import AssessmentReport
from app.db.session import get_db
from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import DamageWorkerUnavailable
from app.main import app
from app.ocr.client import HuggingFaceGradioOCRClient, OCRRecognition, OCRServiceUnavailable
from app.pricing.provider import get_price_provider
from app.schemas.pricing import PricePrediction
from app.schemas.vehicle_lookup import VehicleRegistrationDetails
from app.schemas.vehicle_services import ChallanDetails, ChallanRecord, FastagDetails
from app.vehicle_lookup.provider import VehicleLookupUnavailable


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), color=(180, 180, 180)).save(output, format="PNG")
    return output.getvalue()


class FakeOCR:
    provider_name = "mock_ocr"

    def recognize(self, _content, _filename, _content_type):
        return OCRRecognition("RJ45CA1879", 0.98)


class FakeVehicleDetails:
    provider_name = "mock_vehicle_details"

    def lookup(self, registration_number):
        return VehicleRegistrationDetails(
            registration_number=registration_number,
            make="Example Motors",
            model="City",
            variant="ZX",
            fuel_type="Petrol",
            registration_date=date(2021, 5, 12),
            manufacturing_year=2021,
            engine_cc=1498,
            seats=5,
            max_power_bhp=119.0,
            transmission="manual",
            mileage_kmpl=18.0,
            owner_count=None,
        )


class FakeChallan:
    provider_name = "mock_challan"

    def lookup(self, _registration_number):
        return ChallanDetails(
            status="success",
            records=[
                ChallanRecord(
                    status="pending",
                    violation_description="Speed limit violation",
                    amount=Decimal("500"),
                    date=date(2025, 3, 1),
                    payment_status="unpaid",
                )
            ],
        )


class FailedFastag:
    provider_name = "mock_fastag"

    def lookup(self, _registration_number):
        raise VehicleLookupUnavailable("mock outage")


class FakeDamage:
    provider_name = "mock_damage"

    def detect(self, _content, _extension):
        return DamageInferenceResult(
            detections=(DamageDetection("dent", 0.9, (1.0, 2.0, 12.0, 14.0)),),
            inference_seconds=0.01,
        )


class FakePrice:
    def __init__(self):
        self.received = None

    def predict(self, vehicle):
        self.received = vehicle
        return PricePrediction(
            amount=Decimal("625000"),
            currency="INR",
            lower_bound=None,
            upper_bound=None,
            method="mock",
            model_name="test_price_model",
        )


@pytest.fixture
def workflow_client(tmp_path) -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ocr_client] = lambda: FakeOCR()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: FakeVehicleDetails()
    app.dependency_overrides[get_challan_provider] = lambda: FakeChallan()
    app.dependency_overrides[get_fastag_provider] = lambda: FailedFastag()
    app.dependency_overrides[get_damage_provider] = lambda: FakeDamage()
    app.dependency_overrides[get_price_provider] = lambda: FakePrice()
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_complete_assessment_workflow_preserves_partial_external_results(workflow_client):
    client = workflow_client
    created = client.post(
        "/api/v1/assessments",
        json={
            "asking_price_inr": "700000.00",
            "vehicle_details": {"make": "Example Motors", "model": "City"},
        },
    )
    assert created.status_code == 201
    assessment_id = created.json()["id"]

    image_ids = []
    for index in range(10):
        uploaded = client.post(
            f"/api/v1/assessments/{assessment_id}/images",
            files={"file": (f"vehicle-{index}.png", png_bytes(), "image/png")},
        )
        assert uploaded.status_code == 201
        image_ids.append(uploaded.json()["id"])

    ocr = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_ids[0]}/ocr")
    assert ocr.status_code == 200
    ocr_result_id = ocr.json()["id"]

    details = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_result_id}/registration-lookup"
    )
    services = client.post(
        f"/api/v1/assessments/{assessment_id}/ocr-results/{ocr_result_id}/vehicle-services"
    )
    assert details.status_code == 201
    assert services.status_code == 200
    assert services.json()["status"] == "partial"
    assert services.json()["challan"]["status"] == "available"
    assert services.json()["fastag"]["status"] == "unavailable"

    damage = client.post(
        f"/api/v1/assessments/{assessment_id}/damage",
        json={"image_ids": image_ids},
    )
    assert damage.status_code == 200
    assert damage.json()["status"] == "completed_with_findings"
    assert damage.json()["processed_count"] == 10
    assert all(len(result["findings"]) == 1 for result in damage.json()["results"])

    price = client.post(
        f"/api/v1/assessments/{assessment_id}/price-estimates",
        json={
            "vehicle": {
                "odometer_km": 42000,
                "seller_type": "Individual",
            }
        },
    )
    assert price.status_code == 201

    generated = client.post(f"/api/v1/assessments/{assessment_id}/report")
    retrieved = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert generated.status_code == retrieved.status_code == 200
    report = retrieved.json()["report_data"]
    assert report["assessment"]["asking_price_inr"] == "700000.00"
    assert report["stages"]["ocr"] == {
        "status": "partial",
        "result_count": 1,
        "completed_image_count": 1,
    }
    assert report["vehicle_details_lookup"]["status"] == "available"
    assert report["challan_lookup"]["status"] == "available"
    assert report["fastag_lookup"]["status"] == "unavailable"
    assert report["stages"]["damage_detection"]["finding_count"] == 10
    assert report["stages"]["damage_detection"]["images_with_findings"] == 10
    assert report["stages"]["pricing"]["status"] == "available"
    recommendation = report["buyer_recommendation"]
    assert recommendation["ml_estimate_inr"] == "625000.00"
    assert recommendation["absolute_difference_inr"] == "75000.00"
    assert recommendation["difference_percent_of_ml_estimate"] == "12.00"
    assert recommendation["market_evidence_available"] is False
    assert report["market_valuation"]["status"] == "unavailable"


def _create_workflow_assessment_with_images(
    client, image_count=1, registration_number=None
):
    assessment_payload = {"asking_price_inr": "700000"}
    if registration_number is not None:
        assessment_payload["registration_number"] = registration_number
    created = client.post("/api/v1/assessments", json=assessment_payload)
    assert created.status_code == 201
    assessment_id = created.json()["id"]
    image_ids = []
    for index in range(image_count):
        uploaded = client.post(
            f"/api/v1/assessments/{assessment_id}/images",
            files={"file": (f"vehicle-workflow-{index}.png", png_bytes(), "image/png")},
        )
        assert uploaded.status_code == 201
        image_ids.append(uploaded.json()["id"])
    return assessment_id, image_ids


def test_workflow_runs_gradio_ocr_from_uploaded_image_storage(workflow_client, monkeypatch):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", False)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/gradio_api/upload":
            body = request.read()
            assert request.method == "POST"
            assert b'name="files"' in body
            assert b"image/png" in body
            assert b"\x89PNG\r\n\x1a\n" in body
            return httpx.Response(200, json=["/tmp/gradio/workflow-image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            assert request.method == "POST"
            payload = json.loads(request.read())
            assert payload == {
                "image_path": {
                    "path": "/tmp/gradio/workflow-image.png",
                    "meta": {"_type": "gradio.FileData"},
                },
                "mode": "gundam",
                "prompt": "",
            }
            return httpx.Response(200, json={"event_id": "workflow-event"})
        if request.url.path == "/gradio_api/call/run_ocr/workflow-event":
            return httpx.Response(
                200,
                text='event: complete\ndata: [{"text":"RJ45CA1879","done":true}]\n\n',
                headers={"content-type": "text/event-stream"},
            )
        raise AssertionError(f"Unexpected Gradio request: {request.method} {request.url.path}")

    app.dependency_overrides[get_ocr_client] = lambda: HuggingFaceGradioOCRClient(
        "https://baidu-unlimited-ocr.hf.space",
        transport=httpx.MockTransport(handler),
    )

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stages"]["ocr"]["status"] == "available"
    assert body["stages"]["ocr"]["result"][0]["provider"] == "huggingface_gradio"
    assert body["stages"]["ocr"]["result"][0]["text"] == "RJ45CA1879"
    report = body["stages"]["report"]["result"]["report_data"]
    assert report["stages"]["ocr"]["status"] == "available"
    assert report["ocr_results"][0]["status"] == "completed"
    assert seen_paths == [
        "/gradio_api/upload",
        "/gradio_api/call/v2/run_ocr",
        "/gradio_api/call/run_ocr/workflow-event",
    ]


def test_manual_registration_skips_ocr_and_looks_up_normalized_input(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(
        client, registration_number=" rj45 ca1879 "
    )

    class CountingOCR:
        provider_name = "mock_registration_ocr"
        calls = 0

        def recognize(self, *_args):
            self.calls += 1
            return OCRRecognition("4 RJ45CA1879", 0.98)

    class RecordingLookup:
        provider_name = "manual_registration_test"

        def __init__(self):
            self.registrations = []

        def lookup(self, registration_number):
            self.registrations.append(registration_number)
            return FakeVehicleDetails().lookup(registration_number)

    ocr = CountingOCR()
    lookup = RecordingLookup()
    app.dependency_overrides[get_ocr_client] = lambda: ocr
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: lookup

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    stages = response.json()["stages"]
    assert stages["ocr"]["status"] == "skipped"
    assert stages["vehicle_details_lookup"]["status"] == "available"
    assert ocr.calls == 0
    assert lookup.registrations == ["RJ45CA1879"]
    assert stages["vehicle_details_lookup"]["result"]["details"]["registration_number"] == "RJ45CA1879"
    assert client.get(f"/api/v1/assessments/{assessment_id}").json()[
        "registration_number"
    ] == "RJ45CA1879"


def test_without_manual_registration_ocr_extracts_leading_noise_and_looks_up_plate(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    class CountingOCR:
        provider_name = "mock_registration_ocr"
        calls = 0

        def recognize(self, *_args):
            self.calls += 1
            return OCRRecognition("4 RJ45CA1879", 0.98)

    class RecordingLookup:
        provider_name = "ocr_registration_test"

        def __init__(self):
            self.registrations = []

        def lookup(self, registration_number):
            self.registrations.append(registration_number)
            return FakeVehicleDetails().lookup(registration_number)

    ocr = CountingOCR()
    lookup = RecordingLookup()
    app.dependency_overrides[get_ocr_client] = lambda: ocr
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: lookup

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    stages = response.json()["stages"]
    assert stages["ocr"]["status"] == "available"
    assert ocr.calls == 1
    assert lookup.registrations == ["RJ45CA1879"]
    assert stages["vehicle_details_lookup"]["status"] == "available"


def test_invalid_manual_registration_returns_validation_error_without_ocr_fallback(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(
        client, registration_number="not a registration"
    )

    class CountingOCR:
        calls = 0

        def recognize(self, *_args):
            self.calls += 1
            return OCRRecognition("RJ45CA1879", 0.98)

    class CountingLookup:
        calls = 0

        def lookup(self, _registration_number):
            self.calls += 1
            return FakeVehicleDetails().lookup("RJ45CA1879")

    ocr = CountingOCR()
    lookup = CountingLookup()
    app.dependency_overrides[get_ocr_client] = lambda: ocr
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: lookup

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Manual registration number is invalid."
    assert ocr.calls == 0
    assert lookup.calls == 0


def test_manual_registration_fallback_continues_after_unreadable_ocr_and_persists_one_report(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    class NoPlateOCR:
        provider_name = "mock_ocr"

        def recognize(self, *_args):
            return OCRRecognition("No readable plate text")

    class CountingLookup(FakeVehicleDetails):
        def __init__(self):
            self.calls = []

        def lookup(self, registration_number):
            self.calls.append(registration_number)
            return super().lookup(registration_number)

    lookup = CountingLookup()
    app.dependency_overrides[get_ocr_client] = lambda: NoPlateOCR()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: lookup

    first = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )
    assert first.status_code == 200
    assert first.json()["stages"]["ocr"]["status"] == "available"
    assert first.json()["stages"]["vehicle_details_lookup"]["errors"][0]["code"] in {
        "registration_not_found",
        "registration_needs_review",
    }

    saved = client.post(
        f"/api/v1/assessments/{assessment_id}/registration-number",
        json={"registration_number": " rj45 ca1879 "},
    )
    assert saved.status_code == 200
    assert saved.json()["source"] == "manual"

    continued = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )
    assert continued.status_code == 200
    stages = continued.json()["stages"]
    assert stages["ocr"]["status"] == "skipped"
    assert stages["vehicle_details_lookup"]["status"] == "available"
    assert stages["report"]["status"] == "available"
    assert lookup.calls == ["RJ45CA1879"]
    original_report_id = stages["report"]["result"]["id"]

    assert client.post(
        f"/api/v1/assessments/{assessment_id}/registration-number",
        json={"registration_number": "RJ45CA1879"},
    ).status_code == 200
    repeated = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )
    assert repeated.status_code == 200
    assert lookup.calls == ["RJ45CA1879"]
    assert repeated.json()["stages"]["report"]["result"]["id"] == original_report_id

    report = client.get(f"/api/v1/assessments/{assessment_id}/report").json()["report_data"]
    assert report["assessment"]["registration_number"] == "RJ45CA1879"
    assert report["assessment"]["registration_number_source"] == "manual"
    assert len(report["ocr_results"]) == 1
    assert len(report["registration_lookup_results"]) == 1
    assert report["registration_lookup_results"][0]["source"] == "manual"


def test_manual_registration_lookup_failure_still_generates_report(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    class NoPlateOCR:
        provider_name = "mock_ocr"

        def recognize(self, *_args):
            return OCRRecognition("No plate")

    class FailedLookup:
        provider_name = "failed_lookup"

        def lookup(self, _registration_number):
            raise VehicleLookupUnavailable("mock provider outage")

    app.dependency_overrides[get_ocr_client] = lambda: NoPlateOCR()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: FailedLookup()
    first = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow", json={"image_ids": image_ids}
    )
    assert first.status_code == 200
    assert client.post(
        f"/api/v1/assessments/{assessment_id}/registration-number",
        json={"registration_number": "RJ45CA1879"},
    ).status_code == 200

    continued = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow", json={"image_ids": image_ids}
    )
    assert continued.status_code == 200
    stages = continued.json()["stages"]
    assert stages["vehicle_details_lookup"]["status"] == "unavailable"
    assert stages["report"]["status"] == "available"
    report = client.get(f"/api/v1/assessments/{assessment_id}/report").json()["report_data"]
    assert report["assessment"]["registration_number_source"] == "manual"
    assert report["vehicle_details_lookup"]["error_code"] == "provider_unavailable"


def test_low_confidence_registration_requires_manual_correction_before_lookup(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    class LowConfidenceOCR:
        provider_name = "low_confidence_registration_ocr"

        def recognize(self, *_args):
            return OCRRecognition("RJ45CA1879", 0.42)

    class CountingLookup:
        calls = 0

        def lookup(self, _registration_number):
            self.calls += 1
            return FakeVehicleDetails().lookup("RJ45CA1879")

    lookup = CountingLookup()
    app.dependency_overrides[get_ocr_client] = lambda: LowConfidenceOCR()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: lookup

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    stage = response.json()["stages"]["vehicle_details_lookup"]
    assert stage["status"] == "unavailable"
    assert stage["result"] == {
        "registration_number_candidate": "RJ45CA1879",
        "confidence": 0.42,
        "requires_manual_correction": True,
    }
    assert stage["errors"][0]["code"] == "registration_needs_review"
    assert lookup.calls == 0


def _price_vehicle_payload():
    return {
        "odometer_km": 42000,
        "seller_type": "Individual",
    }


def test_workflow_endpoint_runs_available_stages_and_returns_report(workflow_client, monkeypatch):
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(workflow_client, image_count=2)

    response = workflow_client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": _price_vehicle_payload()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert [body["stages"][name]["status"] for name in (
        "ocr", "vehicle_details_lookup", "damage_detection", "pricing", "report"
    )] == [
        "available",
        "available",
        "completed_with_findings",
        "available",
        "available",
    ]
    assert len(body["stages"]["ocr"]["result"]) == 2
    assert body["stages"]["damage_detection"]["result"]["processed_count"] == 2
    assert body["stages"]["pricing"]["result"]["amount"] == "625000.00"
    assert body["stages"]["report"]["result"]["report_data"]["market_valuation"]["status"] == "unavailable"


def test_workflow_enabled_lookup_fills_only_returned_pricing_features(workflow_client, monkeypatch):
    client = workflow_client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)

    class RecordingVehicleLookup:
        provider_name = "mock_recording_vehicle_details"

        def __init__(self):
            self.registrations = []

        def lookup(self, registration_number):
            self.registrations.append(registration_number)
            return FakeVehicleDetails().lookup(registration_number)

    vehicle_lookup = RecordingVehicleLookup()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: vehicle_lookup

    class RecordingPrice:
        def __init__(self):
            self.vehicle = None

        def predict(self, vehicle):
            self.vehicle = vehicle
            return FakePrice().predict(vehicle)

    provider = RecordingPrice()
    app.dependency_overrides[get_price_provider] = lambda: provider
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={
            "image_ids": image_ids,
            "vehicle": {"odometer_km": 42000, "seller_type": "Individual"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stages"]["vehicle_details_lookup"]["status"] == "available"
    assert body["stages"]["vehicle_details_lookup"]["result"]["field_provenance"]["engine_cc"] == "vehicle_api"
    assert vehicle_lookup.registrations == ["RJ45CA1879"]
    assert body["stages"]["pricing"]["status"] == "available"
    assert provider.vehicle.make == "Example Motors"
    assert provider.vehicle.model == "City"
    assert provider.vehicle.fuel_type.value == "petrol"
    assert provider.vehicle.registration_year == 2021
    report = body["stages"]["report"]["result"]["report_data"]
    assert report["assessment"]["vehicle_details"]["make"] == "Example Motors"
    assert report["assessment"]["registration_number"] == "RJ45CA1879"
    assert report["vehicle_details_lookup"]["details"]["model"] == "City"
    assert report["price_estimates"][0]["vehicle"]["registration_year"] == 2021
    assert report["price_estimates"][0]["provenance"]["make"] == "vehicle_api"
    assert report["price_estimates"][0]["provenance"]["odometer_km"] == "user_input"
    assert "owner_count" not in report["price_estimates"][0]["provenance"]
    assert report["stages"]["pricing"]["status"] == "available"


def test_workflow_lookup_failure_is_explained_in_persisted_report(workflow_client, monkeypatch):
    client = workflow_client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)

    class UnavailableLookup:
        def lookup(self, _registration_number):
            raise VehicleLookupUnavailable("sensitive provider detail must not be surfaced")

    app.dependency_overrides[get_vehicle_registration_provider] = lambda: UnavailableLookup()
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    body = response.json()
    lookup = body["stages"]["vehicle_details_lookup"]
    assert lookup["status"] == "unavailable"
    assert lookup["errors"][0]["code"] == "provider_unavailable"
    report = body["stages"]["report"]["result"]["report_data"]
    assert report["vehicle_details_lookup"]["workflow_status"] == "unavailable"
    assert report["vehicle_details_lookup"]["error_code"] == "provider_unavailable"
    assert "sensitive provider detail" not in str(report)


def test_workflow_disabled_lookup_is_not_called_and_missing_features_are_reported(workflow_client, monkeypatch):
    client = workflow_client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", False)

    class CountingLookup:
        calls = 0

        def lookup(self, _registration):
            self.calls += 1
            return FakeVehicleDetails().lookup(_registration)

    provider = CountingLookup()
    app.dependency_overrides[get_vehicle_registration_provider] = lambda: provider
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": {"odometer_km": 40000}},
    )

    assert response.status_code == 200
    body = response.json()
    assert provider.calls == 0
    assert body["stages"]["vehicle_details_lookup"]["errors"][0]["code"] == "provider_disabled"
    pricing = body["stages"]["pricing"]
    assert pricing["status"] == "unavailable"
    assert pricing["errors"][0]["code"] == "missing_vehicle_attributes"
    assert "registration_year" in pricing["result"]["missing_fields"]
    assert "transmission" in pricing["result"]["missing_fields"]
    assert pricing["result"]["input_provenance"]["odometer_km"] == "user_input"
    report = body["stages"]["report"]["result"]["report_data"]
    assert report["stages"]["pricing"]["status"] == "skipped"
    assert report["stages"]["pricing"]["error_code"] == "missing_vehicle_attributes"
    assert client.get(f"/api/v1/assessments/{assessment_id}/report").json()["report_data"]["stages"]["pricing"]["status"] == "skipped"


def test_workflow_rejects_manual_vehicle_specifications(workflow_client):
    assessment_id, image_ids = _create_workflow_assessment_with_images(workflow_client)

    response = workflow_client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": {"engine_cc": 1200}},
    )

    assert response.status_code == 422


def test_workflow_manual_vehicle_corrections_are_persisted_and_used(workflow_client, monkeypatch):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    initial = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={
            "image_ids": image_ids,
            "vehicle": {"odometer_km": 42000, "seller_type": "Individual"},
        },
    )
    assert initial.status_code == 200
    initial_pricing = initial.json()["stages"]["pricing"]
    assert initial_pricing["status"] == "available"
    assert initial_pricing["result"]["vehicle"]["transmission"] == "manual"
    assert initial_pricing["result"]["vehicle"]["mileage_kmpl"] == 18.0
    assert initial_pricing["result"]["vehicle"]["max_power_bhp"] == 119.0
    assert initial_pricing["result"]["provenance"]["transmission"] == "vehicle_api"

    corrected = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={
            "image_ids": image_ids,
            "vehicle": {
                "odometer_km": 42000,
                "seller_type": "Individual",
                "transmission": "automatic",
                "mileage_kmpl": 19.2,
                "max_power_bhp": 110,
            },
        },
    )
    assert corrected.status_code == 200
    pricing = corrected.json()["stages"]["pricing"]
    assert pricing["status"] == "available"
    assert pricing["result"]["vehicle"]["transmission"] == "automatic"
    assert pricing["result"]["vehicle"]["mileage_kmpl"] == 19.2
    assert pricing["result"]["vehicle"]["max_power_bhp"] == 110
    assert all(pricing["result"]["provenance"][field] == "user_input" for field in (
        "transmission", "mileage_kmpl", "max_power_bhp"
    ))
    assessment = client.get(f"/api/v1/assessments/{assessment_id}").json()
    assert all(assessment["vehicle_details_provenance"][field] == "user_input" for field in (
        "transmission", "mileage_kmpl", "max_power_bhp"
    ))


def test_workflow_can_be_completed_after_missing_vehicle_details_are_supplied(workflow_client, monkeypatch):
    client = workflow_client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", False)

    incomplete = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )
    assert incomplete.status_code == 200
    assert incomplete.json()["stages"]["pricing"]["result"]["missing_fields"]
    assert client.get(f"/api/v1/assessments/{assessment_id}/report").json()["report_data"]["stages"]["pricing"]["status"] == "skipped"

    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    completed = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": _price_vehicle_payload()},
    )
    assert completed.status_code == 200
    body = completed.json()
    assert body["stages"]["pricing"]["status"] == "available"
    assert body["stages"]["pricing"]["result"]["amount"] == "625000.00"
    persisted = client.get(f"/api/v1/assessments/{assessment_id}/report").json()["report_data"]
    assert persisted["stages"]["pricing"]["status"] == "available"
    assert persisted["price_estimates"][-1]["vehicle"]["odometer_km"] == 42000
    assert persisted["market_valuation"]["status"] == "unavailable"


def test_workflow_price_model_failure_is_distinct_and_saved_in_report(workflow_client, monkeypatch):
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    client = workflow_client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    class BrokenPrice:
        def predict(self, _vehicle):
            raise RuntimeError("model inference failed")

    app.dependency_overrides[get_price_provider] = lambda: BrokenPrice()
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": _price_vehicle_payload()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stages"]["pricing"]["status"] == "failed"
    assert body["stages"]["pricing"]["errors"][0]["code"] == "price_provider_failed"
    report = body["stages"]["report"]["result"]["report_data"]
    assert report["stages"]["pricing"]["status"] == "failed"
    assert report["stages"]["pricing"]["error_code"] == "price_provider_failed"


def test_workflow_endpoint_preserves_successful_stages_when_one_ocr_fails(workflow_client, monkeypatch):
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(workflow_client, image_count=2)

    class IntermittentOCR:
        provider_name = "mock_intermittent_ocr"

        def __init__(self):
            self.calls = 0

        def recognize(self, _content, _filename, _content_type):
            self.calls += 1
            if self.calls == 1:
                raise OCRServiceUnavailable("mock OCR outage")
            return OCRRecognition("RJ45CA1879", 0.98)

    app.dependency_overrides[get_ocr_client] = lambda: IntermittentOCR()
    response = workflow_client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": _price_vehicle_payload()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["stages"]["ocr"]["status"] == "partial"
    assert len(body["stages"]["ocr"]["result"]) == 1
    assert body["stages"]["ocr"]["errors"][0]["code"] == "provider_unavailable"
    assert body["stages"]["damage_detection"]["status"] == "completed_with_findings"
    assert body["stages"]["pricing"]["status"] == "available"
    assert body["stages"]["report"]["status"] == "available"
    report_data = body["stages"]["report"]["result"]["report_data"]
    assert report_data["stages"]["ocr"]["status"] == "partial"
    assert report_data["stages"]["ocr"]["result_count"] == 1
    assert {item["status"] for item in report_data["ocr_results"]} == {"completed", "unavailable"}


def test_workflow_endpoint_marks_unconfigured_providers_unavailable(workflow_client):
    assessment_id, image_ids = _create_workflow_assessment_with_images(workflow_client)

    class UnavailableDamage:
        def detect(self, _content, _extension):
            raise DamageWorkerUnavailable("model unavailable")

    app.dependency_overrides[get_ocr_client] = lambda: None
    app.dependency_overrides[get_damage_provider] = lambda: UnavailableDamage()
    app.dependency_overrides[get_price_provider] = lambda: None
    response = workflow_client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": _price_vehicle_payload()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["stages"]["ocr"]["status"] == "unavailable"
    assert body["stages"]["damage_detection"]["status"] == "unavailable"
    assert body["stages"]["pricing"]["status"] == "unavailable"
    assert body["stages"]["report"]["status"] == "available"
    assert body["stages"]["report"]["result"]["report_data"]["market_valuation"]["status"] == "unavailable"


def test_workflow_retries_market_snapshot_after_lookup_resolves_missing_vehicle_identity(
    workflow_client, monkeypatch
):
    client = workflow_client
    monkeypatch.setattr("app.api.v1.workflow.settings.vehicle_lookup_enabled", True)
    assessment_id, image_ids = _create_workflow_assessment_with_images(
        client, registration_number="RJ45CA1879"
    )

    class RecordingMarketProvider:
        search_observability = {
            "queries_issued": 0,
            "raw_results_returned": 0,
            "listings_extracted": 0,
        }
        market_research_results = []
        market_research_rejection_counts = {}
        rejected_listing_counts = {}
        search_scopes = []

        def __init__(self):
            self.criteria = []

        def fetch_listings(self, criteria):
            self.criteria.append(criteria)
            return []

    provider = RecordingMarketProvider()
    app.dependency_overrides[get_market_listing_provider] = lambda: provider
    early = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert early.status_code == 200
    assert (
        early.json()["report_data"]["market_valuation"]["refresh_status"]
        == "insufficient_vehicle_details"
    )

    db_dependency = app.dependency_overrides[get_db]()
    db = next(db_dependency)
    try:
        report = db.query(AssessmentReport).filter_by(
            assessment_id=UUID(assessment_id)
        ).one()
        report_data = dict(report.report_data)
        stages = dict(report_data["stages"])
        stages["pricing"] = {"status": "available"}
        report_data["stages"] = stages
        report.report_data = report_data
        db.commit()
    finally:
        db_dependency.close()

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    stages = response.json()["stages"]
    assert stages["vehicle_details_lookup"]["status"] == "available"
    assert len(provider.criteria) == 1
    assert provider.criteria[0].make == "Example Motors"
    assert provider.criteria[0].model == "City"
