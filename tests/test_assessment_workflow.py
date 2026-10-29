from collections.abc import Generator
from datetime import date
from decimal import Decimal
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.v1.damage import get_damage_provider
from app.api.v1.ocr import get_ocr_client
from app.api.v1.vehicle_lookup import get_challan_provider, get_fastag_provider, get_vehicle_registration_provider
from app.db.base import Base
from app.db.session import get_db
from app.damage.inference import DamageDetection, DamageInferenceResult
from app.main import app
from app.ocr.client import OCRRecognition
from app.pricing.provider import get_price_provider
from app.schemas.pricing import PricePrediction
from app.schemas.vehicle_lookup import VehicleRegistrationDetails
from app.schemas.vehicle_services import ChallanDetails, ChallanRecord, FastagDetails
from app.storage.images import LocalImageStorage, get_image_storage
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
            fuel_type="Petrol",
            registration_date=date(2021, 5, 12),
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
    def predict(self, _vehicle):
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
    storage = LocalImageStorage(tmp_path / "workflow-images")

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_image_storage] = lambda: storage
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
    assert damage.json()["status"] == "completed"
    assert damage.json()["processed_count"] == 10
    assert all(len(result["findings"]) == 1 for result in damage.json()["results"])

    price = client.post(
        f"/api/v1/assessments/{assessment_id}/price-estimates",
        json={
            "vehicle": {
                "make": "Example Motors",
                "model": "City",
                "registration_year": 2021,
                "odometer_km": 42000,
                "fuel_type": "petrol",
                "transmission": "manual",
                "seller_type": "Individual",
                "mileage_kmpl": 21.5,
                "engine_cc": 1197,
                "max_power_bhp": 88.5,
                "seats": 5,
                "owner_count": 1,
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
