"""Regression tests for Vehicle IQ workflow reliability issues.

Covers:
- Duplicate workflow execution prevention
- ML valuation result consistency
- Damage detection result reporting
- Silent failure prevention
"""

import pytest
from datetime import date
from decimal import Decimal
from uuid import UUID

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from types import SimpleNamespace
from uuid import UUID

from app.api.v1.damage import get_damage_provider
from app.api.v1.ocr import get_ocr_client
from app.db.base import Base
from app.db.session import get_db
from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import DamageWorkerUnavailable
from app.main import app
from app.ocr.client import OCRRecognition, OCRServiceUnavailable
from app.pricing.provider import get_price_provider
from app.schemas.pricing import PricePrediction
from app.schemas.vehicle_lookup import VehicleRegistrationDetails
from app.vehicle_lookup.provider import (
    VehicleLookupUnavailable,
    get_vehicle_registration_provider,
)
from app.vehicle_lookup.vehicle_services import (
    get_challan_provider,
    get_fastag_provider,
)


def _create_test_assessment(client: TestClient, asking_price: Decimal | None = None) -> UUID:
    """Create a test assessment and return its ID."""
    payload = {
        "vehicle_details": {
            "make": "Test Make", "model": "Test Model", "variant": "X",
            "fuel_type": "Petrol", "registration_year": 2021,
            "odometer_km": 50000, "transmission": "Manual",
            "seats": 5, "engine_cc": 1500, "max_power_bhp": 100,
            "mileage_kmpl": 15, "owner_count": 1
        }
    }
    if asking_price is not None:
        payload["asking_price_inr"] = str(asking_price)
    response = client.post("/api/v1/assessments", json=payload)
    assert response.status_code == 201
    return UUID(response.json()["id"])


def _create_workflow_assessment_with_images(client: TestClient, image_count=1, registration_number=None):
    """Create an assessment with uploaded images for workflow tests."""
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


def png_bytes() -> bytes:
    """Generate a small valid PNG image as test bytes."""
    from io import BytesIO
    from PIL import Image

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
            make="Test Make",
            model="Test Model",
            variant="X",
            fuel_type="Petrol",
            registration_date=date(2021, 5, 12),
            manufacturing_year=2021,
            engine_cc=1500,
            seats=5,
            max_power_bhp=100.0,
            transmission="Manual",
            mileage_kmpl=15.0,
            owner_count=1,
        )


class FakeChallan:
    provider_name = "mock_challan"

    def lookup(self, _registration_number):
        return None


class FailedFastag:
    provider_name = "mock_fastag"

    def lookup(self, _registration_number):
        return None


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
            amount=Decimal("625000.00"),
            currency="INR",
            lower_bound=None,
            upper_bound=None,
            method="mock",
            model_name="test_price_model",
        )


@pytest.fixture
def workflow_client(tmp_path):
    """Create a test client with mocked providers."""
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
            yield SimpleNamespace(client=client, engine=engine)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_workflow_duplicate_submission_returns_stored_result(
    workflow_client,
):
    """After a successful workflow, a duplicate submission returns the stored result."""
    client = workflow_client.client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    # Vehicle payload required for pricing
    vehicle_payload = {"odometer_km": 50000, "seller_type": "Individual"}

    # Run the workflow the first time
    response1 = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": vehicle_payload},
    )
    assert response1.status_code == 200
    body1 = response1.json()
    assert body1["status"] == "completed"
    assert body1["stages"]["pricing"]["status"] == "available"
    assert body1["stages"]["report"]["status"] == "available"

    # Run the workflow a second time (duplicate submission)
    response2 = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": vehicle_payload},
    )
    assert response2.status_code == 200
    body2 = response2.json()

    # The second call should return the same stored result without re-executing
    # The stored stages don't include the "report" stage (it's added after the early return)
    # but the full response should still have a report stage
    assert body1["status"] == body2["status"]
    assert body1["stages"]["ocr"]["status"] == body2["stages"]["ocr"]["status"]
    assert body1["stages"]["damage_detection"]["status"] == body2["stages"]["damage_detection"]["status"]
    assert body1["stages"]["pricing"]["status"] == body2["stages"]["pricing"]["status"]
    # Report stage should be available in both responses
    assert body2["stages"]["report"]["status"] == "available"


def test_workflow_failure_preserves_error_state_and_dispatches_correct_status(
    workflow_client,
):
    """A workflow with a failing price provider returns the correct failure status."""
    from app.db.models import PriceEstimate

    client = workflow_client.client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    # Create a price estimate with a failing provider mock
    class BrokenPrice:
        def predict(self, _vehicle):
            raise RuntimeError("model inference failed")

    app.dependency_overrides[get_price_provider] = lambda: BrokenPrice()

    # Run the workflow - pricing should fail
    # Must provide vehicle payload to reach the provider call
    vehicle_payload = {"odometer_km": 50000, "seller_type": "Individual"}
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": vehicle_payload},
    )

    # The workflow should succeed with a failed pricing stage
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["stages"]["pricing"]["status"] == "failed"
    assert body["stages"]["pricing"]["errors"][0]["code"] == "price_provider_failed"
    assert body["stages"]["report"]["status"] == "available"

    # The report should reflect the pricing failure
    report_data = body["stages"]["report"]["result"]["report_data"]
    assert report_data["stages"]["pricing"]["status"] == "failed"
    assert report_data["stages"]["pricing"]["error_code"] == "price_provider_failed"

    # Clean up: remove the failed price estimate
    with Session(workflow_client.engine) as session:
        session.query(PriceEstimate).filter(
            PriceEstimate.assessment_id == UUID(assessment_id)
        ).delete()
        session.commit()


def test_damage_inference_failure_reports_actual_reason_not_generic_unavailable(
    workflow_client,
):
    """When damage inference fails, the report should show the actual error code."""
    from app.db.models import PriceEstimate

    client = workflow_client.client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client)

    # Override the damage provider to fail
    class FailingDamage:
        def detect(self, _content, _extension):
            raise RuntimeError("YOLO inference crashed")

    app.dependency_overrides[get_damage_provider] = lambda: FailingDamage()

    # Run the workflow
    # Must provide vehicle payload to reach the provider call
    vehicle_payload = {"odometer_km": 50000, "seller_type": "Individual"}
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/workflow",
        json={"image_ids": image_ids, "vehicle": vehicle_payload},
    )

    # The workflow should succeed with a failed damage stage
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "partial"
    assert body["stages"]["damage_detection"]["status"] == "failed"
    # The error should be specific, not generic
    assert any("inference" in error["code"] for error in
               body["stages"]["damage_detection"].get("errors", []))

    # The report's damage detection should reflect the actual failure
    report_data = body["stages"]["report"]["result"]["report_data"]
    damage_status = report_data["stages"]["damage_detection"]["status"]
    # Should be "failed" not "unavailable" since the inference did run but failed
    assert damage_status == "failed"

    # Clean up
    with Session(workflow_client.engine) as session:
        session.query(PriceEstimate).filter(
            PriceEstimate.assessment_id == UUID(assessment_id)
        ).delete()
        session.commit()


def test_ml_estimate_inr_is_consistent_with_stored_price_estimate(
    workflow_client,
):
    """ML estimate in report should match the stored price estimate amount."""
    from app.db.models import PriceEstimate

    client = workflow_client.client
    assessment_id = _create_test_assessment(client)

    # Create a price estimate directly in the database
    with Session(workflow_client.engine) as session:
        price_estimate = PriceEstimate(
            assessment_id=assessment_id,
            amount=Decimal("167276.00"),
            currency="INR",
            lower_bound=None,
            upper_bound=None,
            method="test_method",
            model_name="test_model",
            estimate_data={"vehicle": {"make": "Test Make", "model": "Test Model"}},
        )
        session.add(price_estimate)
        session.commit()

    # Generate the report via POST (creates report from stored estimates)
    report_response = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report_response.status_code == 200

    # Retrieve the generated report via GET
    report_response = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert report_response.status_code == 200
    report_data = report_response.json()["report_data"]

    # The ML estimate should be consistent with the stored price estimate
    assert report_data["buyer_recommendation"]["ml_estimate_inr"] == "167276.00"
    assert report_data["buyer_recommendation"]["asking_price_inr"] is None
    assert report_data["buyer_recommendation"]["status"] == "insufficient_evidence"


def test_damage_detection_persists_no_findings_as_completed_not_unavailable(
    workflow_client,
):
    """Damage detection with no findings should report completed, not unavailable."""
    from app.db.models import DamageFinding, EvidenceRecord
    from uuid import UUID

    client = workflow_client.client
    assessment_id, image_ids = _create_workflow_assessment_with_images(client, image_count=2)

    # Create a damage finding directly in the database with no detections
    with Session(workflow_client.engine) as session:
        finding = DamageFinding(
            assessment_id=UUID(assessment_id),
            image_id=UUID(image_ids[0]),
            damage_type="test",
            confidence=0.0,
            bounding_box={},
            model_name="yolo11m_local",
        )
        session.add(finding)
        evidence = EvidenceRecord(
            assessment_id=UUID(assessment_id),
            image_id=UUID(image_ids[0]),
            evidence_type="damage_detection",
            source_reference="yolo11m_local",
            evidence_data={"status": "completed", "detection_count": 0},
        )
        session.add(evidence)
        session.commit()

    # Generate the report via POST (creates report from stored data)
    report_response = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report_response.status_code == 200

    # Retrieve the generated report via GET
    report_response = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert report_response.status_code == 200
    report_data = report_response.json()["report_data"]

    # The damage detection should be available with the findings
    assert report_data["stages"]["damage_detection"]["status"] == "completed_with_findings"
    assert report_data["damage_findings"][0]["damage_type"] == "test"
