from collections.abc import Generator
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import PriceEstimate, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.pricing.provider import PipelinePriceProvider, get_price_provider
from app.schemas.pricing import PricePrediction, VehiclePriceInput


VALID_REQUEST = {
    "vehicle": {
        "make": "Maruti Suzuki",
        "model": "Swift",
        "variant": "VXi",
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
}


class FakePriceProvider:
    def __init__(self):
        self.received = None

    def predict(self, vehicle: VehiclePriceInput) -> PricePrediction:
        self.received = vehicle
        return PricePrediction(
            amount=Decimal("625000.00"),
            currency="INR",
            lower_bound=Decimal("600000.00"),
            upper_bound=Decimal("650000.00"),
            method="mock test provider",
            model_name="test-only-fake",
        )


@pytest.fixture
def pricing_api() -> Generator[tuple[TestClient, object], None, None]:
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
        session.add(assessment)
        session.commit()
        assessment_id = str(assessment.id)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_price_provider] = lambda: None
    try:
        with TestClient(app) as client:
            yield client, (engine, assessment_id)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_price_request_rejects_invalid_or_missing_vehicle_fields(pricing_api):
    client, (_engine, assessment_id) = pricing_api

    missing = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json={"vehicle": {}})
    invalid = {**VALID_REQUEST, "vehicle": {**VALID_REQUEST["vehicle"], "odometer_km": -1}}
    bad_value = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=invalid)

    assert missing.status_code == 422
    assert bad_value.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [("seller_type", "broker"), ("mileage_kmpl", 0), ("engine_cc", 10001), ("max_power_bhp", -1), ("seats", 10)],
)
def test_price_request_validates_model_features(pricing_api, field, value):
    client, (_engine, assessment_id) = pricing_api
    invalid = {**VALID_REQUEST, "vehicle": {**VALID_REQUEST["vehicle"], field: value}}

    response = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=invalid)

    assert response.status_code == 422


def test_missing_assessment_returns_404(pricing_api):
    client, (_engine, _assessment_id) = pricing_api

    response = client.post(f"/api/v1/assessments/{uuid4()}/price-estimates", json=VALID_REQUEST)

    assert response.status_code == 404


def test_unconfigured_provider_returns_503_and_persists_nothing(pricing_api):
    client, (engine, assessment_id) = pricing_api

    response = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST)

    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]
    with Session(engine) as session:
        assert session.scalar(select(PriceEstimate)) is None


def test_fake_provider_result_is_persisted_and_returned(pricing_api):
    client, (engine, assessment_id) = pricing_api
    fake = FakePriceProvider()
    app.dependency_overrides[get_price_provider] = lambda: fake

    response = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST)

    assert response.status_code == 201
    body = response.json()
    assert body["amount"] == "625000.00"
    assert body["currency"] == "INR"
    assert body["model_name"] == "test-only-fake"
    assert body["assessment_id"] == assessment_id
    assert fake.received.make == "Maruti Suzuki"
    with Session(engine) as session:
        estimate = session.scalar(select(PriceEstimate))
        assert estimate is not None
        assert str(estimate.assessment_id) == assessment_id
        assert estimate.amount == Decimal("625000.00")
        assert estimate.currency == "INR"
        assert estimate.estimate_data["vehicle"]["odometer_km"] == 42000


def test_pricing_to_report_flow_keeps_ml_and_market_values_separate(pricing_api):
    client, (engine, assessment_id) = pricing_api
    fake = FakePriceProvider()
    app.dependency_overrides[get_price_provider] = lambda: fake

    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("675000.00")
        session.commit()

    estimate_response = client.post(
        f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST
    )
    report_response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert estimate_response.status_code == 201
    assert report_response.status_code == 200
    report = report_response.json()["report_data"]
    assert report["assessment"]["asking_price_inr"] == "675000.00"
    assert report["price_estimates"][-1]["amount"] == "625000.00"
    assert report["market_valuation"]["status"] == "unavailable"
    assert report["market_valuation"]["market_asking_price_range"] is None
    recommendation = report["buyer_recommendation"]
    assert recommendation["status"] == "negotiate"
    assert recommendation["comparison_basis"] == "ml_estimate_only"
    assert recommendation["asking_price_inr"] == "675000.00"
    assert recommendation["ml_estimate_inr"] == "625000.00"
    assert recommendation["absolute_difference_inr"] == "50000.00"
    assert recommendation["difference_percent_of_ml_estimate"] == "8.00"
    assert "not verified market value" in recommendation["explanation"]

    retrieved = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert retrieved.status_code == 200
    assert retrieved.json()["report_data"]["buyer_recommendation"] == recommendation


def test_pricing_to_report_flow_without_asking_price(pricing_api):
    client, (_engine, assessment_id) = pricing_api
    app.dependency_overrides[get_price_provider] = lambda: FakePriceProvider()

    estimate_response = client.post(
        f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST
    )
    report_response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert estimate_response.status_code == 201
    assert report_response.status_code == 200
    report = report_response.json()["report_data"]
    assert report["assessment"]["asking_price_inr"] is None
    assert report["price_estimates"][-1]["amount"] == "625000.00"
    assert report["market_valuation"]["status"] == "unavailable"
    recommendation = report["buyer_recommendation"]
    assert recommendation["status"] == "insufficient_evidence"
    assert recommendation["asking_price_inr"] is None
    assert recommendation["ml_estimate_inr"] == "625000.00"
    assert recommendation["absolute_difference_inr"] is None
    assert recommendation["difference_percent_of_ml_estimate"] is None


@pytest.mark.parametrize("artifact_state", ["missing", "invalid"])
def test_model_artifact_failure_returns_safe_502_without_persisting(
    pricing_api, tmp_path, artifact_state
):
    client, (engine, assessment_id) = pricing_api
    artifact = tmp_path / "price-model.joblib"
    if artifact_state == "invalid":
        artifact.write_bytes(b"not deserialized by this test")
        provider = PipelinePriceProvider(artifact, loader=lambda _path: object())
    else:
        provider = PipelinePriceProvider(artifact)
    app.dependency_overrides[get_price_provider] = lambda: provider

    response = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST)

    assert response.status_code == 502
    assert response.json()["detail"] == "The price provider did not return a valid estimate."
    with Session(engine) as session:
        assert session.scalar(select(PriceEstimate)) is None


def test_provider_failure_does_not_persist_estimate(pricing_api):
    client, (engine, assessment_id) = pricing_api

    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("650000.00")
        session.commit()

    class BrokenProvider:
        def predict(self, _vehicle):
            raise RuntimeError("sensitive internal provider detail")

    app.dependency_overrides[get_price_provider] = BrokenProvider

    response = client.post(f"/api/v1/assessments/{assessment_id}/price-estimates", json=VALID_REQUEST)

    assert response.status_code == 502
    assert "sensitive" not in response.text
    with Session(engine) as session:
        assert session.scalar(select(PriceEstimate)) is None

    report_response = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert report_response.status_code == 200
    report = report_response.json()["report_data"]
    assert report["assessment"]["asking_price_inr"] == "650000.00"
    assert report["price_estimates"] == []
    assert report["market_valuation"]["status"] == "unavailable"
    recommendation = report["buyer_recommendation"]
    assert recommendation["status"] == "insufficient_evidence"
    assert recommendation["asking_price_inr"] == "650000.00"
    assert recommendation["ml_estimate_inr"] is None
    assert recommendation["absolute_difference_inr"] is None
