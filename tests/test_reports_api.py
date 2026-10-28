from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import (
    AssessmentReport,
    DamageFinding,
    OCRResult,
    PriceEstimate,
    UploadedImage,
    VehicleAssessment,
    VehicleAssessment,
)
from app.db.session import get_db
from app.main import app
from app.api.v1.reports import _buyer_recommendation
from app.pricing.market import MarketListing, get_market_listing_provider


@pytest.fixture
def report_api() -> Generator[tuple[TestClient, object, str], None, None]:
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
        assessment = VehicleAssessment(
            vin="TEST1234567890123",
            registration_number="KA01AB1234",
            vehicle_details={"make": "Example Motors", "model": "City"},
        )
        session.add(assessment)
        session.commit()
        assessment_id = str(assessment.id)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client, engine, assessment_id
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def _add_pipeline_results(engine, assessment_id: str) -> None:
    with Session(engine) as session:
        assessment_uuid = UUID(assessment_id)
        image = UploadedImage(
            assessment_id=assessment_uuid,
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key="test-image-key",
            original_filename="car.png",
            content_type="image/png",
            byte_size=123,
            sha256="test-digest",
        )
        session.add(image)
        session.flush()
        session.add_all(
            [
                OCRResult(
                    assessment_id=assessment_uuid,
                    image_id=image.id,
                    result_type="image_ocr",
                    status="completed",
                    provider="test-ocr",
                    raw_text="RJ45CA1879",
                    result_data={"confidence": 0.99},
                ),
                DamageFinding(
                    assessment_id=assessment_uuid,
                    image_id=image.id,
                    damage_type="dent",
                    confidence=0.91,
                    bounding_box={"x1": 1, "y1": 2, "x2": 5, "y2": 6},
                    model_name="test-damage-model",
                ),
                PriceEstimate(
                    assessment_id=assessment_uuid,
                    amount=Decimal("625000.00"),
                    currency="INR",
                    method="test-provider",
                    model_name="test-price-model",
                ),
            ]
        )
        session.commit()


def test_generate_report_gathers_and_persists_available_results(report_api):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("700000.00")
        session.commit()

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    body = response.json()
    assert body["assessment_id"] == assessment_id
    assert body["status"] == "draft"
    assert body["report_data"]["assessment"]["vehicle_details"]["make"] == "Example Motors"
    assert body["report_data"]["assessment"]["asking_price_inr"] == "700000.00"
    assert "asking_price_inr" not in body["report_data"]["assessment"]["vehicle_details"]
    assert body["report_data"]["stages"]["ocr"]["status"] == "available"
    assert body["report_data"]["stages"]["damage_detection"]["status"] == "results_available"
    assert body["report_data"]["stages"]["pricing"]["status"] == "available"
    assert body["report_data"]["ocr_results"][0]["text"] == "RJ45CA1879"
    assert body["report_data"]["ocr_results"][0]["confidence"] == pytest.approx(0.99)
    assert body["report_data"]["damage_findings"][0]["damage_type"] == "dent"
    assert body["report_data"]["price_estimates"][0]["amount"] == "625000.00"
    assert "asking_price_inr" not in body["report_data"]["price_estimates"][0]

    with Session(engine) as session:
        report = session.scalar(select(AssessmentReport))
        assert report is not None
        assert str(report.id) == body["id"]
        assert report.report_data == body["report_data"]


@pytest.mark.parametrize(
    ("asking_price", "expected_status", "expected_comparison", "expected_percent"),
    [
        ("500000", "negotiate", "asking_below_estimate", "20.00"),
        ("650000", "reasonably_priced", "within_tolerance", "4.00"),
        ("700000", "negotiate", "asking_above_estimate", "12.00"),
    ],
)
def test_buyer_recommendation_price_comparison(report_api, asking_price, expected_status, expected_comparison, expected_percent):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        session.query(DamageFinding).filter_by(assessment_id=UUID(assessment_id)).delete()
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal(asking_price)
        session.commit()

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    recommendation = response.json()["report_data"]["buyer_recommendation"]
    assert recommendation["status"] == expected_status
    assert recommendation["price_comparison"] == expected_comparison
    assert recommendation["absolute_difference_inr"] == str(abs(int(asking_price) - 625000)) + ".00"
    assert recommendation["difference_percent_of_ml_estimate"] == expected_percent
    assert recommendation["comparison_tolerance_percent"] == 5
    assert recommendation["comparison_basis"] == "ml_estimate_only"


def test_buyer_recommendation_insufficient_without_asking_or_ml_and_market_unavailable(report_api):
    client, _engine, assessment_id = report_api

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    recommendation = response.json()["report_data"]["buyer_recommendation"]
    assert recommendation["status"] == "insufficient_evidence"
    assert recommendation["asking_price_inr"] is None
    assert recommendation["ml_estimate_inr"] is None
    assert recommendation["absolute_difference_inr"] is None
    assert recommendation["difference_percent_of_ml_estimate"] is None
    assert recommendation["market_evidence_available"] is False
    assert any("Live comparable-market evidence was not available" in item for item in recommendation["limitations"])


def test_buyer_recommendation_damage_takes_priority_and_refresh_persists(report_api):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("625000")
        session.commit()
    path = f"/api/v1/assessments/{assessment_id}/report"

    first = client.post(path)
    refreshed = client.post(path)
    retrieved = client.get(path)

    recommendation = first.json()["report_data"]["buyer_recommendation"]
    assert recommendation["status"] == "further_inspection"
    assert recommendation["absolute_difference_inr"] == "0.00"
    assert "no repair cost" in recommendation["explanation"]
    assert refreshed.json()["report_data"]["buyer_recommendation"] == recommendation
    assert retrieved.json()["report_data"]["buyer_recommendation"] == recommendation


def test_buyer_recommendation_handles_missing_asking_with_prediction(report_api):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        session.query(DamageFinding).filter_by(assessment_id=UUID(assessment_id)).delete()
        session.commit()

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    recommendation = response.json()["report_data"]["buyer_recommendation"]
    assert recommendation["status"] == "insufficient_evidence"
    assert recommendation["asking_price_inr"] is None
    assert recommendation["ml_estimate_inr"] == "625000.00"


def test_buyer_recommendation_handles_missing_ml_prediction_and_configurable_threshold(report_api, monkeypatch):
    client, engine, assessment_id = report_api
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("700000")
        session.commit()

    missing_model = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert missing_model.status_code == 200
    assert missing_model.json()["report_data"]["buyer_recommendation"]["status"] == "insufficient_evidence"

    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        session.query(DamageFinding).filter_by(assessment_id=UUID(assessment_id)).delete()
        session.commit()
    monkeypatch.setattr("app.api.v1.reports.settings.buyer_price_near_threshold_percent", 15)

    within_configured_tolerance = client.post(f"/api/v1/assessments/{assessment_id}/report")
    recommendation = within_configured_tolerance.json()["report_data"]["buyer_recommendation"]
    assert recommendation["status"] == "reasonably_priced"
    assert recommendation["comparison_tolerance_percent"] == 15


def test_buyer_recommendation_rejects_invalid_ml_estimate_safely():
    class InvalidEstimate:
        amount = Decimal("NaN")

    recommendation = _buyer_recommendation(
        Decimal("700000"), InvalidEstimate(), [], "unavailable", {"status": "unavailable"}
    )

    assert recommendation["status"] == "insufficient_evidence"
    assert recommendation["ml_estimate_inr"] is None
    assert recommendation["absolute_difference_inr"] is None


def test_missing_assessment_returns_404(report_api):
    client, _engine, _assessment_id = report_api

    response = client.post(f"/api/v1/assessments/{uuid4()}/report")

    assert response.status_code == 404
    assert response.json()["detail"] == "Assessment not found."


def test_incomplete_results_are_reported_as_pending_or_unavailable(report_api):
    client, _engine, assessment_id = report_api

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    data = response.json()["report_data"]
    assert response.json()["status"] == "draft"
    assert data["stages"] == {
        "ocr": {"status": "unavailable", "result_count": 0, "completed_image_count": 0},
        "damage_detection": {
            "status": "unavailable",
            "finding_count": 0,
            "images_with_findings": 0,
        },
        "pricing": {"status": "pending", "estimate_count": 0},
    }
    assert data["ocr_results"] == []
    assert data["damage_findings"] == []
    assert data["price_estimates"] == []
    assert "cannot distinguish" in data["limitations"][0]
    assert data["market_valuation"]["status"] == "unavailable"
    assert data["market_valuation"]["market_asking_price_range"] is None
    assert data["market_valuation"]["evidence_count"] == 0


def test_report_retrieval_returns_existing_snapshot_and_404_when_absent(report_api):
    client, _engine, assessment_id = report_api
    path = f"/api/v1/assessments/{assessment_id}/report"

    missing = client.get(path)
    generated = client.post(path)
    retrieved = client.get(path)

    assert missing.status_code == 404
    assert generated.status_code == 200
    assert retrieved.status_code == 200
    assert retrieved.json() == generated.json()


def test_regeneration_refreshes_one_report_without_duplicates(report_api):
    client, engine, assessment_id = report_api
    path = f"/api/v1/assessments/{assessment_id}/report"
    first = client.post(path)
    _add_pipeline_results(engine, assessment_id)

    refreshed = client.post(path)

    assert first.status_code == refreshed.status_code == 200
    assert refreshed.json()["id"] == first.json()["id"]
    assert refreshed.json()["report_data"]["stages"]["pricing"]["status"] == "available"
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(AssessmentReport)) == 1


class FakeMarketProvider:
    provider_name = "test-only-market-feed"

    def __init__(self):
        self.calls = []

    def fetch_listings(self, criteria):
        self.calls.append(criteria)
        now = datetime.now(timezone.utc)
        return [
            MarketListing(
                source_name=self.provider_name,
                listing_id=f"test-listing-{index}",
                make=criteria.make,
                model=criteria.model,
                variant=criteria.variant,
                year=criteria.year or 2021,
                fuel_type=criteria.fuel_type,
                transmission=criteria.transmission,
                mileage_km=criteria.mileage_km,
                location=criteria.location,
                advertised_price_inr=Decimal(780_000 + index * 10_000),
                source_url=f"https://market.example/listing/{index}",
                listed_at=now - timedelta(days=2),
                retrieved_at=now,
            )
            for index in range(5)
        ]


def test_report_integrates_market_evidence_separately_from_asking_and_ml(report_api):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.asking_price_inr = Decimal("900000.00")
        session.commit()
    provider = FakeMarketProvider()
    app.dependency_overrides[get_market_listing_provider] = lambda: provider

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")
    body = response.json()
    market = body["report_data"]["market_valuation"]

    assert response.status_code == 200
    assert len(provider.calls) == 1
    assert market["status"] == "available"
    assert market["data_origin"] == "freshly_retrieved"
    assert market["seller_asking_price_inr"] == "900000.00"
    assert market["ml_prediction"]["amount_inr"] == "625000.00"
    assert market["market_asking_price_range"]["lower_inr"] == "790000.00"
    assert market["market_asking_price_range"]["upper_inr"] == "810000.00"
    assert market["evidence_count"] == 5
    assert market["discrepancy"]["status"] == "ml_below_market_asking_range"
    assert market["discrepancy"]["combined_estimate"] is None
    assert len(market["comparables"]) == 5
    assert "asking_price_inr" not in market["ml_prediction"]

    retrieved = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert retrieved.status_code == 200
    assert retrieved.json()["report_data"]["market_valuation"] == market


def test_market_provider_failure_preserves_prior_market_snapshot(report_api):
    client, _engine, assessment_id = report_api
    first_provider = FakeMarketProvider()
    app.dependency_overrides[get_market_listing_provider] = lambda: first_provider
    first = client.post(f"/api/v1/assessments/{assessment_id}/report")
    previous = first.json()["report_data"]["market_valuation"]

    class BrokenMarketProvider:
        def fetch_listings(self, _criteria):
            raise TimeoutError("private provider error")

    app.dependency_overrides[get_market_listing_provider] = lambda: BrokenMarketProvider()
    refreshed = client.post(f"/api/v1/assessments/{assessment_id}/report")
    market = refreshed.json()["report_data"]["market_valuation"]

    assert refreshed.status_code == 200
    assert market["status"] == previous["status"]
    assert market["market_asking_price_range"] == previous["market_asking_price_range"]
    assert market["comparables"] == previous["comparables"]
    assert market["data_origin"] == "stored_previous"
    assert market["refresh_status"] == "refresh_failed"

