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
from app.api.v1.reports import (
    _buyer_recommendation,
    _market_inputs,
    _valid_market_source_url,
)
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
                    estimate_data={
                        "vehicle": {
                            "make": "Example Motors",
                            "odometer_km": 42000,
                            "mileage_kmpl": 15.0,
                            "owner_count": 1,
                        },
                        "provenance": {
                            "make": "vehicle_api",
                            "odometer_km": "user_input",
                            "mileage_kmpl": "generic_fallback",
                            "owner_count": "generic_fallback",
                        },
                    },
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
    assert body["report_data"]["assessment"]["vehicle_details_provenance"]["make"] == "user_input"
    assert body["report_data"]["assessment"]["asking_price_inr"] == "700000.00"
    assert body["report_data"]["assessment"]["asking_price_inr_provenance"] == "user_input"
    assert "asking_price_inr" not in body["report_data"]["assessment"]["vehicle_details"]
    assert body["report_data"]["stages"]["ocr"]["status"] == "available"
    assert body["report_data"]["stages"]["damage_detection"]["status"] == "completed_with_findings"
    assert body["report_data"]["stages"]["pricing"]["status"] == "available"
    assert body["report_data"]["ocr_results"][0]["text"] == "RJ45CA1879"
    assert body["report_data"]["ocr_results"][0]["confidence"] == pytest.approx(0.99)
    registration_ocr = body["report_data"]["registration_ocr_assessment"]
    assert registration_ocr["status"] == "candidate"
    assert registration_ocr["normalized_candidate"] == "RJ45CA1879"
    assert registration_ocr["raw_evidence"][0]["text"] == "RJ45CA1879"
    assert registration_ocr["raw_evidence"][0]["confidence"] == pytest.approx(0.99)
    assert body["report_data"]["damage_findings"][0]["damage_type"] == "dent"
    assert body["report_data"]["stages"]["damage_detection"]["image_results"][0]["status"] == "completed_with_findings"
    assert body["report_data"]["price_estimates"][0]["amount"] == "625000.00"
    assert body["report_data"]["price_estimates"][0]["vehicle"] == {
        "make": "Example Motors",
        "odometer_km": 42000,
    }
    assert body["report_data"]["price_estimates"][0]["provenance"] == {
        "make": "vehicle_api",
        "odometer_km": "user_input",
    }
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


def test_buyer_recommendation_discloses_research_without_price_range():
    recommendation = _buyer_recommendation(
        Decimal("900000"),
        None,
        [],
        "unavailable",
        {
            "status": "unavailable",
            "relevant_result_count": 2,
            "market_asking_price_range": None,
        },
    )

    assert any(
        "Relevant web-search market research is available" in limitation
        for limitation in recommendation["limitations"]
    )


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
            "excluded_image_count": 0,
            "excluded_images": [],
            "completed_image_count": 0,
            "unavailable_count": 0,
            "failed_count": 0,
            "image_results": [],
        },
        "pricing": {"status": "pending", "estimate_count": 0, "input_provenance": {}},
    }
    assert data["ocr_results"] == []
    assert data["damage_findings"] == []
    assert data["price_estimates"] == []
    assert data["stages"]["damage_detection"]["status"] == "unavailable"
    assert "cannot distinguish" in data["limitations"][0]
    assert data["market_valuation"]["status"] == "unavailable"
    assert data["market_valuation"]["market_asking_price_range"] is None
    assert data["market_valuation"]["unavailable_reason"] == (
        "Tavily web search is not configured. Set TAVILY_API_KEY in the backend environment."
    )
    assert data["market_valuation"]["evidence_count"] == 0
    assert data["vehicle_details_lookup"]["status"] == "unavailable"
    assert data["challan_lookup"]["status"] == "unavailable"
    assert data["fastag_lookup"]["status"] == "unavailable"


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


def test_report_retrieval_normalizes_legacy_damage_status_and_narrative(report_api):
    client, engine, assessment_id = report_api
    with Session(engine) as session:
        report = AssessmentReport(
            assessment_id=UUID(assessment_id),
            status="draft",
            report_data={
                "assessment": {"id": assessment_id},
                "stages": {
                    "damage_detection": {
                        "status": "results_available",
                        "finding_count": 1,
                    }
                },
                "damage_findings": [{"damage_type": "dent"}],
                "llm_narrative": {
                    "detected_damage_explanation": "Old narrative.",
                },
            },
        )
        session.add(report)
        session.commit()

    retrieved = client.get(f"/api/v1/assessments/{assessment_id}/report")

    assert retrieved.status_code == 200
    report_data = retrieved.json()["report_data"]
    assert report_data["stages"]["damage_detection"]["status"] == "completed_with_findings"
    assert report_data["llm_narrative"]["detected_damage_explanation"] == (
        "The model returned damage findings in the supplied images: dent. "
        "These are model-generated findings, not confirmed physical damage."
    )


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
                fuel_type=criteria.fuel_type or "petrol",
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


def test_report_market_search_uses_persisted_vehicle_lookup_when_snapshot_fields_are_missing(
    report_api,
):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.vehicle_details = {}
        estimate = session.scalar(
            select(PriceEstimate).where(
                PriceEstimate.assessment_id == UUID(assessment_id)
            )
        )
        estimate.estimate_data = {
            "vehicle": {
                "odometer_km": 42000,
                "mileage_kmpl": 15,
                "owner_count": 1,
            },
            "provenance": {},
        }
        session.add(
            OCRResult(
                assessment_id=UUID(assessment_id),
                result_type="registration_lookup",
                status="completed",
                provider="test-vehicle-lookup",
                result_data={
                    "make": "Hyundai",
                    "model": "i10",
                    "variant": "GLS Sportz",
                    "fuel_type": "petrol",
                    "transmission": "manual",
                    "registration_date": "2013-04-01",
                },
            )
        )
        session.commit()
    provider = FakeMarketProvider()
    app.dependency_overrides[get_market_listing_provider] = lambda: provider

    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    assert len(provider.calls) == 1
    assert provider.calls[0].make == "Hyundai"
    assert provider.calls[0].model == "i10"
    assert provider.calls[0].year == 2013
    market = response.json()["report_data"]["market_valuation"]
    assert market["vehicle_researched"] == {"make": "Hyundai", "model": "i10"}


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
    assert market["unavailable_reason"] is None
    assert market["data_origin"] == "freshly_retrieved"
    assert market["seller_asking_price_inr"] == "900000.00"
    assert market["ml_prediction"]["amount_inr"] == "625000.00"
    assert market["market_asking_price_range"]["lower_inr"] == "790000.00"
    assert market["market_asking_price_range"]["upper_inr"] == "810000.00"
    assert market["evidence_count"] == 5
    assert market["exact_match_count"] == 0
    assert market["close_match_count"] == 0
    assert market["approximate_match_count"] == 5
    assert market["search_observability"]["final_approximate_count"] == 5
    assert market["market_asking_price_range"]["median_inr"] == "800000.00"
    assert market["market_asking_price_range"]["median_inr"] != market["ml_prediction"]["amount_inr"]
    assert market["discrepancy"]["status"] == "ml_below_market_asking_range"
    assert market["discrepancy"]["combined_estimate"] is None
    assert len(market["comparables"]) == 5
    assert "asking_price_inr" not in market["ml_prediction"]

    retrieved = client.get(f"/api/v1/assessments/{assessment_id}/report")
    assert retrieved.status_code == 200
    assert retrieved.json()["report_data"]["market_valuation"] == market


@pytest.mark.parametrize(
    ("provider_listings", "expected_rejections"),
    [
        ([], {}),
        ([{"source_name": "broken", "advertised_price_inr": "not-a-price"}], {"malformed_listing": 1}),
    ],
)
def test_market_provider_without_valid_listings_returns_unavailable(
    report_api, provider_listings, expected_rejections
):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)

    class EmptyMarketProvider:
        def fetch_listings(self, _criteria):
            return provider_listings

    app.dependency_overrides[get_market_listing_provider] = lambda: EmptyMarketProvider()
    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    report = response.json()["report_data"]
    market = report["market_valuation"]
    assert market["status"] == "unavailable"
    assert market["unavailable_reason"] == "The provider returned no valid comparable listings."
    assert market["market_asking_price_range"] is None
    assert market["comparables"] == []
    assert market["rejected_listing_counts"] == expected_rejections
    assert report["price_estimates"][-1]["amount"] == "625000.00"
    assert market["ml_prediction"]["amount_inr"] == "625000.00"


def test_tavily_market_response_contains_only_search_evidence_and_no_api_key(
    report_api, caplog
):
    import httpx

    from app.market.tavily import TavilyMarketListingProvider

    caplog.set_level("INFO")
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.vehicle_details = {"make": "Example Motors", "model": "City"}
        session.commit()

    def handler(_request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": f"Example Motors City {2021} used car listing {index}",
                        "url": f"https://cars24.com/city/{index}",
                        "content": f"Petrol asking price Rs {700000 + index * 10000:,}",
                    }
                    for index in range(5)
                ]
                + [
                    {
                        "title": "Example Motors City 2021 used car listing",
                        "url": "https://cars24.com/city/no-price",
                        "content": "Contact the seller for price.",
                    }
                ]
            },
        )

    provider = TavilyMarketListingProvider(
        "report-secret",
        transport=httpx.MockTransport(handler),
    )
    app.dependency_overrides[get_market_listing_provider] = lambda: provider
    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    market = response.json()["report_data"]["market_valuation"]
    assert market["status"] == "available"
    assert market["evidence_count"] == 5
    assert market["search_observability"]["queries_issued"] == 1
    assert market["search_observability"]["raw_results_returned"] == 6
    assert market["search_observability"]["market_research_rejected"] == 1
    assert market["market_research_rejection_counts"] == {"missing_price": 1}
    assert market["priced_comparable_count"] == 5
    assert market["relevant_result_count"] == 5
    assert len(market["priced_comparables"]) == 5
    assert market["market_research_results"][0]["source_url"] == (
        "https://cars24.com/city/0"
    )
    assert market["market_asking_price_range"] is not None
    assert all(
        item["source_url"].startswith("https://cars24.com/")
        for item in market["comparables"]
    )
    assert all(item["listed_at"] is None for item in market["comparables"])
    assert "report-secret" not in response.text
    assert "report-secret" not in caplog.text
    snapshot_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "market_report_snapshot_results"
    )
    assert snapshot_record.report_market_research_count == 5
    assert snapshot_record.report_priced_comparable_count == 5
    assert snapshot_record.market_queries_issued == 1
    assert snapshot_record.market_raw_results_returned == 6
    assert snapshot_record.market_research_rejection_counts == {"missing_price": 1}
    assert '"event":"market_report_snapshot_results"' in caplog.text
    assert '"report_serialized_priced_listing_count":5' in caplog.text


def test_raw_content_prices_without_year_or_variant_reach_saved_report(report_api):
    import httpx

    from app.market.tavily import TavilyMarketListingProvider

    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.vehicle_details = {
            "make": "Hyundai",
            "model": "i10",
            "registration_year": 2013,
            "variant": "GLS Sportz",
            "fuel_type": "petrol",
            "city": "Jaipur",
        }
        estimate = session.scalar(
            select(PriceEstimate).where(
                PriceEstimate.assessment_id == UUID(assessment_id)
            )
        )
        estimate_data = dict(estimate.estimate_data)
        estimate_data["vehicle"] = {
            **estimate_data.get("vehicle", {}),
            "make": "Hyundai",
            "model": "i10",
        }
        estimate.estimate_data = estimate_data
        session.commit()

    results = [
        {
            "title": "Hyundai i10 used car listing",
            "url": f"https://www.olx.in/item/hyundai-i10-iid-{index}",
            "content": "Used car available in Jaipur.",
            "raw_content": (
                f"Hyundai i10. Asking price INR {200000 + index * 10000:,}. "
                "EMI starts at Rs 6,500 per month."
            ),
        }
        for index in range(5)
    ]

    def handler(_request):
        return httpx.Response(200, json={"results": results})

    provider = TavilyMarketListingProvider(
        "report-secret",
        transport=httpx.MockTransport(handler),
    )
    app.dependency_overrides[get_market_listing_provider] = lambda: provider
    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    market = response.json()["report_data"]["market_valuation"]
    assert provider.search_observability["queries_issued"] <= 2
    assert market["status"] == "unavailable"
    assert market["market_asking_price_range"] is None
    assert market["relevant_result_count"] == 5
    assert market["priced_comparable_count"] == 5
    assert len(market["market_research_results"]) == 5
    assert [
        item["asking_price_inr"] for item in market["market_research_results"]
    ] == [f"{200000 + index * 10000:.2f}" for index in range(5)]
    assert all(item["year"] is None for item in market["market_research_results"])
    assert all(item["variant"] is None for item in market["market_research_results"])
    assert all(
        item["source_url"].startswith("https://www.olx.in/item/")
        for item in market["market_research_results"]
    )
    assert market["evidence_count"] == 0


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


def test_market_inputs_propagate_city_and_state_to_search_criteria(report_api):
    _client, engine, assessment_id = report_api
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.vehicle_details = {
            "make": "Hyundai",
            "model": "i10",
            "city": "Jaipur",
            "state": "Rajasthan",
        }
        session.flush()

        criteria, _asking_price, _ml_prediction = _market_inputs(assessment, [])

    assert criteria is not None
    assert criteria.location == "Jaipur, Rajasthan"


@pytest.mark.parametrize(
    ("source_url", "expected"),
    [
        ("https://cars.example/listing/1", True),
        ("http://cars.example/article", True),
        ("javascript:alert(1)", False),
        ("//cars.example/listing/1", False),
        ("https://user:pass@cars.example/listing/1", False),
        ("not a url", False),
        (None, False),
    ],
)
def test_market_source_url_validation(source_url, expected):
    assert _valid_market_source_url(source_url) is expected


@pytest.mark.parametrize(
    ("result_count", "priced_count"),
    [(0, 0), (1, 1), (3, 2)],
)
def test_market_report_preserves_context_with_zero_or_limited_priced_results(
    report_api, result_count, priced_count
):
    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    with Session(engine) as session:
        assessment = session.get(VehicleAssessment, UUID(assessment_id))
        assessment.vehicle_details = {
            "make": "Hyundai",
            "model": "i10",
            "city": "Jaipur",
            "state": "Rajasthan",
        }
        session.commit()

    class ResearchProvider:
        search_observability = {
            "queries_issued": 3,
            "raw_results_returned": result_count,
            "listings_extracted": priced_count,
        }
        search_scopes = ["Jaipur, Rajasthan and nearby regions"]

        def __init__(self):
            self.market_research_results = [
                {
                    "source_url": f"https://market.example/i10/{index}",
                    "year": 2013 if index <= priced_count else None,
                    "variant": "Sportz" if index <= priced_count else None,
                    "asking_price_inr": "250000.00" if index <= priced_count else None,
                    "priced_used_listing": index <= priced_count,
                }
                for index in range(1, priced_count + 1)
            ]

        def fetch_listings(self, _criteria):
            return []

    app.dependency_overrides[get_market_listing_provider] = lambda: ResearchProvider()
    response = client.post(f"/api/v1/assessments/{assessment_id}/report")

    assert response.status_code == 200
    market = response.json()["report_data"]["market_valuation"]
    assert market["status"] == "unavailable"
    assert market["vehicle_researched"] == {"make": "Example Motors", "model": "i10"}
    assert market["search_location"] == "Jaipur, Rajasthan"
    assert market["geographic_scope"] == ["Jaipur, Rajasthan and nearby regions"]
    assert market["relevant_result_count"] == priced_count
    assert market["priced_comparable_count"] == priced_count
    assert len(market["priced_comparables"]) == priced_count
    assert len(market["market_research_results"]) == priced_count
    assert all(
        set(item) == {"source_url", "asking_price_inr", "variant", "year"}
        for item in market["market_research_results"]
    )
    assert market["market_asking_price_range"] is None
    if result_count:
        assert "Relevant market research was found" in market["unavailable_reason"]
    else:
        assert market["unavailable_reason"] == (
            "The provider returned no valid comparable listings."
        )


def test_report_generation_retries_missing_narrative(monkeypatch, report_api):
    from app.llm.schemas import LLMReportNarrative

    client, engine, assessment_id = report_api
    _add_pipeline_results(engine, assessment_id)
    fake_first = LLMReportNarrative(
        vehicle_overview="first",
        overall_condition_summary="first",
        detected_damage_explanation="first",
        potential_inspection_risks="first",
        buyer_recommendations="first",
        price_estimate_explanation="first",
        missing_information_and_limitations="first",
    )
    monkeypatch.setattr("app.api.v1.reports.generate_llm_narrative", lambda _data: fake_first)

    first = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert first.status_code == 200
    assert first.json()["report_data"]["llm_narrative"]["vehicle_overview"] == "first"

    with Session(engine) as session:
        report = session.scalar(select(AssessmentReport).where(AssessmentReport.assessment_id == UUID(assessment_id)))
        report.report_data = {**report.report_data, "llm_narrative": None}
        session.commit()

    fake_second = LLMReportNarrative(
        vehicle_overview="second",
        overall_condition_summary="second",
        detected_damage_explanation="second",
        potential_inspection_risks="second",
        buyer_recommendations="second",
        price_estimate_explanation="second",
        missing_information_and_limitations="second",
    )
    monkeypatch.setattr("app.api.v1.reports.generate_llm_narrative", lambda _data: fake_second)

    second = client.post(f"/api/v1/assessments/{assessment_id}/report")
    assert second.status_code == 200
    assert second.json()["report_data"]["llm_narrative"]["vehicle_overview"] == "second"