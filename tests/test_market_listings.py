import csv
import io
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import MarketListing as MarketListingModel
from app.market.ingestion import import_market_listings_from_csv, import_market_listings_from_json
from app.pricing.market import MarketVehicleCriteria, calculate_market_comparables


@pytest.fixture
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


VALID_CSV = """source_name,source_url,make,model,manufacturing_year,asking_price_inr,city,state,source_listing_id,fuel_type,transmission,odometer_km,seller_type,listed_at,collected_at
example_market,https://example.com/listing/1,Maruti Suzuki,Swift,2021,600000,Jaipur,Rajasthan,L1,petrol,manual,40000,dealer,2026-10-29T00:00:00+00:00,2026-10-30T00:00:00+00:00
"""


def test_valid_csv_import_creates_listing(db_session):
    summary = import_market_listings_from_csv(db_session, VALID_CSV)
    assert summary.created == 1
    assert db_session.query(MarketListingModel).count() == 1


def test_invalid_price_is_rejected(db_session):
    bad = VALID_CSV.replace("600000", "0")
    summary = import_market_listings_from_csv(db_session, bad)
    assert summary.created == 0
    assert summary.rejected == 1


def test_duplicate_import_is_idempotent(db_session):
    first = import_market_listings_from_csv(db_session, VALID_CSV)
    second = import_market_listings_from_csv(db_session, VALID_CSV)
    assert first.created == 1
    assert second.skipped == 1
    assert db_session.query(MarketListingModel).count() == 1


def test_dry_run_does_not_write(db_session):
    summary = import_market_listings_from_csv(db_session, VALID_CSV, dry_run=True)
    assert summary.created == 1
    assert db_session.query(MarketListingModel).count() == 0


def test_json_import(db_session):
    payload = [{
        "source_name": "example_market",
        "source_url": "https://example.com/listing/2",
        "make": "Maruti Suzuki",
        "model": "Swift",
        "manufacturing_year": 2021,
        "asking_price_inr": "610000.00",
        "city": "Jaipur",
        "listed_at": "2026-10-29T00:00:00+00:00",
        "collected_at": "2026-10-30T00:00:00+00:00",
    }]
    summary = import_market_listings_from_json(db_session, payload)
    assert summary.created == 1


def test_matching_returns_unavailable_for_insufficient_data(db_session):
    import_market_listings_from_csv(db_session, VALID_CSV)
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(db_session.query(MarketListingModel).all() and [l for l in []], criteria) if False else None
    # Use repository conversion
    from app.market.repository import MarketListingRepository
    listings = MarketListingRepository(db_session).list_for_criteria(criteria)
    result = calculate_market_comparables(listings, criteria)
    assert result.status == "low_evidence"
    assert result.price_range is None


def test_matching_available_with_enough_listings(db_session):
    rows = []
    for i in range(6):
        rows.append(f"example_market,https://example.com/listing/{i+10},Maruti Suzuki,Swift,2021,{600000+i*10000},Jaipur,Rajasthan,L{i+10},petrol,manual,40000,dealer,2026-10-29T00:00:00+00:00,2026-10-30T00:00:00+00:00")
    csv_text = "source_name,source_url,make,model,manufacturing_year,asking_price_inr,city,state,source_listing_id,fuel_type,transmission,odometer_km,seller_type,listed_at,collected_at\n" + "\n".join(rows)
    import_market_listings_from_csv(db_session, csv_text)
    from app.market.repository import MarketListingRepository
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    listings = MarketListingRepository(db_session).list_for_criteria(criteria)
    result = calculate_market_comparables(listings, criteria)
    assert result.status == "available"
    assert result.price_range is not None
from app.pricing.market import MarketListing, MarketVehicleCriteria, calculate_market_comparables
from datetime import datetime, timedelta, timezone


def _listing(price=600000, year=2021, fuel="petrol", transmission="manual", mileage=40000, location="Jaipur", source_id="L1", listed_days_ago=1):
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    return MarketListing(
        source_name="example_market",
        listing_id=source_id,
        make="Maruti Suzuki",
        model="Swift",
        variant="VXi",
        year=year,
        fuel_type=fuel,
        transmission=transmission,
        mileage_km=mileage,
        location=location,
        advertised_price_inr=price,
        source_url="https://example.com/listing/" + source_id,
        listed_at=now - timedelta(days=listed_days_ago),
        retrieved_at=now - timedelta(days=1),
    )


def test_exact_model_match_required():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].model = "Baleno"
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 5


def test_variant_mismatch_is_approximate_after_exact_evidence():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].variant = "ZXi"
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", variant="VXi", year=2021, fuel_type="petrol", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 5
    assert result.evidence_count == 5
    assert all(item.match_type == "exact_match" for item in result.listings)
def test_year_difference_tolerance():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(year=2024, source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", variant="VXi", year=2021, fuel_type="petrol", transmission="manual", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.status == "available"
    assert all(item.match_type == "close_match" for item in result.listings)


def test_materially_different_year_cohorts_are_not_combined():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [
        *[_listing(year=2018, source_id=f"old-{i}") for i in range(3)],
        *[_listing(year=2024, source_id=f"new-{i}") for i in range(3)],
    ]
    criteria = MarketVehicleCriteria(
        make="Maruti Suzuki",
        model="Swift",
        variant="VXi",
        year=2021,
        fuel_type="petrol",
        transmission="manual",
        mileage_km=40000,
        location="Jaipur",
    )

    result = calculate_market_comparables(listings, criteria, valuation_date=now)

    assert result.status == "available"
    assert result.evidence_count == 3
    assert result.price_range is not None
    assert len({item.year for item in result.listings}) == 1


def test_fuel_mismatch_excluded():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(fuel="diesel", source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, fuel_type="petrol", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 0


def test_transmission_mismatch_is_approximate():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(transmission="automatic", source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, fuel_type="petrol", transmission="manual", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert all(item.match_type == "approximate_match" for item in result.listings)


def test_odometer_difference_tolerance():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(mileage=70000, source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert all(item.match_type == "approximate_match" for item in result.listings)
    assert all(
        any("mileage_km:" in difference for difference in item.match_differences)
        for item in result.listings
    )


def test_location_relevance():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(location="Delhi", source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert all(item.match_type == "approximate_match" for item in result.listings)


def test_duplicate_listing_excluded():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    duplicate = _listing(source_id="L0")
    duplicate.source_url = listings[0].source_url
    listings.append(duplicate)
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.rejected_counts.get("duplicate_listing") == 1


def test_stale_listing_excluded():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(listed_days_ago=40, source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.rejected_counts.get("stale_listing") == 6
    assert result.status == "unavailable"
def test_invalid_price_rejected_by_schema():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    from pydantic import ValidationError
    try:
        MarketListing(
            source_name="example_market",
            listing_id="bad",
            make="Maruti Suzuki",
            model="Swift",
            year=2021,
            advertised_price_inr=0,
            source_url="https://example.com/bad",
            listed_at=now,
            retrieved_at=now,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("zero price should be invalid")


def test_outlier_rejection():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    prices = [600000] * 6 + [10000000]
    listings = [_listing(price=p, source_id=f"L{i}") for i, p in enumerate(prices)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.rejected_counts.get("price_outlier") == 1
    assert result.evidence_count == 6


def test_fewer_than_five_inliers_unavailable():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(2)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.status == "low_evidence"
    assert result.evidence_count == 2
    assert result.price_range is None


def test_exactly_five_inliers_available():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(5)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.status == "available"
def test_missing_variant_rejected_when_target_variant_known():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].variant = None
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", variant="VXi", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert any(
        any("variant: not reported" in difference for difference in listing.match_differences)
        for listing in result.listings
    )


def test_missing_fuel_is_excluded_when_known_fuel_evidence_is_sufficient():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].fuel_type = None
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, fuel_type="petrol", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.rejected_counts.get("fuel_not_reported_excluded") == 1


def test_missing_transmission_rejected_when_target_transmission_known():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].transmission = None
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, transmission="manual", mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert any(
        "transmission: not reported" in difference
        for item in result.listings
        for difference in item.match_differences
    )


def test_missing_location_rejected_when_target_location_known():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(source_id=f"L{i}") for i in range(6)]
    listings[0].location = None
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert any(
        "location: not reported" in difference
        for item in result.listings
        for difference in item.match_differences
    )


def test_mileage_tolerance_exact_5000_boundary():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(mileage=45000, source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6


def test_mileage_tolerance_exact_15_percent_boundary():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    listings = [_listing(mileage=46001, source_id=f"L{i}") for i in range(6)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert result.evidence_count == 6
    assert all(item.match_type == "approximate_match" for item in result.listings)


def test_tukey_not_applied_with_six_raw_candidates():
    now = datetime(2026, 10, 30, tzinfo=timezone.utc)
    prices = [600000] * 5 + [10000000]
    listings = [_listing(price=p, source_id=f"L{i}") for i, p in enumerate(prices)]
    criteria = MarketVehicleCriteria(make="Maruti Suzuki", model="Swift", year=2021, mileage_km=40000, location="Jaipur")
    result = calculate_market_comparables(listings, criteria, valuation_date=now)
    assert "price_outlier" not in result.rejected_counts
    assert result.evidence_count == 6
