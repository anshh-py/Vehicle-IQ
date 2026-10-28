from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.pricing.market import (
    MIN_COMPARABLE_COUNT,
    MarketListing,
    MarketVehicleCriteria,
    calculate_market_comparables,
)


NOW = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
CRITERIA = MarketVehicleCriteria(
    make="Maruti",
    model="Swift",
    variant="VXi",
    year=2021,
    fuel_type="petrol",
    transmission="manual",
    mileage_km=42_000,
    location="Jaipur",
)


def listing_payload(index: int, price: int, **overrides):
    values = {
        "source_name": "Verified Listings Feed",
        "listing_id": f"car-{index}",
        "make": " maruti ",
        "model": "SWIFT",
        "variant": "VXi",
        "year": 2021,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "mileage_km": 42_000,
        "location": "Jaipur",
        "advertised_price_inr": Decimal(price),
        "source_url": f"https://listings.example/cars/{index}?campaign=test",
        "listed_at": NOW - timedelta(days=2),
        "retrieved_at": NOW - timedelta(hours=1),
    }
    values.update(overrides)
    return values


def listing(index: int, price: int, **overrides) -> MarketListing:
    return MarketListing.model_validate(listing_payload(index, price, **overrides))


def test_filters_duplicates_malformed_stale_and_irrelevant_listings():
    duplicate_url = listing(2, 810_000, source_url="https://listings.example/cars/1")
    stale = listing(3, 805_000, listed_at=NOW - timedelta(days=31))
    stale_retrieval = listing(4, 805_000, retrieved_at=NOW - timedelta(days=31))
    wrong_car = listing(5, 810_000, model="Baleno")
    bad_payload = {"source_name": "broken", "advertised_price_inr": "invalid"}

    result = calculate_market_comparables(
        [listing(1, 800_000), duplicate_url, stale, stale_retrieval, wrong_car, bad_payload],
        CRITERIA,
        valuation_date=NOW,
    )

    assert result.status == "low_evidence"
    assert result.evidence_count == 1
    assert result.price_range is None
    assert result.rejected_counts == {
        "duplicate_listing": 1,
        "stale_listing": 1,
        "stale_retrieval_timestamp": 1,
        "model_mismatch": 1,
        "malformed_listing": 1,
    }


def test_empty_or_insufficient_evidence_never_returns_a_price_range():
    empty = calculate_market_comparables([], CRITERIA, valuation_date=NOW)
    four = calculate_market_comparables(
        [listing(index, 800_000 + index * 1000) for index in range(MIN_COMPARABLE_COUNT - 1)],
        CRITERIA,
        valuation_date=NOW,
    )

    assert empty.status == "low_evidence"
    assert empty.price_range is None
    assert four.status == "low_evidence"
    assert four.evidence_count == 4
    assert four.price_range is None


def test_robust_range_removes_outlier_and_uses_advertised_price_iqr():
    prices = [800_000, 810_000, 820_000, 830_000, 840_000, 5_000_000]

    result = calculate_market_comparables(
        [listing(index, price) for index, price in enumerate(prices)],
        CRITERIA,
        valuation_date=NOW,
    )

    assert result.status == "available"
    assert result.evidence_count == 5
    assert result.rejected_counts["price_outlier"] == 1
    assert result.price_range is not None
    assert result.price_range.lower_inr == Decimal("810000.00")
    assert result.price_range.median_inr == Decimal("820000.00")
    assert result.price_range.upper_inr == Decimal("830000.00")
    assert all(str(entry.source_url).startswith("https://") for entry in result.listings)


@pytest.mark.parametrize(
    "override",
    [
        {"advertised_price_inr": 0},
        {"source_url": "http://listings.example/car"},
        {"listed_at": datetime(2026, 10, 1)},
    ],
)
def test_malformed_market_listing_is_rejected(override):
    invalid = listing_payload(1, 800_000, **override)
    result = calculate_market_comparables(
        [invalid], CRITERIA, valuation_date=NOW
    )

    assert result.status == "low_evidence"
    assert result.rejected_counts["malformed_listing"] == 1
