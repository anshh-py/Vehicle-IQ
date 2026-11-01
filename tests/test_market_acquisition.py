import json
import httpx
import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.models import MarketListing as MarketListingModel
from app.market.acquisition import (
    HttpJsonMarketSource,
    LocalPermittedDatasetSource,
    MarketSourceError,
    MarketSourceRateLimitError,
    MarketSourceTimeoutError,
    discover_market_listings,
    import_discovered_listings,
    normalize_source_record,
)
from app.market.repository import MarketListingRepository
from app.pricing.market import MarketVehicleCriteria, calculate_market_comparables

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
CRITERIA = MarketVehicleCriteria(
    make="Hyundai",
    model="i10",
    variant="GLS Sportz",
    year=2013,
    fuel_type="petrol",
    transmission="manual",
    mileage_km=40000,
    location="Jaipur",
)


def _record(**overrides):
    values = {
        "listing_id": "H1",
        "make": "Hyundai",
        "model": "i10",
        "variant": "GLS Sportz",
        "year": 2013,
        "fuel_type": "Petrol",
        "transmission": "Manual",
        "mileage_km": 39000,
        "location": "Jaipur, Rajasthan",
        "asking_price_inr": "450000",
        "source_url": "https://example.com/listings/h1",
        "listed_at": "2026-10-25T00:00:00+00:00",
    }
    values.update(overrides)
    return values


@pytest.fixture
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()
def test_normalize_source_record_maps_fields():
    normalized = normalize_source_record("local_permitted_dataset", _record(), retrieved_at=NOW)
    assert normalized is not None
    assert normalized["make"] == "Hyundai"
    assert normalized["advertised_price_inr"] == Decimal("450000")
    assert normalized["retrieved_at"] == NOW


def test_normalize_source_record_rejects_missing_required_fields():
    assert normalize_source_record("local_permitted_dataset", _record(listed_at=None), retrieved_at=NOW) is None
    assert normalize_source_record("local_permitted_dataset", _record(asking_price_inr=0), retrieved_at=NOW) is None
    assert normalize_source_record("local_permitted_dataset", _record(source_url=None), retrieved_at=NOW) is None


def test_local_source_parses_json_and_deduplicates():
    payload = [_record(), _record(listing_id="H1"), _record(listing_id="H2", source_url="https://example.com/listings/h2")]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = source.fetch(CRITERIA, now=NOW)
    assert len(listings) == 2


def test_local_source_parses_csv():
    csv_text = "listing_id,make,model,year,asking_price_inr,source_url,listed_at,city,state\nL1,Hyundai,i10,2013,450000,https://example.com/l1,2026-10-25T00:00:00+00:00,Jaipur,Rajasthan\n"
    source = LocalPermittedDatasetSource(text=csv_text, input_format="csv")
    listings = source.fetch(CRITERIA, now=NOW)
    assert len(listings) == 1
    assert listings[0].listing_id == "L1"

def test_http_source_success_with_mocked_response():
    def handler(request):
        return httpx.Response(200, json={"results": [_record()]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpJsonMarketSource(
        source_name="local_permitted_dataset",
        base_url="https://example.test/search",
        automated_access_permitted=True,
        client=client,
    )
    listings = source.fetch(CRITERIA, now=NOW)
    assert len(listings) == 1
    assert listings[0].make == "Hyundai"


def test_http_source_rate_limit():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpJsonMarketSource(
        source_name="local_permitted_dataset",
        base_url="https://example.test/search",
        automated_access_permitted=True,
        client=client,
    )
    with pytest.raises(MarketSourceRateLimitError):
        source.fetch(CRITERIA, now=NOW)


def test_http_source_timeout():
    def handler(request):
        raise httpx.TimeoutException("timed out")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpJsonMarketSource(
        source_name="local_permitted_dataset",
        base_url="https://example.test/search",
        automated_access_permitted=True,
        client=client,
    )
    with pytest.raises(MarketSourceTimeoutError):
        source.fetch(CRITERIA, now=NOW)


def test_http_source_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "server error"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpJsonMarketSource(
        source_name="local_permitted_dataset",
        base_url="https://example.test/search",
        automated_access_permitted=True,
        client=client,
    )
    with pytest.raises(MarketSourceError):
        source.fetch(CRITERIA, now=NOW)


def test_http_source_not_permitted_raises():
    source = HttpJsonMarketSource(
        source_name="olx_india",
        base_url="https://example.test/search",
        automated_access_permitted=True,
    )
    with pytest.raises(MarketSourceError):
        source.fetch(CRITERIA, now=NOW)


def test_http_source_invalid_json_raises():
    def handler(request):
        return httpx.Response(200, text="not json")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    source = HttpJsonMarketSource(
        source_name="local_permitted_dataset",
        base_url="https://example.test/search",
        automated_access_permitted=True,
        client=client,
    )
    with pytest.raises(MarketSourceError):
        source.fetch(CRITERIA, now=NOW)

def test_discover_market_listings_deduplicates():
    payload = [_record(), _record(listing_id="H2", source_url="https://example.com/listings/h2")]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = discover_market_listings(source, CRITERIA, now=NOW)
    assert len(listings) == 2


def test_dry_run_import_does_not_write(db_session):
    payload = [_record()]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = discover_market_listings(source, CRITERIA, now=NOW)
    summary = import_discovered_listings(db_session, listings, dry_run=True)
    assert summary.created == 1
    assert db_session.query(MarketListingModel).count() == 0


def test_database_import_persists_discovered_listings(db_session):
    payload = [_record()]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = discover_market_listings(source, CRITERIA, now=NOW)
    summary = import_discovered_listings(db_session, listings, dry_run=False)
    assert summary.created == 1
    assert db_session.query(MarketListingModel).count() == 1


def test_stale_discovered_listing_is_rejected_by_matching(db_session):
    stale_record = _record(listed_at=(NOW - timedelta(days=31)).isoformat())
    payload = [stale_record]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = discover_market_listings(source, CRITERIA, now=NOW)
    import_discovered_listings(db_session, listings, dry_run=False)
    db_listings = MarketListingRepository(db_session).list_for_criteria(CRITERIA)
    result = calculate_market_comparables(db_listings, CRITERIA, valuation_date=NOW)
    assert result.rejected_counts.get("stale_listing") == 1


def test_insufficient_discovered_evidence_returns_low_evidence(db_session):
    payload = [_record()]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = discover_market_listings(source, CRITERIA, now=NOW)
    import_discovered_listings(db_session, listings, dry_run=False)
    db_listings = MarketListingRepository(db_session).list_for_criteria(CRITERIA)
    result = calculate_market_comparables(db_listings, CRITERIA, valuation_date=NOW)
    assert result.status == "low_evidence"
    assert result.price_range is None


def test_missing_fields_are_retained_and_matching_rejects():
    payload = [_record(variant=None, fuel_type=None, transmission=None, mileage_km=None, location=None)]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = source.fetch(CRITERIA, now=NOW)
    assert len(listings) == 1
    assert listings[0].variant is None


def test_duplicate_records_are_deduplicated():
    payload = [_record(), _record(listing_id="H2", source_url="https://example.com/listings/h1")]
    source = LocalPermittedDatasetSource(text=json.dumps(payload), input_format="json")
    listings = source.fetch(CRITERIA, now=NOW)
    assert len(listings) == 1

