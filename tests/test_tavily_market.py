from datetime import datetime, timezone
from decimal import Decimal
import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.market.tavily import (
    TAVILY_SEARCH_ENDPOINT,
    TavilyError,
    TavilyInvalidResponse,
    TavilyMarketListingProvider,
    TavilyRateLimited,
    get_tavily_market_listing_provider,
)
from app.pricing.market import (
    MIN_COMPARABLE_COUNT,
    MarketVehicleCriteria,
    calculate_market_comparables,
)


CRITERIA = MarketVehicleCriteria(
    make="Hyundai",
    model="Creta",
    variant="SX",
    year=2021,
    fuel_type="petrol",
    transmission="manual",
    mileage_km=42_000,
    location="Jaipur",
)


def _result(
    index: int,
    *,
    title: str | None = None,
    content: str | None = None,
    url: str | None = None,
):
    return {
        "title": title or f"Hyundai Creta SX 2021 petrol manual Jaipur {42_000 + index} km",
        "url": url or f"https://cars24.com/listing/{index}",
        "content": content or f"Used car asking price Rs {12_00_000 + index * 10_000:,}",
        "published_date": "2026-10-25",
    }


def _provider(handler, *, key="test-secret"):
    return TavilyMarketListingProvider(
        key,
        result_limit=9,
        transport=httpx.MockTransport(handler),
    )


def _first_query_only(results):
    def handler(request):
        payload = json.loads(request.read())
        response_results = results if "SX" in payload["query"] else []
        return httpx.Response(200, json={"results": response_results})

    return handler


def test_search_builds_vehicle_and_location_query_and_extracts_supported_fields():
    observed = {"payloads": []}

    def handler(request):
        observed["url"] = str(request.url)
        observed["payloads"].append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={
                "results": [
                    _result(index)
                    for index in range(1, MIN_COMPARABLE_COUNT + 1)
                ]
            },
        )

    provider = _provider(handler)
    listings = provider.fetch_listings(CRITERIA)

    assert observed["url"] == TAVILY_SEARCH_ENDPOINT
    assert observed["payloads"][0] == {
        "api_key": "test-secret",
        "query": (
            '"Hyundai Creta" 2021 SX petrol manual 42000 km used car prices India -Grand -Nios near Jaipur'
        ),
        "search_depth": "basic",
        "topic": "general",
        "max_results": 9,
        "include_answer": False,
        "include_raw_content": True,
    }
    assert len(listings) == MIN_COMPARABLE_COUNT
    listing = listings[0]
    assert listing.title.startswith("Hyundai Creta SX")
    assert listing.make == "Hyundai"
    assert listing.model == "Creta"
    assert listing.variant == "SX"
    assert listing.year == 2021
    assert listing.fuel_type == "petrol"
    assert listing.transmission == "manual"
    assert listing.mileage_km == 42_001
    assert listing.location == "Jaipur"
    assert listing.advertised_price_inr == Decimal("1210000.00")
    assert listing.source_name == "cars24.com"
    assert str(listing.source_url) == "https://cars24.com/listing/1"
    assert listing.listed_at is None
    assert listing.retrieved_at.tzinfo == timezone.utc
    assert listing.match_type == "exact_match"
    assert provider.search_observability["queries_issued"] == 1


def test_missing_api_key_disables_provider_without_revealing_secrets(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", None)
    assert get_tavily_market_listing_provider() is None
    monkeypatch.setattr(settings, "tavily_api_key", SecretStr("test-secret"))
    provider = get_tavily_market_listing_provider()
    assert isinstance(provider, TavilyMarketListingProvider)
    assert "test-secret" not in repr(provider)


def test_timeout_and_rate_limit_are_sanitized():
    def timeout(request):
        raise httpx.ReadTimeout("test-secret leaked upstream", request=request)

    with pytest.raises(TavilyError, match="timed out") as error:
        _provider(timeout).fetch_listings(CRITERIA)
    assert "test-secret" not in str(error.value)

    rate_limited = _provider(
        lambda _request: httpx.Response(429, text="test-secret sensitive body")
    )
    with pytest.raises(TavilyRateLimited, match="limit was reached") as rate_error:
        rate_limited.fetch_listings(CRITERIA)
    assert "test-secret" not in str(rate_error.value)
    assert "sensitive body" not in str(rate_error.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"results": {}}),
    ],
)
def test_malformed_tavily_responses_raise_safe_errors(response):
    provider = _provider(lambda _request: response)
    with pytest.raises(TavilyInvalidResponse):
        provider.fetch_listings(CRITERIA)


def test_results_without_prices_or_relevant_vehicle_are_not_listings():
    provider = _provider(
        _first_query_only(
            [
                _result(
                    1,
                    title="Hyundai Creta SX 2021 petrol manual Jaipur",
                    content="Used car, 42,000 km. Contact seller for price.",
                ),
                _result(
                    2,
                    title="Maruti Baleno 2021 petrol manual Jaipur",
                    content="Asking price Rs 700000",
                ),
            ]
        )
    )

    assert provider.fetch_listings(CRITERIA) == []
    assert provider.rejected_listing_counts["missing_or_ambiguous_price"] == 1
    assert provider.rejected_listing_counts["irrelevant_vehicle"] == 1
    assert provider.search_observability["queries_issued"] == 2


def test_duplicate_urls_are_removed_by_comparable_calculation():
    responses = [_result(index) for index in range(MIN_COMPARABLE_COUNT)]
    responses.append(_result(0, url="https://cars24.com/listing/0?campaign=search"))
    provider = _provider(
        lambda _request: httpx.Response(200, json={"results": responses})
    )
    listings = provider.fetch_listings(CRITERIA)

    result = calculate_market_comparables(listings, CRITERIA)

    assert result.status == "available"
    assert result.evidence_count == MIN_COMPARABLE_COUNT
    assert result.rejected_counts.get("duplicate_listing", 0) == 0
    assert provider.rejected_listing_counts["duplicate_listing"] == 1


def test_exact_and_approximate_matches_are_distinguished():
    exact_provider = _provider(
        _first_query_only([_result(index) for index in range(MIN_COMPARABLE_COUNT)])
    )
    exact_listing = exact_provider.fetch_listings(CRITERIA)[0]
    approximate_provider = _provider(
        lambda _request: httpx.Response(
            200,
            json={
                "results": (
                    [
                    _result(
                        1,
                        title="Hyundai Creta 2020 petrol",
                        content="Used car listed for ₹ 12 lakh",
                    )
                    ]
                    if "SX" in json.loads(_request.read())["query"]
                    else []
                )
            },
        )
    )
    approximate_listing = approximate_provider.fetch_listings(CRITERIA)[0]

    assert exact_listing.match_type == "exact_match"
    assert approximate_listing.match_type == "approximate_match"
    assert approximate_listing.match_differences
    assert approximate_listing.variant is None
    assert approximate_listing.fuel_type == "petrol"
    assert approximate_listing.transmission is None
    assert approximate_listing.mileage_km is None
    assert approximate_listing.location is None
    assert approximate_listing.listed_at is None


def test_insufficient_evidence_has_no_range_and_five_results_make_a_range():
    one_provider = _provider(
        lambda _request: httpx.Response(200, json={"results": [_result(1)]})
    )
    too_few = calculate_market_comparables(
        one_provider.fetch_listings(CRITERIA), CRITERIA
    )
    enough_provider = _provider(
        lambda _request: httpx.Response(
            200,
            json={"results": [_result(index) for index in range(MIN_COMPARABLE_COUNT)]},
        )
    )
    enough = calculate_market_comparables(
        enough_provider.fetch_listings(CRITERIA), CRITERIA
    )

    assert too_few.status == "low_evidence"
    assert too_few.price_range is None
    assert enough.status == "available"
    assert enough.evidence_count == MIN_COMPARABLE_COUNT
    assert enough.price_range is not None
    assert enough.price_range.lower_inr > 0


def test_missing_values_are_never_fabricated_and_secret_not_logged(caplog):
    provider = _provider(
        lambda _request: httpx.Response(
            200,
            json={
                "results": [
                    _result(
                        1,
                        title="Hyundai Creta 2021",
                        content="Used car asking price Rs 12 lakh",
                    )
                ]
            },
        )
    )

    listings = provider.fetch_listings(CRITERIA)

    assert len(listings) == 1
    assert listings[0].fuel_type is None
    assert "fuel_type: not reported" in listings[0].match_differences
    assert "test-secret" not in caplog.text


def test_search_diagnostics_log_result_decisions_without_sensitive_url_data(caplog):
    caplog.set_level("INFO")
    results = [
        _result(
            1,
            title="Hyundai Creta SX RJ45CA1879 2021 petrol manual",
            content="Used car asking price Rs 12 lakh",
            url="https://cars24.com/listing/RJ45CA1879?token=private",
        ),
        _result(
            2,
            title="Hyundai Creta SX 2021 petrol manual",
            content="Used car. Contact the seller for price.",
            url="https://cars24.com/listing/no-price",
        ),
    ]
    provider = _provider(_first_query_only(results))

    listings = provider.fetch_listings(CRITERIA)

    response_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tavily_search_response"
    )
    result_records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tavily_search_result_processed"
    ]
    dropped_record = next(
        record
        for record in result_records
        if record.result_url.endswith("/listing/no-price")
    )
    completed_record = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tavily_search_processing_completed"
    )

    assert len(listings) == 1
    assert response_record.query.startswith('"Hyundai Creta"')
    assert response_record.http_status == 200
    assert response_record.result_count == 2
    assert len(result_records) == 2
    assert result_records[0].result_domain == "cars24.com"
    assert result_records[0].price_extracted is True
    assert result_records[0].extracted_price_inr == "1200000.00"
    assert result_records[0].market_research_accepted is True
    assert dropped_record.market_research_rejection_reason == "missing_price"
    assert dropped_record.comparable_rejection_reason == "missing_or_ambiguous_price"
    query_summary = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tavily_query_processing_summary"
    )
    query_summaries = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "tavily_query_processing_summary"
    ]
    assert query_summary.query.startswith('"Hyundai Creta"')
    assert query_summary.result_count == 2
    assert query_summary.market_research_accepted_count == 1
    assert query_summary.market_research_rejection_counts == {"missing_price": 1}
    assert query_summary.comparable_accepted_count == 1
    assert len(query_summaries) == provider.search_observability["queries_issued"]
    assert sorted(record.result_count for record in query_summaries) == [0, 2]
    assert completed_record.final_accepted_listing_count == 1
    assert completed_record.final_market_research_listing_count == 1
    assert '"event":"tavily_search_response"' in caplog.text
    assert '"event":"tavily_search_result_processed"' in caplog.text
    assert '"event":"tavily_query_processing_summary"' in caplog.text
    assert '"event":"tavily_search_processing_completed"' in caplog.text
    assert '"result_domain":"cars24.com"' in caplog.text
    assert '"result_url":"https://cars24.com/listing/no-price"' in caplog.text
    assert '"comparable_rejection_reason":"missing_or_ambiguous_price"' in caplog.text
    assert "private" not in caplog.text
    assert "RJ45CA1879" not in caplog.text
    assert "test-secret" not in caplog.text


def test_progressive_search_expands_year_then_model_and_records_match_types():
    observed_queries = []

    def handler(request):
        query = json.loads(request.read())["query"]
        observed_queries.append(query)
        if "SX" in query:
            results = [
                _result(
                    1,
                    title="Hyundai Creta SX 2021 petrol manual Jaipur",
                    content="42,000 km. Used car asking price Rs 12,10,000",
                )
            ]
        elif "2021" in query:
            results = [
                _result(
                    2,
                    title="Hyundai Creta SX 2020 petrol manual Jaipur",
                    content="42,000 km. Used car asking price Rs 12,20,000",
                ),
                _result(
                    3,
                    title="Hyundai Creta SX 2022 petrol manual Jaipur",
                    content="42,000 km. Used car asking price Rs 12,30,000",
                ),
                _result(
                    4,
                    title="Hyundai Creta EX 2021 petrol automatic",
                    content="Used car asking price Rs 12,40,000",
                ),
                _result(
                    5,
                    title="Hyundai Creta E 2021 petrol manual",
                    content="Used car asking price Rs 12,50,000",
                ),
            ]
        return httpx.Response(200, json={"results": results})

    provider = _provider(handler)
    listings = provider.fetch_listings(CRITERIA)

    assert len(observed_queries) == 2
    assert observed_queries[0].startswith('"Hyundai Creta" ')
    assert "2021 SX petrol manual 42000 km" in observed_queries[0]
    assert observed_queries[1] == '"Hyundai Creta" 2021 used car prices India -Grand -Nios near Jaipur'
    assert all("India" in query for query in observed_queries)
    assert all("Jaipur" in query for query in observed_queries)
    assert [item.match_type for item in listings].count("exact_match") == 1
    assert [item.match_type for item in listings].count("close_match") >= 2
    assert len(listings) >= MIN_COMPARABLE_COUNT
    assert all(item.variant == "SX" for item in listings)
    assert provider.search_observability["queries_issued"] == 2
    assert provider.search_observability["raw_results_returned"] == 5
    assert provider.search_observability["listings_extracted"] == 5


def test_same_model_variants_and_transmission_mileage_are_approximate():
    retrieved = datetime.now(timezone.utc)
    variant_result = _result(
        1,
        title="Hyundai Creta EX 2021 petrol automatic",
        content="42,000 km. Used car asking price Rs 12,00,000",
    )
    listing, reason = TavilyMarketListingProvider._listing_from_result(
        variant_result, CRITERIA, retrieved
    )
    assert reason is None
    assert listing is not None
    result = calculate_market_comparables([listing], CRITERIA)
    assert result.listings[0].match_type == "approximate_match"
    assert any("variant:" in difference for difference in result.listings[0].match_differences)
    assert any(
        "transmission:" in difference
        for difference in result.listings[0].match_differences
    )


def test_other_fuels_and_years_beyond_three_are_not_combined():
    retrieved = datetime.now(timezone.utc)
    petrol = _result(1, title="Hyundai Creta SX 2020 petrol manual")
    diesel = _result(
        2,
        title="Hyundai Creta SX 2021 diesel manual",
        content="Used car asking price Rs 12,00,000",
    )
    too_old = _result(
        3,
        title="Hyundai Creta SX 2017 petrol manual",
        content="Used car asking price Rs 12,00,000",
    )
    extracted = [
            TavilyMarketListingProvider._listing_from_result(
                result, CRITERIA, retrieved
            )
            for result in (petrol, diesel, too_old)
    ]
    listings = [item for item, reason in extracted if item is not None and reason is None]

    comparable = calculate_market_comparables(listings, CRITERIA)

    assert len(comparable.listings) == 1
    assert comparable.listings[0].year == 2020
    assert comparable.rejected_counts["fuel_mismatch"] == 1
    assert extracted[2][0] is not None
    assert extracted[2][1] is None
    assert comparable.rejected_counts["year_mismatch"] == 1
    diesel_listing = extracted[1][0]
    assert diesel_listing is not None
    from app.pricing.market import _classify

    match_type, differences = _classify(diesel_listing, CRITERIA)
    assert match_type == "approximate_match"
    assert any("fuel_type: diesel vs petrol" in difference for difference in differences)


def test_rejects_non_listing_sources_editorial_pages_and_implausible_prices():
    bad_results = [
        _result(1, url="https://youtube.com/watch?v=abc"),
        _result(2, url="https://cars24.com/news/creta-review"),
        _result(3, content="Used car asking price Rs 5000"),
        _result(4, url="http://cars24.com/listing/4"),
    ]
    provider = _provider(_first_query_only(bad_results))

    assert provider.fetch_listings(CRITERIA) == []
    assert provider.rejected_listing_counts["non_listing_source"] == 1
    assert provider.rejected_listing_counts["editorial_or_video_result"] == 1
    assert provider.rejected_listing_counts["implausible_price"] == 1
    assert provider.rejected_listing_counts["invalid_listing_url"] == 1


def test_duplicate_advertisements_are_removed_across_expansion_queries():
    def handler(request):
        query = json.loads(request.read())["query"]
        if "SX petrol" in query:
            return httpx.Response(200, json={"results": [_result(1)]})
        if "2021" in query:
            expanded = [
                _result(1, url="https://cars24.com/listing/1?campaign=nearby"),
                *[
                    _result(
                        index,
                        title=f"Hyundai Creta SX 2020 petrol manual Jaipur",
                        content=(
                            f"42,000 km. Used car asking price "
                            f"Rs {12_00_000 + index * 10000:,}"
                        ),
                    )
                    for index in range(2, 6)
                ],
            ]
            return httpx.Response(200, json={"results": expanded})
        return httpx.Response(200, json={"results": []})

    provider = _provider(handler)
    listings = provider.fetch_listings(CRITERIA)

    assert len(listings) == 5
    assert provider.search_observability["duplicates_removed"] == 1
    assert provider.search_observability["queries_issued"] == 2


def test_xgboost_feature_order_remains_unchanged():
    from app.pricing.provider import MODEL_FEATURE_ORDER

    assert MODEL_FEATURE_ORDER == (
        "brand",
        "model",
        "seller_type",
        "fuel_type",
        "transmission_type",
        "vehicle_age",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "seats",
        "km_per_year",
    )


def test_search_candidates_are_ranked_and_capped_at_two_without_broad_fallback():
    observed_queries = []

    def handler(request):
        observed_queries.append(json.loads(request.read())["query"])
        return httpx.Response(200, json={"results": []})

    provider = _provider(handler)
    provider.fetch_listings(CRITERIA)

    assert observed_queries == [
        '"Hyundai Creta" 2021 SX petrol manual 42000 km used car prices India -Grand -Nios near Jaipur',
        '"Hyundai Creta" 2021 used car prices India -Grand -Nios near Jaipur',
    ]
    assert provider.search_scopes == [
        "Jaipur and nearby regions",
        "Jaipur and nearby regions",
    ]
    regional_criteria = CRITERIA.model_copy(update={"location": "Jaipur, Rajasthan"})
    assert all(
        query.endswith("near Jaipur, Rajasthan")
        for query in TavilyMarketListingProvider._queries(regional_criteria)
    )
    assert all("India" in query and "-Grand -Nios" in query for query in TavilyMarketListingProvider._queries(regional_criteria))


def test_missing_optional_details_produce_one_useful_identity_query():
    criteria = MarketVehicleCriteria(make="Hyundai", model="i10")

    queries = TavilyMarketListingProvider._queries(criteria)

    assert queries == ['"Hyundai i10" used car prices India -Grand -Nios']


def test_query_priority_keeps_make_model_identity_and_year_variant_location():
    criteria = MarketVehicleCriteria(
        make="Honda",
        model="City",
        year=2020,
        variant="ZX",
        location="Jaipur, Rajasthan",
    )

    queries = TavilyMarketListingProvider._queries(criteria)

    assert len(queries) == 2
    assert queries[0] == '"Honda City" 2020 ZX used car prices India -Grand -Nios near Jaipur, Rajasthan'
    assert queries[1] == '"Honda City" 2020 used car prices India -Grand -Nios near Jaipur, Rajasthan'
    assert all(query.startswith('"Honda City" ') for query in queries)


def test_unpriced_relevant_results_are_retained_as_market_context():
    result = _result(
        1,
        title="Hyundai Creta SX used car market information",
        content="A review discusses ownership and resale demand; ask the seller for price.",
    )
    provider = _provider(_first_query_only([result]))

    assert provider.fetch_listings(CRITERIA) == []
    assert provider.market_research_results == []


def test_different_year_and_variant_are_retained_as_optional_listing_details():
    result = _result(
        1,
        title="Hyundai Creta EX 2019 petrol manual Jaipur",
        content="Used car asking price Rs. 3.2 Lakh",
    )
    provider = _provider(_first_query_only([result]))

    listings = provider.fetch_listings(CRITERIA)

    assert len(listings) == 1
    research = provider.market_research_results[0]
    assert research["year"] == 2019
    assert research["variant"] == "EX"
    assert research["asking_price_inr"] == "320000.00"
    assert research["title"] == "Hyundai Creta EX 2019 petrol manual Jaipur"
    assert research["make"] == "Hyundai"
    assert research["model"] == "Creta"
    assert research["currency"] == "INR"
    assert research["retrieved_at"]
    assert 0 <= research["relevance_score"] <= 100
    assert research["priced_used_listing"] is True
    assert set(research) == {
        "title",
        "source_url",
        "source_name",
        "make",
        "model",
        "year",
        "variant",
        "fuel_type",
        "transmission",
        "mileage_km",
        "location",
        "currency",
        "asking_price_inr",
        "retrieved_at",
        "relevance_score",
        "priced_used_listing",
    }


def test_variant_after_year_is_extracted_only_from_the_listing_title():
    result = _result(
        1,
        title="Hyundai Creta 2019 EX used car",
        content="Used car asking price Rs 8 lakh.",
    )

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["year"] == 2019
    assert research["variant"] == "EX"


def test_same_year_different_variant_is_classified_separately():
    result = _result(
        1,
        title="Hyundai Creta EX 2021 petrol manual Jaipur",
        content="Used car asking price ₹2,50,000",
    )
    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["variant"] == "EX"
    assert research["asking_price_inr"] == "250000.00"


def test_exact_vehicle_match_has_highest_relevance():
    result = _result(
        1,
        title="Hyundai Creta SX 2021 petrol manual Jaipur",
        content="42,000 km. Used car asking price INR 1200000",
    )
    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["year"] == 2021
    assert research["variant"] == "SX"
    assert research["asking_price_inr"] == "1200000.00"
    assert research["relevance_score"] > 50


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("asking price ₹2.5 lakh", Decimal("250000.00")),
        ("asking price Rs 2,50,000", Decimal("250000.00")),
        ("asking price 250000", Decimal("250000.00")),
    ],
)
def test_common_indian_asking_price_formats(label, expected):
    assert TavilyMarketListingProvider._asking_price(label) == expected


def test_missing_fuel_is_retained_as_approximate_when_needed_for_minimum_evidence():
    provider = _provider(
        _first_query_only([
            _result(1, title="Hyundai Creta SX 2021 Jaipur 42000 km", content="Used car asking price Rs 12 lakh"),
            _result(2, title="Hyundai Creta SX 2021 petrol manual Jaipur 42000 km", content="Used car asking price Rs 12,10,000"),
            _result(3, title="Hyundai Creta SX 2021 petrol manual Jaipur 42000 km", content="Used car asking price Rs 12,20,000"),
        ])
    )
    listings = provider.fetch_listings(CRITERIA)
    result = calculate_market_comparables(listings, CRITERIA)
    assert result.status == "available"
    assert result.evidence_count == MIN_COMPARABLE_COUNT
    assert any("fuel_type: not reported" in diff for item in result.listings for diff in item.match_differences)


def test_missing_fuel_is_not_used_when_enough_same_fuel_matches_exist():
    raw = [
        TavilyMarketListingProvider._listing_from_result(
            _result(
                index,
                title=("Hyundai Creta SX 2021 Jaipur 42000 km" if index == 9 else None),
                content="Used car asking price Rs 12 lakh" if index == 9 else f"Used car asking price Rs {12_00_000 + index * 10000:,}",
            ),
            CRITERIA,
            datetime.now(timezone.utc),
        )[0]
        for index in (1, 2, 3, 9)
    ]
    result = calculate_market_comparables([item for item in raw if item], CRITERIA)
    assert result.status == "available"
    assert result.evidence_count == MIN_COMPARABLE_COUNT
    assert result.rejected_counts["fuel_not_reported_excluded"] == 1


def test_unknown_target_fuel_and_unknown_listing_fuel_do_not_crash():
    criteria = CRITERIA.model_copy(update={"fuel_type": None})
    listings = [
        TavilyMarketListingProvider._listing_from_result(
            _result(
                index,
                title=f"Hyundai Creta SX 2021 Jaipur {42_000 + index} km",
                content=f"Used car asking price Rs {12_00_000 + index * 10_000:,}",
            ),
            criteria,
            datetime.now(timezone.utc),
        )[0]
        for index in range(1, MIN_COMPARABLE_COUNT + 1)
    ]
    result = calculate_market_comparables(
        [listing for listing in listings if listing is not None], criteria
    )
    assert result.status == "available"
    assert result.evidence_count == MIN_COMPARABLE_COUNT


def test_editorial_information_is_not_in_the_used_listing_output():
    result = _result(
        1,
        title="Hyundai Creta resale market review",
        url="https://automotive.example/articles/creta-resale",
        content="Market information and ownership trends; prices vary by condition.",
    )
    provider = _provider(_first_query_only([result]))

    assert provider.fetch_listings(CRITERIA) == []
    assert provider.market_research_results == []


def test_missing_vehicle_attributes_are_omitted_but_a_sourced_price_is_kept():
    result = _result(
        1,
        title="Hyundai Creta resale",
        content="Used car asking price Rs 8 lakh.",
    )

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["year"] is None
    assert research["variant"] is None
    assert research["asking_price_inr"] == "800000.00"


def test_unrelated_models_are_not_market_context():
    result = _result(
        1,
        title="Hyundai Venue 2021 petrol",
        content="Used car for sale Rs 700000",
    )

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert research is None
    assert reason == "irrelevant_vehicle"


@pytest.mark.parametrize(
    ("title", "content"),
    [
        (
            "Hyundai Grand i10 2018 Sportz used car",
            "Petrol used car asking price Rs 250000",
        ),
        (
            "Hyundai Grand i10 Nios 2019 Sportz used car",
            "Hyundai i10 variants are popular; petrol used car asking price Rs 250000",
        ),
        (
            "Hyundai i10 Nios 2019 Sportz used car",
            "Hyundai i10 variants are popular; petrol used car asking price Rs 250000",
        ),
    ],
)
def test_i10_search_does_not_accept_grand_i10_or_nios(title, content):
    criteria = MarketVehicleCriteria(
        make="Hyundai",
        model="i10",
        variant="Sportz",
        year=2018,
        fuel_type="petrol",
        transmission="manual",
        location="Jaipur",
    )
    result = {
        "title": title,
        "url": "https://cars24.com/listing/grand-i10",
        "content": content,
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, criteria, datetime.now(timezone.utc)
    )
    listing, listing_reason = TavilyMarketListingProvider._listing_from_result(
        result, criteria, datetime.now(timezone.utc)
    )

    assert research is None
    assert reason == "irrelevant_vehicle"
    assert listing is None
    assert listing_reason == "irrelevant_vehicle"


def test_i10_exact_model_identity_is_retained():
    criteria = MarketVehicleCriteria(make="Hyundai", model="i10")
    result = {
        "title": "Hyundai i10 2018 Sportz used car",
        "url": "https://cars24.com/listing/i10",
        "content": "Petrol used car asking price Rs 250000",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, criteria, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == "250000.00"
    assert research["source_url"] == result["url"]


@pytest.mark.parametrize(
    ("url", "expected_reason"),
    [
        ("http://cars24.com/i10", None),
        ("https://cars24.com/i10", None),
        ("javascript:alert(1)", "invalid_listing_url"),
        ("//market.example/i10", "invalid_listing_url"),
    ],
)
def test_market_research_source_urls_allow_only_http_and_https(url, expected_reason):
    result = {
        "title": "Hyundai i10 used car market information",
        "url": url,
        "content": "Hyundai i10 used car asking price Rs 250000.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10"),
        datetime.now(timezone.utc),
    )

    if expected_reason is None:
        assert reason is None
        assert research is not None
        assert research["source_url"] == url
    else:
        assert research is None
        assert reason == expected_reason


@pytest.mark.parametrize(
    ("title", "url", "content", "raw_content", "expected_price"),
    [
        (
            "Hyundai i10 GLS Sportz 2013 used car",
            "https://www.cars24.com/listing/hyundai-i10-gls-sportz-2013-12345",
            "Used car listing in Jaipur.",
            "Asking price ₹1.98 lakh.",
            "198000.00",
        ),
        (
            "Hyundai i10 2013 GLS Sportz for sale",
            "https://www.carwale.com/listing/hyundai-i10-gls-sportz-2013-12345",
            "Used car asking price Rs. 2,25,000.",
            "",
            "225000.00",
        ),
        (
            "Hyundai i10 Sportz 2013 in Jaipur",
            "https://www.olx.in/item/hyundai-i10-sportz-2013-iid-123",
            "Car listing with seller details.",
            "INR 2.5 lakh",
            "250000.00",
        ),
        (
            "Hyundai i10 2013 GLS used car",
            "https://cars.trovit.co.in/listing/hyundai-i10-2013",
            "Price: Rs 2,50,000.",
            "",
            "250000.00",
        ),
    ],
)
def test_sourced_prices_from_indian_used_car_platform_result_shapes(
    title, url, content, raw_content, expected_price
):
    criteria = MarketVehicleCriteria(
        make="Hyundai", model="i10", year=2013, variant="GLS Sportz"
    )
    result = {
        "title": title,
        "url": url,
        "content": content,
        "raw_content": raw_content,
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, criteria, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == expected_price
    assert research["source_url"] == url
    assert research["year"] == 2013
    assert research["variant"] is not None


def test_raw_content_is_used_for_identity_price_and_listing_extraction():
    criteria = MarketVehicleCriteria(make="Hyundai", model="i10", year=2013)
    result = {
        "title": "2013 Sportz used car listing",
        "url": "https://www.cars24.com/listing/hyundai-i10-sportz-2013-12345",
        "content": "Price details below.",
        "raw_content": "Hyundai i10. Asking price INR 2.5 lakh.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, criteria, datetime.now(timezone.utc)
    )
    listing, listing_reason = TavilyMarketListingProvider._listing_from_result(
        result, criteria, datetime.now(timezone.utc)
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == "250000.00"
    assert research["source_url"] == result["url"]
    assert listing_reason is None
    assert listing is not None
    assert listing.advertised_price_inr == Decimal("250000.00")


@pytest.mark.parametrize(
    ("field", "text", "expected_reason"),
    [
        ("raw_content", "New car ex-showroom price ₹8 lakh.", "new_or_ex_showroom_price"),
        ("raw_content", "Used car price $8,000.", "foreign_currency_price"),
        ("content", "Used car; contact seller for price.", "missing_price"),
    ],
)
def test_research_rejections_explain_price_and_vehicle_exclusions(
    field, text, expected_reason
):
    result = {
        "title": "Hyundai i10 Sportz 2013 used car",
        "url": "https://www.cars24.com/listing/hyundai-i10-sportz-2013-12345",
        "content": "",
        "raw_content": "",
    }
    result[field] = text

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10"),
        datetime.now(timezone.utc),
    )

    assert research is None
    assert reason == expected_reason


def test_provider_exposes_market_research_rejection_counts():
    result = {
        "title": "Hyundai i10 Sportz 2013 used car",
        "url": "https://www.cars24.com/listing/hyundai-i10-sportz-2013-12345",
        "content": "Contact the seller for price.",
    }
    provider = _provider(_first_query_only([result]))

    provider.fetch_listings(
        MarketVehicleCriteria(make="Hyundai", model="i10", variant="SX")
    )

    assert provider.market_research_rejection_counts == {"missing_price": 1}
    assert provider.search_observability["market_research_rejected"] == 1


def test_market_research_uses_listing_price_not_unrelated_raw_page_amounts():
    result = {
        "title": "Hyundai i10 GLS Sportz used car listing",
        "url": "https://www.olx.in/item/hyundai-i10-sportz-2013-iid-123",
        "content": (
            "2013 Hyundai i10 GLS Sportz petrol used car asking price "
            "Rs. 2,25,000. Jaipur listing."
        ),
        "raw_content": (
            "Hyundai i10 used listing. EMI starts at Rs 6,500 per month."
        ),
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10"),
        datetime.now(timezone.utc),
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == "225000.00"
    assert research["source_url"] == result["url"]
    comparable, comparable_reason = TavilyMarketListingProvider._listing_from_result(
        result,
        MarketVehicleCriteria(
            make="Hyundai",
            model="i10",
            year=2013,
            variant="GLS Sportz",
            fuel_type="petrol",
        ),
        datetime.now(timezone.utc),
    )
    assert comparable_reason is None
    assert comparable is not None
    assert comparable.advertised_price_inr == Decimal("225000.00")


@pytest.mark.parametrize(
    ("title", "url"),
    [
        (
            "149 Used Hyundai i10 2013 cars in India",
            "https://www.cars24.com/buy-used-car/hyundai-i10/",
        ),
        (
            "Used i10 Cars from 2013 in Bangalore",
            "https://www.spinny.com/used-cars-in-bangalore/hyundai-i10/",
        ),
        (
            "948 Used I10 2013 in India",
            "https://www.olx.in/cars_c84/q-hyundai-i10",
        ),
    ],
)
def test_marketplace_search_pages_are_rejected_as_collections_not_new_cars(title, url):
    result = {
        "title": title,
        "url": url,
        "content": "Used Hyundai i10 search results. New car prices may also appear.",
        "raw_content": "EMI from Rs 6,500 per month; price fragments ₹24, ₹2, ₹50, ₹29.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013),
        datetime.now(timezone.utc),
    )

    assert research is None
    assert reason == "collection_page"


def test_used_listing_is_not_rejected_for_unrelated_new_car_copy():
    result = {
        "title": "Hyundai i10 GLS Sportz 2013 used car listing",
        "url": "https://www.cars24.com/listing/hyundai-i10-gls-sportz-2013-12345",
        "content": (
            "Used car asking price INR 2,25,000. "
            "New car ex-showroom prices are discussed elsewhere on this page."
        ),
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013),
        datetime.now(timezone.utc),
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == "225000.00"
    assert research["source_url"] == result["url"]


def test_genuine_new_car_price_context_is_still_rejected():
    result = {
        "title": "Hyundai i10 2013 used car listing",
        "url": "https://www.cars24.com/listing/hyundai-i10-12345",
        "content": "New car ex-showroom price ₹8 lakh.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013),
        datetime.now(timezone.utc),
    )

    assert research is None
    assert reason == "new_or_ex_showroom_price"


def test_price_parser_excludes_emi_year_mileage_and_engine_values():
    result = {
        "title": "Hyundai i10 GLS Sportz 2013 used car listing",
        "url": "https://www.olx.in/item/hyundai-i10-iid-123",
        "content": (
            "Used car, model year 2013, mileage 42,000 km, engine 1,200 cc. "
            "Asking price Rs. 2,25,000. EMI starts at INR 6,500 per month."
        ),
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10"),
        datetime.now(timezone.utc),
    )

    assert reason is None
    assert research is not None
    assert research["asking_price_inr"] == "225000.00"


@pytest.mark.parametrize("price_fragment", ["₹24", "₹2", "₹50", "₹29", "₹1,9,000"])
def test_suspicious_or_malformed_price_fragments_are_not_listings(price_fragment):
    result = {
        "title": "Hyundai i10 2013 used car listing",
        "url": "https://www.olx.in/item/hyundai-i10-iid-123",
        "content": f"Used car asking price {price_fragment}.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013),
        datetime.now(timezone.utc),
    )

    assert research is None
    assert reason == "implausible_price"


@pytest.mark.parametrize(
    "url",
    [
        "https://www.cars24.com/buy-used-car/hyundai-i10/",
        "https://www.carwale.com/used/cars-for-sale/hyundai-i10/",
        "https://www.olx.in/cars_c84/q-hyundai-i10",
        "https://www.carwale.com/used/delhi/under-1-lakh/page-4",
        "https://www.cartrade.com/second-hand/hyundai-i10/page-44",
    ],
)
def test_market_research_rejects_collection_pages(url):
    result = {
        "title": "Hyundai i10 GLS Sportz used car listing",
        "url": url,
        "content": "Used car asking price Rs. 2,25,000.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10"),
        datetime.now(timezone.utc),
    )

    assert research is None
    assert reason == "collection_page"


def test_market_research_accepts_missing_year_and_variant_as_unavailable():
    result = {
        "title": "Hyundai i10 used car listing",
        "url": "https://www.olx.in/item/hyundai-i10-iid-123",
        "content": "Used car listing in Jaipur.",
        "raw_content": "Hyundai i10. Asking price INR 2.5 lakh.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013, variant="GLS"),
        datetime.now(timezone.utc),
    )

    assert reason is None
    assert research is not None
    assert research["year"] is None
    assert research["variant"] is None
    assert research["asking_price_inr"] == "250000.00"


def test_market_research_extracts_reported_year_and_variant_from_snippet():
    result = {
        "title": "Hyundai i10 used car listing",
        "url": "https://www.olx.in/item/hyundai-i10-iid-123",
        "content": "Hyundai i10 2013 GLS Sportz petrol used car listing.",
        "raw_content": "Hyundai i10. Asking price INR 2.5 lakh.",
    }

    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result,
        MarketVehicleCriteria(make="Hyundai", model="i10", year=2013),
        datetime.now(timezone.utc),
    )

    assert reason is None
    assert research is not None
    assert research["year"] == 2013
    assert research["variant"] == "GLS Sportz"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hyundai Creta ₹2,50,000", Decimal("250000.00")),
        ("Hyundai Creta Rs 2,50,000", Decimal("250000.00")),
        ("Hyundai Creta INR 250000", Decimal("250000.00")),
        ("Hyundai Creta ₹2.5 lakh", Decimal("250000.00")),
        ("Hyundai Creta Rs. 3.2 Lakh", Decimal("320000.00")),
        ("Hyundai Creta 2.5 L", Decimal("250000.00")),
    ],
)
def test_supported_indian_price_formats_are_parsed(text, expected):
    assert TavilyMarketListingProvider._asking_price(text) == expected


def test_ambiguous_and_implausible_prices_are_not_displayed_as_reliable():
    ambiguous, ambiguous_status = TavilyMarketListingProvider._asking_price_result(
        "Hyundai Creta price Rs 250000; another listing Rs 300000"
    )
    implausible = _result(
        1,
        title="Hyundai Creta 2021 petrol manual Jaipur",
        content="Used car asking price Rs 5000",
    )
    context, reason = TavilyMarketListingProvider._market_research_from_result(
        implausible, CRITERIA, datetime.now(timezone.utc)
    )

    assert ambiguous is None
    assert ambiguous_status == "ambiguous"
    assert reason == "implausible_price"
    assert context is None


@pytest.mark.parametrize(
    "content",
    [
        "Brand new Hyundai Creta ex-showroom price Rs 12 lakh.",
        "Hyundai Creta new car price Rs 12 lakh.",
        "Hyundai Creta used car price $12,000.",
    ],
)
def test_new_vehicle_and_foreign_currency_prices_are_excluded(content):
    result = _result(
        1,
        title="Hyundai Creta used car listing",
        content=content,
    )
    research, reason = TavilyMarketListingProvider._market_research_from_result(
        result, CRITERIA, datetime.now(timezone.utc)
    )

    assert research is None
    assert reason in {
        "new_or_ex_showroom_price",
        "foreign_currency_price",
        "missing",
    }
