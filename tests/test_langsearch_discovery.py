from datetime import timezone

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.pricing.langsearch_discovery import (
    LANGSEARCH_ENDPOINT,
    LangSearchDiscoveryClient,
    LangSearchError,
    LangSearchInvalidResponse,
    LangSearchRateLimited,
    get_langsearch_discovery_client,
)
from app.pricing.market import MarketListing, MarketVehicleCriteria


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


def test_search_sends_documented_http_request_and_parses_discovery_fields():
    observed = {}

    def handler(request):
        observed["url"] = str(request.url)
        observed["headers"] = dict(request.headers)
        observed["payload"] = request.read()
        return httpx.Response(
            200,
            json={
                "data": {
                    "webPages": {
                        "value": [
                            {
                                "name": "Swift VXi listing",
                                "url": "https://cars.example/listing/1",
                                "snippet": "2021 Swift VXi INR 6,20,000",
                                "text": "Page text supplied by search API.",
                                "datePublished": "2026-10-21",
                            }
                        ]
                    }
                },
                "usage": {"input_tokens": 12, "output_tokens": 45},
            },
        )

    result = LangSearchDiscoveryClient(
        "unit-test-secret", result_limit=7, transport=httpx.MockTransport(handler)
    ).search(CRITERIA)

    assert observed["url"] == LANGSEARCH_ENDPOINT
    assert observed["headers"]["authorization"] == "Bearer unit-test-secret"
    import json

    payload = json.loads(observed["payload"])
    assert payload == {
        "query": "Maruti Swift VXi 2021 petrol manual 42000 km Jaipur used car for sale price",
        "count": 7,
        "contents": {"text": True},
        "freshness": "noLimit",
    }
    assert len(result) == 1
    assert result[0].title == "Swift VXi listing"
    assert result[0].url == "https://cars.example/listing/1"
    assert result[0].snippet == "2021 Swift VXi INR 6,20,000"
    assert result[0].page_text == "Page text supplied by search API."
    assert result[0].publication_date == "2026-10-21"
    assert result[0].retrieved_at.tzinfo == timezone.utc
    assert not isinstance(result[0], MarketListing)


def test_skips_results_without_verifiable_https_title_and_url():
    payload = {
        "data": {
            "webPages": {
                "value": [
                    {"url": "https://cars.example/1"},
                    {"name": "No URL"},
                    {"name": "HTTP page", "url": "http://cars.example/2"},
                    {"name": "Valid", "url": "https://cars.example/3"},
                ]
            }
        }
    }
    client = LangSearchDiscoveryClient(
        "fake", transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    )

    results = client.search(CRITERIA)

    assert [item.title for item in results] == ["Valid"]
    assert results[0].snippet is None
    assert results[0].page_text is None
    assert results[0].publication_date is None


@pytest.mark.parametrize("status_code", [401, 403, 500])
def test_http_failures_are_safe_and_do_not_include_response_body(status_code):
    client = LangSearchDiscoveryClient(
        "fake-secret",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(status_code, text="sensitive upstream body")
        ),
    )

    with pytest.raises(LangSearchError, match=f"HTTP {status_code}") as error:
        client.search(CRITERIA)
    assert "sensitive upstream body" not in str(error.value)
    assert "fake-secret" not in str(error.value)


def test_rate_limit_is_reported_without_retry():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return httpx.Response(429, text="limit details")

    client = LangSearchDiscoveryClient("fake", transport=httpx.MockTransport(handler))
    with pytest.raises(LangSearchRateLimited, match="usage limit"):
        client.search(CRITERIA)
    assert calls == 1


def test_timeout_and_transport_failures_are_sanitized():
    def timeout(request):
        raise httpx.ReadTimeout("contains a fake-secret", request=request)

    client = LangSearchDiscoveryClient("fake-secret", transport=httpx.MockTransport(timeout))
    with pytest.raises(LangSearchError, match="timed out") as error:
        client.search(CRITERIA)
    assert "fake-secret" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"data": {"webPages": {"value": {}}}}),
    ],
)
def test_malformed_responses_raise_safe_error(response):
    client = LangSearchDiscoveryClient(
        "fake", transport=httpx.MockTransport(lambda _request: response)
    )
    with pytest.raises(LangSearchInvalidResponse):
        client.search(CRITERIA)


def test_factory_requires_configured_secret_without_exposing_it(monkeypatch):
    monkeypatch.setattr(settings, "langsearch_api_key", None)
    assert get_langsearch_discovery_client() is None
    monkeypatch.setattr(settings, "langsearch_api_key", SecretStr("fake-test-key"))
    client = get_langsearch_discovery_client()
    assert isinstance(client, LangSearchDiscoveryClient)
    assert "fake-test-key" not in repr(client)
