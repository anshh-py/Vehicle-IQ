from datetime import date

import httpx
import pytest

from app.core.config import settings
from app.vehicle_lookup.provider import (
    VehicleLookupAuthenticationError,
    VehicleLookupInvalidResponse,
    VehicleLookupRejected,
    VehicleLookupUnavailable,
)
from app.vehicle_lookup.vehicle_services import (
    ChallanHTTPProvider,
    FastagHTTPProvider,
    get_challan_provider,
    get_fastag_provider,
)


def challan_provider(handler):
    return ChallanHTTPProvider(
        "https://challan-info-new-abhigyan.vercel.app/challaninfo",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )


def fastag_provider(handler):
    return FastagHTTPProvider(
        "https://abhigyan-fastag-info.vercel.app/vehtofastaginfo",
        timeout_seconds=3,
        transport=httpx.MockTransport(handler),
    )


def test_challan_provider_normalizes_nested_records_and_url_encodes_registration():
    def handler(request):
        assert request.url.params["vehicle"] == "RJ45CA1879"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "challans": [
                        {
                            "challan_status": "pending",
                            "violation_description": "Speed limit violation",
                            "amount": "₹1,250.00",
                            "date": "2025-03-01",
                            "payment_status": "unpaid",
                            "owner_name": "Private Person",
                            "phone": "PRIVATE",
                        }
                    ]
                },
            },
        )

    result = challan_provider(handler).lookup("RJ45 CA 1879")

    assert result.records[0].status == "pending"
    assert result.records[0].amount == 1250
    assert result.records[0].currency is None
    assert result.records[0].date == date(2025, 3, 1)
    assert "owner_name" not in result.records[0].model_dump()


def test_challan_provider_accepts_explicit_empty_list_without_fabricating_records():
    provider = challan_provider(lambda _request: httpx.Response(200, json={"data": {"challans": []}}))

    result = provider.lookup("KA01AB1234")

    assert result.records == []


def test_challan_provider_partial_fields_remain_missing():
    provider = challan_provider(
        lambda _request: httpx.Response(
            200,
            json={"challans": [{"description": "No parking", "amount": 500}]},
        )
    )

    result = provider.lookup("KA01AB1234")

    assert result.records[0].violation_description == "No parking"
    assert result.records[0].date is None
    assert result.records[0].payment_status is None


def test_fastag_provider_normalizes_nested_supported_fields_only():
    provider = fastag_provider(
        lambda _request: httpx.Response(
            200,
            json={
                "result": {
                    "fastag": {
                        "tag_status": "active",
                        "issuing_bank": "Example Bank",
                        "issue_date": "2024-02-11",
                        "vehicle_class": "4W",
                        "account_number": "PRIVATE",
                        "mobile": "PRIVATE",
                    }
                }
            },
        )
    )

    result = provider.lookup("RJ45CA1879")

    assert result.status == "active"
    assert result.issuing_bank == "Example Bank"
    assert result.issue_date == date(2024, 2, 11)
    assert "account_number" not in result.model_dump()


def test_fastag_provider_preserves_null_fields_in_partial_response():
    provider = fastag_provider(
        lambda _request: httpx.Response(200, json={"data": {"issuing_bank": "Example Bank", "issue_date": None}})
    )

    result = provider.lookup("RJ45CA1879")

    assert result.issuing_bank == "Example Bank"
    assert result.issue_date is None
    assert result.status is None
    assert result.vehicle_class is None


@pytest.mark.parametrize("factory", [challan_provider, fastag_provider])
def test_service_providers_reject_invalid_registration_before_request(factory):
    requests = []
    provider = factory(lambda request: requests.append(request) or httpx.Response(200, json={}))

    with pytest.raises(VehicleLookupRejected):
        provider.lookup("not a registration")

    assert requests == []


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, VehicleLookupAuthenticationError),
        (403, VehicleLookupAuthenticationError),
        (429, VehicleLookupUnavailable),
        (503, VehicleLookupUnavailable),
        (404, VehicleLookupRejected),
    ],
)
def test_service_providers_map_http_errors(status_code, expected):
    provider = challan_provider(lambda _request: httpx.Response(status_code))

    with pytest.raises(expected):
        provider.lookup("RJ45CA1879")


def test_service_providers_handle_malformed_or_unexpected_payloads():
    malformed = challan_provider(lambda _request: httpx.Response(200, content=b"invalid JSON"))
    unexpected = fastag_provider(lambda _request: httpx.Response(200, json=[]))
    rejected = challan_provider(lambda _request: httpx.Response(200, json={"success": False}))

    with pytest.raises(VehicleLookupInvalidResponse):
        malformed.lookup("RJ45CA1879")
    with pytest.raises(VehicleLookupInvalidResponse):
        unexpected.lookup("RJ45CA1879")
    with pytest.raises(VehicleLookupRejected):
        rejected.lookup("RJ45CA1879")


def test_service_providers_handle_timeout_without_exposing_request_data():
    def timeout(_request):
        raise httpx.ReadTimeout("sensitive response")

    provider = fastag_provider(timeout)
    with pytest.raises(VehicleLookupUnavailable) as error:
        provider.lookup("RJ45CA1879")
    assert "sensitive" not in str(error.value)


def test_service_provider_factories_are_independently_disabled(monkeypatch):
    monkeypatch.setattr(settings, "challan_lookup_enabled", False)
    monkeypatch.setattr(settings, "fastag_lookup_enabled", False)
    assert get_challan_provider() is None
    assert get_fastag_provider() is None

    monkeypatch.setattr(settings, "challan_lookup_enabled", True)
    monkeypatch.setattr(settings, "challan_lookup_base_url", "https://challan.example/api")
    assert get_challan_provider() is not None
    assert get_fastag_provider() is None

    monkeypatch.setattr(settings, "fastag_lookup_enabled", True)
    monkeypatch.setattr(settings, "fastag_lookup_base_url", "https://fastag.example/api")
    assert get_fastag_provider() is not None
