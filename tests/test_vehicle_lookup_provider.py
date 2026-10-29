from datetime import date

import httpx
import pytest

from app.vehicle_lookup.provider import (
    VehicleDetailsHTTPProvider,
    VehicleLookupAuthenticationError,
    VehicleLookupInvalidResponse,
    VehicleLookupRejected,
    VehicleLookupUnavailable,
    get_vehicle_registration_provider,
)
from app.core.config import settings


def provider_for(handler):
    return VehicleDetailsHTTPProvider(
        "https://vehiclex.profilework239.workers.dev/",
        timeout_seconds=2,
        transport=httpx.MockTransport(handler),
    )


def test_provider_maps_nested_supported_fields_and_discards_sensitive_data():
    def handler(request):
        assert request.method == "GET"
        assert request.url.path == "/"
        assert request.url.params["regn"] == "RJ45CA1879"
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "vehicleDetails": {
                        "maker_description": "Maruti Suzuki",
                        "model_desc": "Swift",
                        "variant": "VXi",
                        "fuel_type": "PETROL",
                        "reg_date": "12-05-2021",
                        "manufacturing_year": "2021",
                        "vehicle_class": "Motor Car",
                        "rto_name": "Jaipur",
                        "fit_upto": "11-05-2036",
                        "pucc_upto": "2026-02-01",
                        "insurance_upto": "2026-08-15",
                        "blacklist_status": "N",
                        "owner_name": "Private Person",
                        "address": "Private Address",
                        "chassis_no": "PRIVATE-CHASSIS",
                        "engine_no": "PRIVATE-ENGINE",
                    }
                },
            },
        )

    details = provider_for(handler).lookup("RJ45CA1879")

    assert details.registration_number == "RJ45CA1879"
    assert details.make == "Maruti Suzuki"
    assert details.model == "Swift"
    assert details.variant == "VXi"
    assert details.registration_date == date(2021, 5, 12)
    assert details.manufacturing_year == 2021
    assert details.fitness_expiry == date(2036, 5, 11)
    assert details.blacklisted is False
    serialized = details.model_dump(mode="json")
    assert "owner_name" not in serialized
    assert "address" not in serialized
    assert "chassis_no" not in serialized
    assert "engine_no" not in serialized


def test_provider_partial_response_keeps_missing_fields_null():
    provider = provider_for(lambda _request: httpx.Response(200, json={"data": {"make": "Tata"}}))

    details = provider.lookup("KA01AB1234")

    assert details.make == "Tata"
    assert details.model is None
    assert details.fuel_type is None
    assert details.registration_date is None


def test_provider_rejects_invalid_registration_without_http_request():
    called = []
    provider = provider_for(lambda _request: called.append(True) or httpx.Response(200, json={"make": "Tata"}))

    with pytest.raises(VehicleLookupRejected):
        provider.lookup("not-a-registration")

    assert called == []


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(200, json=[]), VehicleLookupInvalidResponse),
        (httpx.Response(200, json={"success": False}), VehicleLookupRejected),
        (httpx.Response(200, content=b"not-json"), VehicleLookupInvalidResponse),
        (httpx.Response(401), VehicleLookupAuthenticationError),
        (httpx.Response(403), VehicleLookupAuthenticationError),
        (httpx.Response(429), VehicleLookupUnavailable),
        (httpx.Response(503), VehicleLookupUnavailable),
    ],
)
def test_provider_handles_unexpected_status_and_payload_safely(response, error):
    provider = provider_for(lambda _request: response)

    with pytest.raises(error):
        provider.lookup("RJ45CA1879")


def test_provider_handles_timeout_without_leaking_request_details():
    def timeout(_request):
        raise httpx.ReadTimeout("private request data")

    provider = provider_for(timeout)

    with pytest.raises(VehicleLookupUnavailable) as error:
        provider.lookup("RJ45CA1879")

    assert "private" not in str(error.value)


def test_provider_rejects_oversized_response_without_exposing_body(monkeypatch):
    monkeypatch.setattr(settings, "max_external_response_bytes", 16)
    provider = provider_for(
        lambda _request: httpx.Response(200, content=b"sensitive-response-payload")
    )

    with pytest.raises(VehicleLookupInvalidResponse) as error:
        provider.lookup("RJ45CA1879")

    assert "exceeded the allowed size" in str(error.value)
    assert "sensitive-response-payload" not in str(error.value)


def test_provider_factory_is_disabled_by_default_and_configurable(monkeypatch):
    monkeypatch.setattr(settings, "vehicle_lookup_enabled", False)
    assert get_vehicle_registration_provider() is None

    monkeypatch.setattr(settings, "vehicle_lookup_enabled", True)
    monkeypatch.setattr(settings, "vehicle_lookup_base_url", "https://lookup.example/")
    monkeypatch.setattr(settings, "vehicle_lookup_timeout_seconds", 3.0)
    provider = get_vehicle_registration_provider()
    assert isinstance(provider, VehicleDetailsHTTPProvider)
    assert provider.base_url == "https://lookup.example/"
    assert provider.timeout_seconds == 3.0
