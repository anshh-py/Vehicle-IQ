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


def test_provider_body_level_integer_statuses_on_http_200():
    base = {"status": "Found", "msg": "Vehicle details found"}
    cases = [
        ({**base, "status": 401}, VehicleLookupAuthenticationError),
        ({**base, "status": 429}, VehicleLookupUnavailable),
        ({**base, "status": 500}, VehicleLookupUnavailable),
    ]
    for payload, error in cases:
        with pytest.raises(error):
            provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")


def test_provider_body_level_explicit_lookup_rejection_on_http_200():
    base = {"status": "Found", "msg": "Vehicle details found"}
    rejection_values = ["not_found", "invalid_registration", "failed", "error"]
    for value in rejection_values:
        payload = {**base, "Data": {"status": value, "serviceResponse": {"status": "success", "results": []}}}
        with pytest.raises(VehicleLookupRejected):
            provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")


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


def test_provider_maps_observed_nested_vehiclex_response():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "Found",
                "msg": "Vehicle details found",
                "Data": {
                    "status": "success",
                    "manf_name": "Maruti Suzuki",
                    "model_name": "Swift",
                    "version_name": "VXi",
                    "fuel_type": "Petrol",
                    "cubic_cap": 1197,
                    "reg_year": "2021",
                    "manf_date": "01/01/2021",
                    "vehicle_regi_at": "12/05/21",
                    "serviceResponse": {
                        "status": "success",
                        "description": "Vehicle records retrieved",
                        "results": [
                            {
                                "vehicle": {
                                    "fla_maker_desc": "Tata Motors",
                                    "fla_model_desc": "Nexon",
                                    "fla_variant": "XZ+",
                                    "fla_fuel_type_desc": "Diesel",
                                    "regn_dt": "15/08/2022",
                                    "manu_yr": "2022",
                                    "vh_class_desc": "Motor Car",
                                    "rto_name": "Pune",
                                    "fit_upto": "14/08/2037",
                                    "pucc_upto": "2027-12-01",
                                    "blacklist_flag": "Y",
                                    "owner_name": "Must Not Persist",
                                    "chassis_no": "Must Not Persist",
                                    "engine_no": "Must Not Persist",
                                },
                                "insurance": {
                                    "insurance_upto": "10/10/2026",
                                    "policy_number": "Must Not Persist",
                                },
                            }
                        ],
                    },
                },
            },
        )

    details = provider_for(handler).lookup("MH12AB3456")

    assert details.registration_number == "MH12AB3456"
    # Vehicle-level fields take precedence over top-level Data fields.
    assert details.make == "Tata Motors"
    assert details.model == "Nexon"
    assert details.variant == "XZ+"
    assert details.fuel_type == "Diesel"
    assert details.registration_date == date(2022, 8, 15)
    assert details.manufacturing_year == 2022
    assert details.vehicle_class == "Motor Car"
    assert details.rto == "Pune"
    assert details.fitness_expiry == date(2037, 8, 14)
    assert details.puc_expiry == date(2027, 12, 1)
    assert details.insurance_expiry == date(2026, 10, 10)
    assert details.blacklisted is True
    serialized = details.model_dump(mode="json")
    assert "owner_name" not in serialized
    assert "chassis_no" not in serialized
    assert "engine_no" not in serialized
    assert "policy_number" not in serialized


def test_provider_falls_back_to_top_level_data_when_no_results():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "Found",
                "msg": "Found",
                "Data": {
                    "status": "success",
                    "manf_name": "Mahindra",
                    "model_name": "Thar",
                    "version_name": "AXL",
                    "fuel_type": "Petrol",
                    "reg_year": "2020",
                    "vehicle_regi_at": "5/6/2020",
                    "serviceResponse": {
                        "status": "success",
                        "results": [],
                    },
                },
            },
        )

    details = provider_for(handler).lookup("GJ01Z9999")

    assert details.make == "Mahindra"
    assert details.model == "Thar"
    assert details.variant == "AXL"
    assert details.fuel_type == "Petrol"
    assert details.manufacturing_year == 2020
    assert details.registration_date == date(2020, 6, 5)


def test_provider_partial_response_keeps_missing_fields_null_with_nested_shape():
    provider = provider_for(
        lambda _request: httpx.Response(
            200,
            json={
                "status": "Found",
                "Data": {
                    "status": "success",
                    "manf_name": "Only Make",
                    "serviceResponse": {"status": "success", "results": [{}]},
                },
            },
        )
    )

    details = provider.lookup("KA01AB1234")

    assert details.make == "Only Make"
    assert details.model is None
    assert details.registration_date is None
    assert details.blacklisted is None


def test_provider_parses_observed_date_formats():
    cases = [
        ("12/05/2021", date(2021, 5, 12)),
        ("12/05/21", date(2021, 5, 12)),
        ("5/6/2020", date(2020, 6, 5)),
        ("not-a-date", None),
        ("", None),
        (None, None),
    ]

    for raw, expected in cases:
        payload = {
            "status": "Found",
            "Data": {
                "status": "success",
                "manf_name": "Make",
                "vehicle_regi_at": raw,
                "serviceResponse": {"status": "success", "results": []},
            },
        }
        details = provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")
        assert details.registration_date == expected, f"failed for raw={raw!r}"


def test_provider_rejects_provider_declared_failure():
    failures = [
        {
            "status": "Found",
            "Data": {"status": "failed", "serviceResponse": {"status": "success", "results": []}},
        },
        {
            "status": "Found",
            "Data": {
                "status": "success",
                "serviceResponse": {"status": "error", "description": "Vehicle not found", "results": []},
            },
        },
        {
            "status": "Found",
            "Data": {
                "status": "success",
                "serviceResponse": {
                    "status": "success",
                    "description": "not_found",
                    "results": [],
                },
            },
        },
    ]

    for payload in failures:
        with pytest.raises(VehicleLookupRejected):
            provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")


def test_provider_rejects_missing_or_empty_results_without_usable_data():
    empty_cases = [
        {"status": "Found", "msg": "Found", "Data": {"status": "success"}},
        {"status": "Found", "msg": "Found", "Data": {"status": "success", "serviceResponse": {"status": "success"}}},
        {"status": "Found", "msg": "Found", "Data": {"status": "success", "serviceResponse": {"status": "success", "results": []}}},
        {"status": "Found", "msg": "Found", "Data": {"status": "success", "serviceResponse": {"status": "success", "results": [{}]}}},
    ]

    for payload in empty_cases:
        with pytest.raises(VehicleLookupInvalidResponse):
            provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")


def test_provider_tolerates_null_optional_nested_fields():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "Found",
                "Data": {
                    "status": "success",
                    "manf_name": None,
                    "model_name": None,
                    "serviceResponse": {
                        "status": "success",
                        "results": [
                            {
                                "vehicle": {
                                    "fla_maker_desc": None,
                                    "fla_model_desc": "Swift",
                                    "regn_dt": None,
                                    "blacklist_flag": None,
                                },
                                "insurance": {"insurance_upto": None},
                            }
                        ],
                    },
                },
            },
        )

    details = provider_for(handler).lookup("DL01AB1234")

    assert details.make is None
    assert details.model == "Swift"
    assert details.registration_date is None
    assert details.blacklisted is None


def test_provider_accepts_vehiclex_success_status_true_and_code_100():
    """VehicleX success: Data.status=true (bool), serviceResponse.status=100 (int),
    description='Record found'. These are provider-specific success codes, not HTTP."""
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "Found",
                "msg": "Vehicle details found",
                "Data": {
                    "status": True,
                    "manf_name": "Maruti Suzuki",
                    "model_name": "Swift",
                    "version_name": "VXi",
                    "fuel_type": "Petrol",
                    "reg_year": "2021",
                    "vehicle_regi_at": "12/05/21",
                    "serviceResponse": {
                        "status": 100,
                        "description": "Record found",
                        "results": [
                            {
                                "vehicle": {
                                    "fla_maker_desc": "Tata Motors",
                                    "fla_model_desc": "Nexon",
                                    "fla_variant": "XZ+",
                                    "fla_fuel_type_desc": "Diesel",
                                    "regn_dt": "15/08/2022",
                                    "manu_yr": "2022",
                                    "vh_class_desc": "Motor Car",
                                    "rto_name": "Pune",
                                    "fit_upto": "14/08/2037",
                                    "pucc_upto": "2027-12-01",
                                    "blacklist_flag": "Y",
                                },
                                "insurance": {
                                    "insurance_upto": "10/10/2026",
                                },
                            }
                        ],
                    },
                },
            },
        )

    details = provider_for(handler).lookup("MH12AB3456")

    assert details.registration_number == "MH12AB3456"
    assert details.make == "Tata Motors"
    assert details.model == "Nexon"
    assert details.variant == "XZ+"
    assert details.fuel_type == "Diesel"
    assert details.registration_date == date(2022, 8, 15)
    assert details.manufacturing_year == 2022
    assert details.vehicle_class == "Motor Car"
    assert details.rto == "Pune"
    assert details.fitness_expiry == date(2037, 8, 14)
    assert details.puc_expiry == date(2027, 12, 1)
    assert details.insurance_expiry == date(2026, 10, 10)
    assert details.blacklisted is True
    serialized = details.model_dump(mode="json")
    assert "owner_name" not in serialized
    assert "chassis_no" not in serialized
    assert "engine_no" not in serialized


def test_provider_accepts_top_level_data_without_service_response():
    def handler(_request):
        return httpx.Response(
            200,
            json={
                "status": "Found",
                "msg": "Found",
                "Data": {
                    "status": True,
                    "manf_name": "Mahindra",
                    "model_name": "Thar",
                    "version_name": "AXL",
                    "fuel_type": "Petrol",
                    "reg_year": "2020",
                    "vehicle_regi_at": "5/6/2020",
                },
            },
        )

    details = provider_for(handler).lookup("GJ01Z9999")

    assert details.make == "Mahindra"
    assert details.model == "Thar"
    assert details.variant == "AXL"
    assert details.fuel_type == "Petrol"
    assert details.manufacturing_year == 2020
    assert details.registration_date == date(2020, 6, 5)


def test_provider_conflicting_status_indicators_are_rejected():
    """Error status wins over success status — conservative interpretation."""
    # Data.status=true but serviceResponse.status=error → rejected
    payload = {
        "status": "Found",
        "Data": {
            "status": True,
            "manf_name": "Maruti Suzuki",
            "serviceResponse": {
                "status": "error",
                "description": "Vehicle not found",
                "results": [],
            },
        },
    }
    with pytest.raises(VehicleLookupRejected):
        provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")

    # Data.status=false → rejected even with vehicle fields present
    payload = {
        "status": "Found",
        "Data": {
            "status": False,
            "manf_name": "Maruti Suzuki",
            "serviceResponse": {
                "status": 100,
                "results": [{"vehicle": {"fla_maker_desc": "Tata"}}],
            },
        },
    }
    with pytest.raises(VehicleLookupRejected):
        provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")


def test_provider_integer_status_500_is_unavailable_not_rejected():
    """HTTP 200 with body-level int 500 → provider unavailable, not rejected."""
    payload = {
        "status": "Found",
        "Data": {
            "status": 500,
            "serviceResponse": {"status": 100, "results": []},
        },
    }
    with pytest.raises(VehicleLookupUnavailable):
        provider_for(lambda _request: httpx.Response(200, json=payload)).lookup("TN10AB1234")

