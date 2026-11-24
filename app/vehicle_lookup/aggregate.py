"""Aggregate the vehicle-data lookup results into one normalized vehicle record.

The three data sources (registration/vehicle details, challan, FASTag) are
fetched separately by dedicated providers. This module only *combines* the
already-normalized results into a single record that:
  - preserves each field's source and retrieval status,
  - distinguishes unavailable / not found / invalid response / provider failure,
  - tolerates partial provider failures without discarding successful results,
  - never overwrites valid data with null/failed responses,
  - never fabricates data.

It is intentionally a pure function over the report-level lookup structures so
it can be unit-tested without a database.
"""

from __future__ import annotations

from datetime import date as Date
from decimal import Decimal
from typing import Any

from app.schemas.vehicle_aggregate import (
    NormalizedVehicleRecord,
    VehicleAggregateChallan,
    VehicleAggregateFastag,
    VehicleAggregateInsurance,
    VehicleAggregateSource,
)

_REGISTRATION_ALLOWED = {
    "registration_number",
    "make",
    "model",
    "variant",
    "fuel_type",
    "registration_date",
    "manufacturing_year",
    "vehicle_class",
    "registration_valid_until",
    "rto",
    "fitness_expiry",
    "puc_expiry",
    "insurance_expiry",
    "blacklisted",
    "engine_cc",
    "seats",
    "max_power_bhp",
    "transmission",
    "owner_count",
    "mileage_kmpl",
}


def _classify_service(raw: Any) -> str:
    """Classify a provider block's retrieval status.

    - ``None`` / an absent source -> "unavailable" (genuinely no data).
    - A non-dict payload, or a dict whose ``status`` is missing or not one of
      the known values -> "malformed" (the provider sent unusable data).
    - Otherwise the declared, known status is used as-is.
    """
    if raw is None:
        return "unavailable"
    if not isinstance(raw, dict):
        return "malformed"
    status = raw.get("status")
    if status in {"available", "partial", "unavailable", "failed", "not_found"}:
        return status
    return "malformed"


def _service_provider(raw: Any) -> str | None:
    if isinstance(raw, dict):
        provider = raw.get("provider")
        return provider if isinstance(provider, str) and provider else None
    return None


def _service_error_code(raw: Any) -> str | None:
    if isinstance(raw, dict):
        code = raw.get("error_code")
        return code if isinstance(code, str) and code else None
    return None


def _as_date(value: Any) -> Date | None:
    if isinstance(value, Date):
        return value
    if isinstance(value, str):
        for pattern in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                from datetime import datetime

                return datetime.strptime(value, pattern).date()
            except ValueError:
                continue
    return None


def _coerce_decimal(value: Any) -> Decimal | None:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        try:
            return Decimal(str(value))
        except Exception:
            return None
    if isinstance(value, str):
        try:
            return Decimal(value.replace(",", ""))
        except Exception:
            return None
    return None


def _registration_record(vehicle_lookup: Any) -> tuple[dict[str, Any], VehicleAggregateSource, dict[str, str]]:
    """Turn the vehicle_details_lookup report block into specs + provenance.

    Returns (specs, source, provenance).
    """
    source = VehicleAggregateSource(
        status=_classify_service(vehicle_lookup),
        provider=_service_provider(vehicle_lookup),
        error_code=_service_error_code(vehicle_lookup),
    )
    specs: dict[str, Any] = {}
    provenance: dict[str, str] = {}

    if not isinstance(vehicle_lookup, dict):
        return specs, source, provenance

    details = vehicle_lookup.get("details") if isinstance(vehicle_lookup.get("details"), dict) else None

    if details is None:
        return specs, source, provenance

    for key in _REGISTRATION_ALLOWED:
        if key in details and details[key] is not None:
            specs[key] = details[key]
    for key, value in specs.items():
        provenance[key] = "vehicle_api"

    # Merge in any stored provenance overrides (e.g. manual user entries).
    stored_provenance = vehicle_lookup.get("field_provenance")
    if isinstance(stored_provenance, dict):
        for key, src in stored_provenance.items():
            if isinstance(src, str) and src in {"vehicle_api", "user_input"}:
                provenance[key] = src

    return specs, source, provenance


def _insurance_block(specs: dict[str, Any], reg_source: VehicleAggregateSource) -> VehicleAggregateInsurance:
    expiry = specs.get("insurance_expiry")
    parsed = _as_date(expiry)
    if parsed is not None:
        return VehicleAggregateInsurance(status="available", expiry=parsed, provider=reg_source.provider)
    if reg_source.status in {"available", "partial"}:
        # Registration succeeded but no insurance field was supplied: the provider
        # did not provide insurance data — not a failure.
        return VehicleAggregateInsurance(status="not_found", expiry=None, provider=reg_source.provider)
    if reg_source.status == "not_found":
        return VehicleAggregateInsurance(status="not_found", provider=reg_source.provider)
    return VehicleAggregateInsurance(status=reg_source.status, provider=reg_source.provider)


def _challan_block(raw: Any) -> VehicleAggregateChallan:
    status = _classify_service(raw)
    provider = _service_provider(raw)
    error_code = _service_error_code(raw)
    details = raw.get("details") if isinstance(raw, dict) and isinstance(raw.get("details"), dict) else None

    records = details.get("records") if details is not None else None
    record_list = records if isinstance(records, list) else []

    total: Decimal | None = Decimal("0")
    unpaid = 0
    for record in record_list:
        if not isinstance(record, dict):
            continue
        amount = _coerce_decimal(record.get("amount"))
        if amount is not None:
            total = (total or Decimal("0")) + amount
        payment = record.get("payment_status")
        if isinstance(payment, str) and payment.strip().casefold() in {
            "unpaid",
            "pending",
            "due",
            "not paid",
        }:
            unpaid += 1

    return VehicleAggregateChallan(
        status=status,
        provider=provider,
        error_code=error_code,
        records=len(record_list),
        total_amount_inr=total if record_list else None,
        unpaid_count=unpaid,
    )


def _fastag_block(raw: Any) -> VehicleAggregateFastag:
    status = _classify_service(raw)
    provider = _service_provider(raw)
    error_code = _service_error_code(raw)
    details = raw.get("details") if isinstance(raw, dict) and isinstance(raw.get("details"), dict) else None

    bank = details.get("issuing_bank") if details is not None else None
    vehicle_class = details.get("vehicle_class") if details is not None else None
    issue_date = _as_date(details.get("issue_date")) if details is not None else None

    return VehicleAggregateFastag(
        status=status,
        provider=provider,
        error_code=error_code,
        issuing_bank=bank if isinstance(bank, str) and bank else None,
        issue_date=issue_date,
        vehicle_class=vehicle_class if isinstance(vehicle_class, str) and vehicle_class else None,
    )


def _overall(
    registration: VehicleAggregateSource,
    challan: VehicleAggregateChallan,
    fastag: VehicleAggregateFastag,
) -> str:
    states = {registration.status, challan.status, fastag.status}
    if states <= {"available"}:
        return "available"
    if states <= {"available", "not_found", "partial"}:
        return "partial" if "partial" in states else "available"
    # A failed or malformed source is treated as a failure of that source.
    if "failed" in states or "malformed" in states:
        return "partial" if states & {"available", "partial", "not_found"} else "failed"
    return "unavailable"


def build_normalized_vehicle_record(
    *,
    assessment_id: Any = None,
    vehicle_lookup: Any = None,
    challan_lookup: Any = None,
    fastag_lookup: Any = None,
) -> NormalizedVehicleRecord:
    """Combine report-level lookup blocks into one normalized vehicle record."""
    specs, reg_source, provenance = _registration_record(vehicle_lookup)
    insurance = _insurance_block(specs, reg_source)
    challan = _challan_block(challan_lookup)
    fastag = _fastag_block(fastag_lookup)

    registration_number = specs.get("registration_number")
    if registration_number is None and isinstance(vehicle_lookup, dict):
        details = vehicle_lookup.get("details")
        if isinstance(details, dict):
            registration_number = details.get("registration_number")

    errors: list[dict[str, str]] = []
    for label, src in (("registration", reg_source), ("challan", challan), ("fastag", fastag)):
        if src.status in {"failed", "unavailable"} and src.error_code:
            errors.append({"source": label, "code": src.error_code})

    sources = {
        "registration": reg_source,
        "challan": VehicleAggregateSource(status=challan.status, provider=challan.provider, error_code=challan.error_code),
        "fastag": VehicleAggregateSource(status=fastag.status, provider=fastag.provider, error_code=fastag.error_code),
    }

    return NormalizedVehicleRecord(
        assessment_id=assessment_id,
        registration_number=registration_number if isinstance(registration_number, str) else None,
        status=_overall(reg_source, challan, fastag),
        vehicle_specs=specs,
        insurance=insurance,
        challan=challan,
        fastag=fastag,
        field_provenance=provenance,
        sources=sources,
        errors=errors,
    )
