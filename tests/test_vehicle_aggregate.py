"""Focused unit tests for the M3 aggregation builder.

Covers app/vehicle_lookup/aggregate.py and app/schemas/vehicle_aggregate.py.

These tests target the pure normalization logic only. They deliberately do NOT
verify any wiring into the report/API/workflow, which is intentionally not yet
connected for M3.
"""

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.vehicle_aggregate import (
    NormalizedVehicleRecord,
    VehicleAggregateChallan,
    VehicleAggregateFastag,
    VehicleAggregateInsurance,
    VehicleAggregateSource,
)
from app.vehicle_lookup.aggregate import build_normalized_vehicle_record


# ---------------------------------------------------------------------------
# Input-block helpers (report-level lookup shapes the builder consumes).
# ---------------------------------------------------------------------------

def reg_block(status="available", provider="vehiclex_candidate", error_code=None, details=None, provenance=None):
    block: dict = {"status": status}
    if provider is not None:
        block["provider"] = provider
    if error_code is not None:
        block["error_code"] = error_code
    if details is not None:
        block["details"] = details
    if provenance is not None:
        block["field_provenance"] = provenance
    return block


def challan_block(status="available", provider="challan-provider", error_code=None, records=None, details=None):
    block: dict = {"status": status}
    if provider is not None:
        block["provider"] = provider
    if error_code is not None:
        block["error_code"] = error_code
    if details is not None:
        block["details"] = details
    elif records is not None:
        block["details"] = {"records": records}
    return block


def fastag_block(status="available", provider="fastag-provider", error_code=None, details=None):
    block: dict = {"status": status}
    if provider is not None:
        block["provider"] = provider
    if error_code is not None:
        block["error_code"] = error_code
    if details is not None:
        block["details"] = details
    return block


def full_details(**overrides):
    details = {
        "registration_number": "RJ45CA1879",
        "make": "Maruti Suzuki",
        "model": "Swift",
        "fuel_type": "Petrol",
        "registration_date": "2021-05-12",
        "vehicle_class": "Motor Car",
        "insurance_expiry": "2026-05-11",
    }
    details.update(overrides)
    return details


def build(**kwargs):
    return build_normalized_vehicle_record(assessment_id=uuid4(), **kwargs)


# ---------------------------------------------------------------------------
# Schema behavior.
# ---------------------------------------------------------------------------

class TestSchema:
    def test_source_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VehicleAggregateSource(status="available", bogus="x")

    def test_insurance_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VehicleAggregateInsurance(status="available", bogus=1)

    def test_challan_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VehicleAggregateChallan(status="available", bogus=1)

    def test_fastag_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            VehicleAggregateFastag(status="available", bogus=1)

    def test_normalized_record_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            NormalizedVehicleRecord(wise="x")

    def test_malformed_is_valid_status(self):
        # malformed is a first-class, distinct status.
        assert VehicleAggregateSource(status="malformed").status == "malformed"

    def test_default_record_is_empty_and_unavailable(self):
        rec = NormalizedVehicleRecord()
        assert rec.status == "unavailable"
        assert rec.vehicle_specs == {}
        assert rec.field_provenance == {}
        assert rec.sources == {}
        assert rec.errors == []
        assert rec.insurance.status == "unavailable"
        assert rec.challan.status == "unavailable"
        assert rec.fastag.status == "unavailable"

    def test_challan_negative_total_rejected(self):
        with pytest.raises(ValidationError):
            VehicleAggregateChallan(status="available", total_amount_inr=Decimal("-1"))


# ---------------------------------------------------------------------------
# Registration aggregation.
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_whitelist_filters_unexpected_keys(self):
        details = full_details(extra_bogus="should-be-dropped", secret="nope")
        rec = build(
            vehicle_lookup=reg_block(details=details),
            challan_lookup=challan_block(),
            fastag_lookup=fastag_block(),
        )
        assert "extra_bogus" not in rec.vehicle_specs
        assert "secret" not in rec.vehicle_specs
        assert rec.vehicle_specs["make"] == "Maruti Suzuki"

    def test_none_detail_values_are_skipped(self):
        details = full_details(make=None, model=None, manufacturing_year=2021)
        rec = build(vehicle_lookup=reg_block(details=details))
        assert "make" not in rec.vehicle_specs
        assert "model" not in rec.vehicle_specs
        assert rec.vehicle_specs["manufacturing_year"] == 2021

    def test_registration_number_extracted_from_details(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()))
        assert rec.registration_number == "RJ45CA1879"

    def test_provenance_all_vehicle_api(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()))
        assert all(v == "vehicle_api" for v in rec.field_provenance.values())
        assert rec.field_provenance["make"] == "vehicle_api"

    def test_stored_user_input_override_wins(self):
        details = full_details()
        provenance = {"make": "user_input", "model": "vehicle_api"}
        rec = build(vehicle_lookup=reg_block(details=details, provenance=provenance))
        assert rec.field_provenance["make"] == "user_input"
        assert rec.field_provenance["model"] == "vehicle_api"

    def test_invalid_provenance_override_ignored(self):
        details = full_details()
        provenance = {"make": "user_input", "model": "hacker_supplied", "fuel_type": "whatever"}
        rec = build(vehicle_lookup=reg_block(details=details, provenance=provenance))
        assert rec.field_provenance["make"] == "user_input"
        # Invalid overrides fall back to the default vehicle_api.
        assert rec.field_provenance["model"] == "vehicle_api"
        assert rec.field_provenance["fuel_type"] == "vehicle_api"

    def test_malformed_non_dict_registration(self):
        rec = build(vehicle_lookup="not-a-dict", challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.sources["registration"].status == "malformed"
        assert rec.vehicle_specs == {}
        assert rec.registration_number is None

    def test_registration_none_is_unavailable(self):
        rec = build(vehicle_lookup=None, challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.sources["registration"].status == "unavailable"

    def test_registration_dict_missing_status_is_malformed(self):
        rec = build(vehicle_lookup={"provider": "x", "details": full_details()})
        assert rec.sources["registration"].status == "malformed"

    def test_registration_unknown_status_is_malformed(self):
        rec = build(vehicle_lookup=reg_block(status="no-such-status", details=full_details()))
        assert rec.sources["registration"].status == "malformed"

    def test_registration_not_found_preserved(self):
        rec = build(vehicle_lookup=reg_block(status="not_found", details=None))
        assert rec.sources["registration"].status == "not_found"

    def test_registration_failed_without_error_code_preserved(self):
        rec = build(vehicle_lookup=reg_block(status="failed", error_code=None, details=None))
        assert rec.sources["registration"].status == "failed"
        # No fabricated error code.
        assert rec.sources["registration"].error_code is None


# ---------------------------------------------------------------------------
# Insurance (derived from the registration block).
# ---------------------------------------------------------------------------

class TestInsurance:
    def test_insurance_available_when_expiry_parsed(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()))
        assert rec.insurance.status == "available"
        assert rec.insurance.expiry == date(2026, 5, 11)
        assert rec.insurance.provider == "vehiclex_candidate"

    def test_insurance_date_alt_format_parsed(self):
        details = full_details(insurance_expiry="11-05-2026")
        rec = build(vehicle_lookup=reg_block(details=details))
        assert rec.insurance.status == "available"
        assert rec.insurance.expiry == date(2026, 5, 11)

    def test_insurance_not_found_when_registration_ok_but_missing(self):
        details = full_details(insurance_expiry=None)
        rec = build(vehicle_lookup=reg_block(details=details))
        assert rec.insurance.status == "not_found"
        assert rec.insurance.expiry is None

    def test_insurance_not_found_when_registration_not_found(self):
        rec = build(vehicle_lookup=reg_block(status="not_found", details=None))
        assert rec.insurance.status == "not_found"

    def test_insurance_failed_inherits_registration_failure(self):
        rec = build(vehicle_lookup=reg_block(status="failed", error_code="UPSTREAM_DOWN", details=None))
        assert rec.insurance.status == "failed"

    def test_insurance_unavailable_inherits_registration_unavailable(self):
        rec = build(vehicle_lookup=None)
        assert rec.insurance.status == "unavailable"

    def test_insurance_malformed_inherits_registration_malformed(self):
        rec = build(vehicle_lookup="garbage")
        assert rec.insurance.status == "malformed"


# ---------------------------------------------------------------------------
# Challan aggregation.
# ---------------------------------------------------------------------------

class TestChallan:
    def test_challan_available_with_records(self):
        records = [
            {"amount": "500", "payment_status": "Unpaid"},
            {"amount": 1200.5, "payment_status": "paid"},
            {"amount": Decimal("80"), "payment_status": "PENDING"},
            {"amount": "1,000", "payment_status": "cleared"},
            {"not_a_dict": True},
            {"amount": "x", "payment_status": "paid"},
        ]
        rec = build(challan_lookup=challan_block(records=records))
        assert rec.challan.status == "available"
        assert rec.challan.records == 6  # counted even when a record is skipped for the total
        assert rec.challan.total_amount_inr == Decimal("2780.5")
        assert rec.challan.unpaid_count == 2

    def test_challan_empty_records(self):
        rec = build(challan_lookup=challan_block(records=[]))
        assert rec.challan.records == 0
        assert rec.challan.total_amount_inr is None
        assert rec.challan.unpaid_count == 0

    def test_challan_missing_details(self):
        rec = build(challan_lookup=challan_block(details=None))
        assert rec.challan.records == 0
        assert rec.challan.total_amount_inr is None

    def test_challan_malformed_non_dict(self):
        rec = build(challan_lookup="bogus-payload")
        assert rec.challan.status == "malformed"
        assert rec.challan.records == 0

    def test_challan_none_is_unavailable(self):
        rec = build(challan_lookup=None)
        assert rec.challan.status == "unavailable"

    def test_challan_not_found_preserved(self):
        rec = build(challan_lookup=challan_block(status="not_found"))
        assert rec.challan.status == "not_found"

    def test_challan_payment_status_normalization(self):
        records = [
            {"amount": 100, "payment_status": "due"},
            {"amount": 100, "payment_status": "Not Paid"},
            {"amount": 100, "payment_status": "paid"},
        ]
        rec = build(challan_lookup=challan_block(records=records))
        assert rec.challan.unpaid_count == 2

    def test_challan_total_excludes_unparsable_amounts(self):
        records = [
            {"amount": 100, "payment_status": "paid"},
            {"amount": "n/a", "payment_status": "paid"},
            {"amount": None, "payment_status": "paid"},
        ]
        rec = build(challan_lookup=challan_block(records=records))
        assert rec.challan.total_amount_inr == Decimal("100")


# ---------------------------------------------------------------------------
# FASTag aggregation.
# ---------------------------------------------------------------------------

class TestFastag:
    def test_fastag_available_with_details(self):
        details = {"issuing_bank": "ICICI", "issue_date": "2023-04-04", "vehicle_class": "N1"}
        rec = build(fastag_lookup=fastag_block(details=details))
        assert rec.fastag.status == "available"
        assert rec.fastag.issuing_bank == "ICICI"
        assert rec.fastag.issue_date == date(2023, 4, 4)
        assert rec.fastag.vehicle_class == "N1"

    def test_fastag_missing_details(self):
        rec = build(fastag_lookup=fastag_block(details=None))
        assert rec.fastag.issuing_bank is None
        assert rec.fastag.issue_date is None
        assert rec.fastag.vehicle_class is None

    def test_fastag_malformed_non_dict(self):
        rec = build(fastag_lookup=[1, 2, 3])
        assert rec.fastag.status == "malformed"

    def test_fastag_none_is_unavailable(self):
        rec = build(fastag_lookup=None)
        assert rec.fastag.status == "unavailable"

    def test_fastag_not_found_preserved(self):
        rec = build(fastag_lookup=fastag_block(status="not_found"))
        assert rec.fastag.status == "not_found"

    def test_fastag_non_string_fields_normalized(self):
        details = {"issuing_bank": "", "vehicle_class": 123}
        rec = build(fastag_lookup=fastag_block(details=details))
        assert rec.fastag.issuing_bank is None
        assert rec.fastag.vehicle_class is None


# ---------------------------------------------------------------------------
# Error collection and no-fabrication guarantees.
# ---------------------------------------------------------------------------

class TestErrors:
    def test_failed_with_error_code_collected(self):
        rec = build(
            vehicle_lookup=reg_block(status="failed", error_code="UPSTREAM_DOWN", details=None),
            challan_lookup=challan_block(),
            fastag_lookup=fastag_block(),
        )
        assert {"source": "registration", "code": "UPSTREAM_DOWN"} in rec.errors

    def test_failed_without_error_code_not_fabricated(self):
        rec = build(
            vehicle_lookup=reg_block(status="failed", error_code=None, details=None),
            challan_lookup=challan_block(),
            fastag_lookup=fastag_block(),
        )
        assert rec.sources["registration"].status == "failed"
        assert rec.errors == []  # status preserved, no invented code

    def test_malformed_does_not_fabricate_error(self):
        rec = build(vehicle_lookup="garbage", challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.sources["registration"].status == "malformed"
        assert rec.errors == []

    def test_unavailable_without_error_code_no_error_entry(self):
        rec = build(vehicle_lookup=None, challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.errors == []

    def test_multi_source_errors_collected(self):
        rec = build(
            vehicle_lookup=reg_block(status="failed", error_code="V1", details=None),
            challan_lookup=challan_block(status="failed", error_code="C1"),
            fastag_lookup=fastag_block(status="failed", error_code="F1"),
        )
        codes = {e["code"] for e in rec.errors}
        assert codes == {"V1", "C1", "F1"}


# ---------------------------------------------------------------------------
# Overall status collapse.
# ---------------------------------------------------------------------------

class TestOverall:
    def test_all_available(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()), challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.status == "available"

    def test_available_and_not_found_is_available(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()), challan_lookup=challan_block(status="not_found"), fastag_lookup=fastag_block())
        assert rec.status == "available"

    def test_partial_present(self):
        rec = build(vehicle_lookup=reg_block(details=full_details()), challan_lookup=challan_block(status="partial"), fastag_lookup=fastag_block())
        assert rec.status == "partial"

    def test_failed_plus_available_is_partial(self):
        rec = build(vehicle_lookup=reg_block(status="failed", error_code="E", details=None), challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.status == "partial"

    def test_all_failed_is_failed(self):
        rec = build(
            vehicle_lookup=reg_block(status="failed", error_code="E1", details=None),
            challan_lookup=challan_block(status="failed", error_code="E2"),
            fastag_lookup=fastag_block(status="failed", error_code="E3"),
        )
        assert rec.status == "failed"

    def test_malformed_plus_available_is_partial(self):
        rec = build(vehicle_lookup="garbage", challan_lookup=challan_block(), fastag_lookup=fastag_block())
        assert rec.status == "partial"

    def test_all_malformed_is_failed(self):
        rec = build(vehicle_lookup="garbage", challan_lookup=[1], fastag_lookup="nope")
        assert rec.status == "failed"

    def test_all_unavailable_is_unavailable(self):
        rec = build()
        assert rec.status == "unavailable"

    def test_malformed_plus_not_found_is_partial(self):
        rec = build(
            vehicle_lookup="garbage",
            challan_lookup=challan_block(status="not_found"),
            fastag_lookup=fastag_block(status="not_found"),
        )
        assert rec.status == "partial"
