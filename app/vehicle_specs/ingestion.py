import csv
import io
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import VehicleSpecification
from app.vehicle_specs.repository import VehicleSpecificationRepository
from app.vehicle_specs.sources import get_source, is_approved_source

SUPPORTED_FUELS = {"petrol", "diesel", "cng", "lpg", "electric"}
SUPPORTED_TRANSMISSIONS = {"manual", "automatic", "amt", "cvt", "dct"}
VALID_VERIFICATION_STATUSES = {"unverified", "pending", "verified"}

REQUIRED_IDENTITY_FIELDS = ("manufacturer", "model")
NUMERIC_RANGES = {
    "engine_cc": (1, 10000),
    "max_power_bhp": (0.1, 2000),
    "mileage_kmpl": (0.1, 100),
    "seats": (1, 9),
    "owner_count": (1, 20),
}


@dataclass
class ImportSummary:
    created: int = 0
    skipped: int = 0
    rejected: int = 0
    conflicts: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "created": self.created,
            "skipped": self.skipped,
            "rejected": self.rejected,
            "conflicts": self.conflicts,
            "errors": self.errors,
            "dry_run": self.dry_run,
        }
def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_int(value: Any, field: str) -> int | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _parse_float(value: Any, field: str) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number") from exc


def _validate_row(row: dict[str, Any], row_number: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    normalized: dict[str, Any] = {}

    manufacturer = _clean(row.get("manufacturer"))
    model = _clean(row.get("model"))
    if not manufacturer:
        errors.append("manufacturer is required")
    if not model:
        errors.append("model is required")
    normalized["make"] = manufacturer
    normalized["model"] = model
    normalized["variant"] = _clean(row.get("variant"))
    normalized["fuel_type"] = _clean(row.get("fuel_type"))

    year_start = None
    year_end = None
    try:
        year_start = _parse_int(row.get("year_start"), "year_start")
        year_end = _parse_int(row.get("year_end"), "year_end")
    except ValueError as exc:
        errors.append(str(exc))
    if year_start is not None and (year_start < 1900 or year_start > 2100):
        errors.append("year_start must be between 1900 and 2100")
    if year_end is not None and (year_end < 1900 or year_end > 2100):
        errors.append("year_end must be between 1900 and 2100")
    if year_start is not None and year_end is not None and year_end < year_start:
        errors.append("year_end must be greater than or equal to year_start")
    normalized["year_start"] = year_start
    normalized["year_end"] = year_end

    for field in ("engine_cc", "seats", "owner_count"):
        try:
            normalized[field] = _parse_int(row.get(field), field)
        except ValueError as exc:
            errors.append(str(exc))
            normalized[field] = None
    for field in ("max_power_bhp", "mileage_kmpl"):
        try:
            normalized[field] = _parse_float(row.get(field), field)
        except ValueError as exc:
            errors.append(str(exc))
            normalized[field] = None

    for field, (low, high) in NUMERIC_RANGES.items():
        value = normalized.get(field)
        if value is not None and not (low <= value <= high):
            errors.append(f"{field} must be between {low} and {high}")

    transmission = _clean(row.get("transmission"))
    if transmission is not None:
        normalized_transmission = transmission.casefold()
        if normalized_transmission not in SUPPORTED_TRANSMISSIONS:
            errors.append("transmission is not supported")
        normalized["transmission"] = normalized_transmission
    else:
        normalized["transmission"] = None

    if normalized["fuel_type"] is not None:
        fuel = normalized["fuel_type"].casefold()
        if fuel not in SUPPORTED_FUELS:
            errors.append("fuel_type is not supported")
        normalized["fuel_type"] = fuel

    source_name = _clean(row.get("source_name"))
    source_reference = _clean(row.get("source_reference"))
    if not source_name:
        errors.append("source_name is required")
    if not source_reference:
        errors.append("source_reference is required")
    normalized["source_name"] = source_name
    normalized["source_reference"] = source_reference

    verification_status = (_clean(row.get("verification_status")) or "unverified").casefold()
    if verification_status not in VALID_VERIFICATION_STATUSES:
        errors.append("verification_status must be unverified, pending, or verified")
    if verification_status == "verified":
        if not is_approved_source(source_name):
            errors.append("verified status requires an approved source")
        if not source_reference:
            errors.append("verified status requires a source reference")
    normalized["verification_status"] = verification_status

    try:
        confidence = _parse_float(row.get("confidence"), "confidence")
    except ValueError as exc:
        errors.append(str(exc))
        confidence = None
    if confidence is not None and not (0 <= confidence <= 1):
        errors.append("confidence must be between 0 and 1")
    normalized["confidence"] = confidence

    field_sources_raw = _clean(row.get("field_sources"))
    if field_sources_raw:
        parsed = {}
        for part in field_sources_raw.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()
        normalized["field_sources"] = parsed
    else:
        normalized["field_sources"] = {}

    allowed_provenance_fields = {
        "make", "model", "variant", "fuel_type", "year_start", "year_end",
        "engine_cc", "max_power_bhp", "transmission", "mileage_kmpl", "seats", "owner_count",
    }
    for key, value in normalized["field_sources"].items():
        if key not in allowed_provenance_fields:
            errors.append(f"field_sources contains unknown field {key}")
        if get_source(value) is None:
            errors.append(f"field_sources references unknown source {value}")
        if verification_status == "verified" and not is_approved_source(value):
            errors.append(f"field_sources cannot mark {key} as verified using unapproved source {value}")

    return normalized, errors
def _row_to_specification(normalized: dict[str, Any]) -> VehicleSpecification:
    return VehicleSpecification(
        make=normalized["make"],
        model=normalized["model"],
        variant=normalized["variant"],
        fuel_type=normalized["fuel_type"],
        year_start=normalized["year_start"],
        year_end=normalized["year_end"],
        engine_cc=normalized["engine_cc"],
        max_power_bhp=normalized["max_power_bhp"],
        transmission=normalized["transmission"],
        mileage_kmpl=normalized["mileage_kmpl"],
        seats=normalized["seats"],
        owner_count=normalized["owner_count"],
        source_provider=normalized["source_name"],
        source_reference=normalized["source_reference"],
        verification_status=normalized["verification_status"],
        confidence=normalized["confidence"],
        field_sources=normalized["field_sources"],
    )


def _identity_matches(existing: VehicleSpecification, normalized: dict[str, Any]) -> bool:
    return (
        existing.make == normalized["make"]
        and existing.model == normalized["model"]
        and existing.variant == normalized["variant"]
        and existing.fuel_type == normalized["fuel_type"]
        and existing.year_start == normalized["year_start"]
        and existing.year_end == normalized["year_end"]
    )


def _has_conflict(existing: VehicleSpecification, normalized: dict[str, Any]) -> bool:
    if existing.source_provider != normalized["source_name"]:
        return True
    fields = ("engine_cc", "max_power_bhp", "transmission", "mileage_kmpl", "seats", "owner_count", "source_reference", "verification_status", "confidence")
    for field in fields:
        existing_value = getattr(existing, field)
        new_value = normalized.get("source_name") if field == "source_provider" else normalized.get(field)
        if field == "source_provider":
            new_value = normalized["source_name"]
        if existing_value != new_value:
            return True
    if existing.field_sources != normalized["field_sources"]:
        return True
    return False


def import_vehicle_specifications(
    db: Session,
    csv_text: str,
    *,
    dry_run: bool = False,
) -> ImportSummary:
    summary = ImportSummary(dry_run=dry_run)
    repository = VehicleSpecificationRepository(db)
    reader = csv.DictReader(io.StringIO(csv_text))

    try:
        db.begin()
        for row_number, row in enumerate(reader, start=2):
            normalized, errors = _validate_row(row, row_number)
            if errors:
                summary.rejected += 1
                summary.errors.append({"row": row_number, "errors": errors})
                continue

            existing = repository.find_best(
                make=normalized["make"],
                model=normalized["model"],
                variant=normalized["variant"],
                fuel_type=normalized["fuel_type"],
                manufacturing_year=normalized["year_start"],
            )
            exact_existing = existing if existing is not None and _identity_matches(existing, normalized) else None

            if exact_existing is not None:
                if exact_existing.verification_status == "verified" and normalized["verification_status"] != "verified":
                    summary.rejected += 1
                    summary.errors.append({"row": row_number, "errors": ["cannot overwrite a verified record with unverified data"]})
                    continue
                if _has_conflict(exact_existing, normalized):
                    summary.conflicts += 1
                    summary.errors.append({"row": row_number, "errors": ["conflicting specification for an existing identity"]})
                    continue
                summary.skipped += 1
                continue

            if normalized["verification_status"] == "verified" and not is_approved_source(normalized["source_name"]):
                summary.rejected += 1
                summary.errors.append({"row": row_number, "errors": ["verified status requires an approved source"]})
                continue

            specification = _row_to_specification(normalized)
            if not dry_run:
                repository.add(specification)
            summary.created += 1

        if dry_run:
            db.rollback()
        else:
            db.commit()
    except Exception:
        db.rollback()
        raise
    return summary