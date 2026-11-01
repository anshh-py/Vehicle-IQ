import csv
import io
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.db.models import MarketListing as MarketListingModel
from app.market.repository import MarketListingRepository
from app.pricing.market import MarketListing, MarketVehicleCriteria

REQUIRED_FIELDS = ("source_name", "source_url", "make", "model", "manufacturing_year", "asking_price_inr")


@dataclass
class MarketImportSummary:
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


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "source_name": _clean(row.get("source_name")),
        "source_listing_id": _clean(row.get("source_listing_id")),
        "source_url": _clean(row.get("source_url")),
        "make": _clean(row.get("make")),
        "model": _clean(row.get("model")),
        "variant": _clean(row.get("variant")),
        "manufacturing_year": int(row["manufacturing_year"]) if _clean(row.get("manufacturing_year")) is not None else None,
        "fuel_type": _clean(row.get("fuel_type")),
        "transmission": _clean(row.get("transmission")),
        "odometer_km": int(row["odometer_km"]) if _clean(row.get("odometer_km")) is not None else None,
        "asking_price_inr": row.get("asking_price_inr"),
        "city": _clean(row.get("city")),
        "state": _clean(row.get("state")),
        "seller_type": _clean(row.get("seller_type")),
        "listed_at": _parse_datetime(row.get("listed_at")),
        "collected_at": _parse_datetime(row.get("collected_at")) or datetime.now(timezone.utc),
        "raw_data": row.get("raw_data") if isinstance(row.get("raw_data"), dict) else None,
    }
    return normalized
def _validate_row(row: dict[str, Any], row_number: int) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        normalized = _normalize_row(row)
    except (TypeError, ValueError) as exc:
        return {}, [str(exc)]
    for field in REQUIRED_FIELDS:
        if normalized.get(field) is None:
            errors.append(f"{field} is required")
    if normalized.get("manufacturing_year") is not None and not (1900 <= normalized["manufacturing_year"] <= 2100):
        errors.append("manufacturing_year must be between 1900 and 2100")
    if normalized.get("odometer_km") is not None and not (0 <= normalized["odometer_km"] <= 2_000_000):
        errors.append("odometer_km must be between 0 and 2000000")
    if normalized.get("asking_price_inr") is not None:
        try:
            from decimal import Decimal

            normalized["asking_price_inr"] = str(Decimal(str(normalized["asking_price_inr"])))
        except Exception:
            errors.append("asking_price_inr must be a valid decimal")
    if normalized.get("source_url") is not None and not str(normalized["source_url"]).startswith("https://"):
        errors.append("source_url must use HTTPS")
    if normalized.get("fuel_type") is not None:
        normalized["fuel_type"] = normalized["fuel_type"].casefold().capitalize() if False else normalized["fuel_type"].strip()
    # Validate against existing market contract.
    try:
        from app.pricing.market import MarketListing

        MarketListing.model_validate(
            {
                "source_name": normalized["source_name"],
                "listing_id": normalized["source_listing_id"],
                "make": normalized["make"],
                "model": normalized["model"],
                "variant": normalized["variant"],
                "year": normalized["manufacturing_year"],
                "fuel_type": normalized["fuel_type"],
                "transmission": normalized["transmission"],
                "mileage_km": normalized["odometer_km"],
                "location": normalized["city"] or normalized["state"],
                "advertised_price_inr": normalized["asking_price_inr"],
                "source_url": normalized["source_url"],
                "listed_at": normalized["listed_at"] or datetime.now(timezone.utc),
                "retrieved_at": normalized["collected_at"],
            }
        )
    except Exception as exc:
        errors.append(f"invalid market listing: {exc}")
    return normalized, errors


def import_market_listings(db, payload, *, input_format="csv", dry_run=False):
    from app.market.repository import MarketListingRepository

    summary = MarketImportSummary(dry_run=dry_run)
    repository = MarketListingRepository(db)
    rows = []
    if input_format == "csv":
        reader = csv.DictReader(io.StringIO(str(payload)))
        rows = list(reader)
    elif input_format == "json":
        if isinstance(payload, str):
            payload = json.loads(payload)
        rows = list(payload)
    else:
        raise ValueError("input_format must be csv or json")

    seen_batch = set()
    try:
        db.begin()
        for row_number, row in enumerate(rows, start=2 if input_format == "csv" else 1):
            normalized, errors = _validate_row(row, row_number)
            if errors:
                summary.rejected += 1
                summary.errors.append({"row": row_number, "errors": errors})
                continue
            identity = (normalized["source_name"], normalized["source_listing_id"], normalized["source_url"])
            if identity in seen_batch:
                summary.rejected += 1
                summary.errors.append({"row": row_number, "errors": ["duplicate listing within import batch"]})
                continue
            seen_batch.add(identity)
            existing = repository.find_duplicate(
                source_name=normalized["source_name"],
                source_listing_id=normalized["source_listing_id"],
                source_url=normalized["source_url"],
            )
            if existing is not None:
                summary.skipped += 1
                continue
            from app.db.models import MarketListing as MarketListingModel

            listing = MarketListingModel(
                source_name=normalized["source_name"],
                source_listing_id=normalized["source_listing_id"],
                source_url=normalized["source_url"],
                make=normalized["make"],
                model=normalized["model"],
                variant=normalized["variant"],
                manufacturing_year=normalized["manufacturing_year"],
                fuel_type=normalized["fuel_type"],
                transmission=normalized["transmission"],
                odometer_km=normalized["odometer_km"],
                asking_price_inr=normalized["asking_price_inr"],
                city=normalized["city"],
                state=normalized["state"],
                seller_type=normalized["seller_type"],
                listed_at=normalized["listed_at"],
                collected_at=normalized["collected_at"],
                raw_data=normalized["raw_data"],
            )
            if not dry_run:
                repository.add(listing)
            summary.created += 1
        if dry_run:
            db.rollback()
        else:
            db.commit()
    except Exception:
        db.rollback()
        raise
    return summary


def import_market_listings_from_csv(db, csv_text, *, dry_run=False):
    return import_market_listings(db, csv_text, input_format="csv", dry_run=dry_run)


def import_market_listings_from_json(db, payload, *, dry_run=False):
    return import_market_listings(db, payload, input_format="json", dry_run=dry_run)