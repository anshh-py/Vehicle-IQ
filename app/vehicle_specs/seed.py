import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import VehicleSpecification


def load_seed_catalog(path: Path | None = None) -> list[dict[str, Any]]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "vehicle-spec-catalog.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_specification(db: Session, item: dict[str, Any]) -> VehicleSpecification | None:
    return db.scalar(
        select(VehicleSpecification).where(
            VehicleSpecification.make == item["make"],
            VehicleSpecification.model == item["model"],
            VehicleSpecification.variant == item.get("variant"),
            VehicleSpecification.fuel_type == item.get("fuel_type"),
            VehicleSpecification.year_start == item.get("year_start"),
            VehicleSpecification.year_end == item.get("year_end"),
        )
    )


def seed_vehicle_specifications(db: Session, path: Path | None = None) -> int:
    created = 0
    for item in load_seed_catalog(path):
        if _existing_specification(db, item) is not None:
            continue
        specification = VehicleSpecification(
            make=item["make"],
            model=item["model"],
            variant=item.get("variant"),
            fuel_type=item.get("fuel_type"),
            year_start=item.get("year_start"),
            year_end=item.get("year_end"),
            engine_cc=item.get("engine_cc"),
            max_power_bhp=item.get("max_power_bhp"),
            transmission=item.get("transmission"),
            mileage_kmpl=item.get("mileage_kmpl"),
            seats=item.get("seats"),
            owner_count=item.get("owner_count"),
            source_provider=item.get("source_provider", "seed_catalog"),
            source_reference=item.get("source_reference"),
            verification_status=item.get("verification_status", "unverified"),
            verified_at=datetime.now(timezone.utc) if item.get("verification_status") == "verified" else None,
            confidence=item.get("confidence"),
            field_sources=item.get("field_sources") or {},
        )
        db.add(specification)
        created += 1
    db.commit()
    return created