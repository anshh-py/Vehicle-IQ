from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ResolvedVehicleSpecification(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    make: str
    model: str
    variant: str | None = None
    fuel_type: str | None = None
    manufacturing_year: int | None = None
    engine_cc: int | None = None
    max_power_bhp: float | None = None
    transmission: str | None = None
    mileage_kmpl: float | None = None
    seats: int | None = None
    owner_count: int | None = None
    source_provider: str
    source_reference: str | None = None
    verification_status: str = "unverified"
    verified_at: datetime | None = None
    confidence: float | None = None
    field_sources: dict[str, str] = {}
    catalog_id: UUID | None = None


class VehicleSpecificationResolution(BaseModel):
    requested: dict[str, Any]
    resolved: ResolvedVehicleSpecification | None = None
    missing_fields: list[str] = []
    provenance: dict[str, str] = {}