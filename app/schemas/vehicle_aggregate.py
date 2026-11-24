from datetime import date as Date
from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ServiceStatus = Literal[
    "available",
    "unavailable",
    "partial",
    "failed",
    "not_found",
    "malformed",
]

AggregateStatus = Literal["available", "unavailable", "partial", "failed"]


class VehicleAggregateSource(BaseModel):
    """Retrieval state of one data source within the aggregated record."""

    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus = "unavailable"
    provider: str | None = None
    error_code: str | None = None


class VehicleAggregateInsurance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus = "unavailable"
    expiry: Date | None = None
    provider: str | None = None


class VehicleAggregateChallan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus = "unavailable"
    provider: str | None = None
    error_code: str | None = None
    records: int = 0
    total_amount_inr: Decimal | None = Field(default=None, ge=0)
    unpaid_count: int = 0


class VehicleAggregateFastag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus = "unavailable"
    provider: str | None = None
    error_code: str | None = None
    issuing_bank: str | None = None
    issue_date: Date | None = None
    vehicle_class: str | None = None


class NormalizedVehicleRecord(BaseModel):
    """One reliable vehicle record aggregated from all configured sources.

    Each field keeps its source and retrieval status. A source that failed or
    is unavailable never overwrites valid data from another source. The record
    only reports facts that were actually supplied by a provider; it never
    fabricates data.
    """

    model_config = ConfigDict(extra="forbid")

    assessment_id: UUID | None = None
    registration_number: str | None = None
    status: AggregateStatus = "unavailable"

    # Registration / vehicle specifications (provider: vehiclex_candidate).
    vehicle_specs: dict[str, object] = Field(default_factory=dict)

    # Insurance (currently derived from the registration provider's fields).
    insurance: VehicleAggregateInsurance = Field(default_factory=VehicleAggregateInsurance)

    # Challan and FASTag lookups.
    challan: VehicleAggregateChallan = Field(default_factory=VehicleAggregateChallan)
    fastag: VehicleAggregateFastag = Field(default_factory=VehicleAggregateFastag)

    # Per-field source provenance: field -> "vehicle_api" | "user_input".
    field_provenance: dict[str, str] = Field(default_factory=dict)

    # Source-level retrieval states.
    sources: dict[str, VehicleAggregateSource] = Field(default_factory=dict)

    # Human-readable collection of per-source errors (codes only, no secrets).
    errors: list[dict[str, str]] = Field(default_factory=list)
