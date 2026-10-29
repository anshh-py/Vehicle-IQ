from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class VehicleRegistrationDetails(BaseModel):
    """Normalized facts returned by an authorized registration provider."""

    model_config = ConfigDict(extra="forbid")

    registration_number: str = Field(min_length=1, max_length=32)
    make: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=120)
    variant: str | None = Field(default=None, max_length=120)
    fuel_type: str | None = Field(default=None, max_length=50)
    registration_date: date | None = None
    manufacturing_year: int | None = Field(default=None, ge=1900, le=2100)
    vehicle_class: str | None = Field(default=None, max_length=100)
    registration_valid_until: date | None = None
    rto: str | None = Field(default=None, max_length=120)
    fitness_expiry: date | None = None
    puc_expiry: date | None = None
    insurance_expiry: date | None = None
    blacklisted: bool | None = None


class VehicleRegistrationLookupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    image_id: UUID | None
    provider: str
    status: str
    details: VehicleRegistrationDetails
    created_at: datetime
