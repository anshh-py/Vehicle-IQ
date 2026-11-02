from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ManualRegistrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registration_number: str = Field(min_length=1, max_length=32)


class ManualRegistrationResponse(BaseModel):
    assessment_id: UUID
    registration_number: str
    source: Literal["manual"] = "manual"


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
    # Pricing-relevant fields from VehicleX
    engine_cc: int | None = Field(default=None, ge=1, le=10000, description="Engine displacement in cubic centimetres.")
    seats: int | None = Field(default=None, ge=1, le=9, description="Number of seats.")
    max_power_bhp: float | None = Field(default=None, ge=0, le=2000, description="Maximum engine power in brake horsepower.")
    transmission: str | None = Field(default=None, max_length=50, description="Transmission type (e.g., Manual, Automatic).")
    owner_count: int | None = Field(default=None, ge=1, le=20, description="Number of previous owners.")
    mileage_kmpl: float | None = Field(default=None, gt=0, le=100, description="Fuel efficiency in kilometres per litre.")


class VehicleRegistrationLookupResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    image_id: UUID | None
    provider: str
    status: str
    details: VehicleRegistrationDetails
    created_at: datetime

    @computed_field
    @property
    def field_provenance(self) -> dict[str, str]:
        return {
            key: "vehicle_api"
            for key, value in self.details.model_dump(exclude_none=True).items()
            if value is not None
        }
