from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyMake = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
NonEmptyModel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
NonEmptyVariant = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]
NonEmptyLabel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class FuelType(str, Enum):
    PETROL = "petrol"
    DIESEL = "diesel"
    CNG = "cng"
    LPG = "lpg"
    ELECTRIC = "electric"


class TransmissionType(str, Enum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    AMT = "amt"
    CVT = "cvt"
    DCT = "dct"


class SellerType(str, Enum):
    DEALER = "Dealer"
    INDIVIDUAL = "Individual"
    TRUSTMARK_DEALER = "Trustmark Dealer"


class VehiclePriceInput(BaseModel):
    """User-supplied vehicle features; no missing value is inferred by the API."""

    model_config = ConfigDict(extra="forbid")

    make: NonEmptyMake
    model: NonEmptyModel
    variant: NonEmptyVariant | None = None
    registration_year: int = Field(
        ge=1886,
        le=date.today().year,
        description="Vehicle's first registration year, not inferred from other fields.",
    )
    odometer_km: int = Field(ge=0, le=2_000_000, description="Odometer reading in kilometres.")
    fuel_type: FuelType
    transmission: TransmissionType
    seller_type: SellerType
    mileage_kmpl: float = Field(gt=0, le=100, description="Fuel efficiency in kilometres per litre.")
    engine_cc: int = Field(gt=0, le=10_000, description="Engine displacement in cubic centimetres.")
    max_power_bhp: float = Field(gt=0, le=2_000, description="Maximum engine power in brake horsepower.")
    seats: int = Field(ge=1, le=9)
    owner_count: int = Field(
        ge=1,
        le=20,
        description="Ownership count including the current registered owner (first owner is 1).",
    )


class PriceEstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle: VehiclePriceInput


class PricePrediction(BaseModel):
    """Validated output required from a real future pricing provider."""

    model_config = ConfigDict(extra="forbid")

    amount: Decimal = Field(gt=0, max_digits=12, decimal_places=2)
    currency: Literal["INR"]
    lower_bound: Decimal | None = Field(default=None, ge=0, max_digits=12, decimal_places=2)
    upper_bound: Decimal | None = Field(default=None, gt=0, max_digits=12, decimal_places=2)
    method: NonEmptyLabel
    model_name: NonEmptyLabel

    @model_validator(mode="after")
    def bounds_include_estimate(self) -> "PricePrediction":
        if self.lower_bound is not None and self.lower_bound > self.amount:
            raise ValueError("lower_bound cannot exceed amount")
        if self.upper_bound is not None and self.upper_bound < self.amount:
            raise ValueError("upper_bound cannot be less than amount")
        return self


class PriceEstimateResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    amount: Decimal
    currency: str
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    method: str | None
    model_name: str | None
    created_at: datetime
