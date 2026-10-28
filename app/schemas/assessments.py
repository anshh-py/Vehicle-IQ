from enum import Enum
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


NonEmptyVIN = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=17)]
NonEmptyRegistration = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=32)]
AskingPriceINR = Annotated[
    Decimal,
    Field(
        gt=0,
        le=Decimal("100000000"),
        max_digits=12,
        decimal_places=2,
        description="Optional seller asking price in INR, from above zero through ₹10 crore.",
    ),
]


class AssessmentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AssessmentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vin: NonEmptyVIN | None = None
    registration_number: NonEmptyRegistration | None = None
    vehicle_details: dict[str, Any] | None = None
    asking_price_inr: AskingPriceINR | None = None


class AssessmentPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vin: NonEmptyVIN | None = None
    registration_number: NonEmptyRegistration | None = None
    vehicle_details: dict[str, Any] | None = None
    asking_price_inr: AskingPriceINR | None = None
    status: AssessmentStatus | None = None

    @field_validator("status")
    @classmethod
    def status_cannot_be_null(cls, value: AssessmentStatus | None) -> AssessmentStatus:
        if value is None:
            raise ValueError("status cannot be null")
        return value

    @model_validator(mode="after")
    def require_at_least_one_change(self) -> "AssessmentPatch":
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        return self


class AssessmentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    vin: str | None
    registration_number: str | None
    vehicle_details: dict[str, Any] | None
    asking_price_inr: Decimal | None
    created_at: datetime
    updated_at: datetime


class AssessmentListResponse(BaseModel):
    items: list[AssessmentResponse]
    total: int
    limit: int
    offset: int


AssessmentLimit = Annotated[int, Field(ge=1, le=100)]
AssessmentOffset = Annotated[int, Field(ge=0)]
