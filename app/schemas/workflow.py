from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datetime import date

from app.schemas.pricing import FuelType, NonEmptyMake, NonEmptyModel, NonEmptyVariant, SellerType, TransmissionType


class WorkflowVehicleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    make: NonEmptyMake | None = None
    model: NonEmptyModel | None = None
    variant: NonEmptyVariant | None = None
    registration_year: int | None = Field(default=None, ge=1886, le=date.today().year)
    odometer_km: int | None = Field(default=None, ge=0, le=2_000_000)
    fuel_type: FuelType | None = None
    seller_type: SellerType | None = None
    transmission: TransmissionType | None = None
    mileage_kmpl: float | None = Field(default=None, gt=0, le=100)
    max_power_bhp: float | None = Field(default=None, gt=0, le=2_000)
    engine_cc: int | None = Field(default=None, ge=1, le=10_000)
    seats: int | None = Field(default=None, ge=1, le=9)


class AssessmentWorkflowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ids: list[UUID] = Field(min_length=1, max_length=10)
    vehicle: WorkflowVehicleInput | None = None
    image_types: dict[UUID, Literal["vehicle_exterior", "vehicle_closeup", "registration_plate", "document", "vehicle_interior"]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def image_ids_must_be_unique(self) -> "AssessmentWorkflowRequest":
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must not contain duplicates")
        if not set(self.image_types).issubset(set(self.image_ids)):
            raise ValueError("image_types keys must be selected image_ids")
        return self


class WorkflowStageError(BaseModel):
    image_id: UUID | None = None
    code: str
    detail: str


class WorkflowStageResult(BaseModel):
    status: Literal[
        "available",
        "partial",
        "unavailable",
        "failed",
        "skipped",
        "completed_with_findings",
        "completed_no_findings",
        "no_text",
        "processing",
        "not_started",
        "disabled",
        "not_run",
        "completed",
    ]
    result: Any | None = None
    errors: list[WorkflowStageError] = Field(default_factory=list)


class AssessmentWorkflowResponse(BaseModel):
    assessment_id: UUID
    status: Literal["completed", "partial", "failed"]
    stages: dict[str, WorkflowStageResult]
