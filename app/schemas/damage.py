from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


DamageInspectionStatus = Literal[
    "completed_with_findings",
    "completed_no_findings",
    "unavailable",
    "failed",
]
DamageImageType = Literal["vehicle_exterior", "vehicle_closeup", "registration_plate", "document", "vehicle_interior"]


class DamageFindingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    image_id: UUID | None
    component: str | None
    damage_type: str
    confidence: float | None
    bounding_box: dict[str, float | str] | None
    model_name: str | None


class ImageDamageResponse(BaseModel):
    assessment_id: UUID
    image_id: UUID
    provider: str
    status: DamageInspectionStatus
    reused: bool
    findings: list[DamageFindingResponse]
    image_type: str | None = None
    reason: str | None = None
    reason_code: str | None = None


class AssessmentDamageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_ids: list[UUID] = Field(min_length=1, max_length=10)
    image_types: dict[UUID, DamageImageType] = Field(default_factory=dict)

    @model_validator(mode="after")
    def image_ids_must_be_unique(self) -> "AssessmentDamageRequest":
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must not contain duplicates")
        if not set(self.image_types).issubset(set(self.image_ids)):
            raise ValueError("image_types keys must be selected image_ids")
        return self


class AssessmentImageDamageResult(BaseModel):
    image_id: UUID
    status: DamageInspectionStatus
    reused: bool = False
    findings: list[DamageFindingResponse] = Field(default_factory=list)
    error: str | None = None
    image_type: str | None = None
    reason: str | None = None
    reason_code: str | None = None


class AssessmentDamageResponse(BaseModel):
    assessment_id: UUID
    status: DamageInspectionStatus
    processed_count: int
    failed_count: int
    unavailable_count: int = 0
    results: list[AssessmentImageDamageResult]
