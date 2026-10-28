from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


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
    status: Literal["completed"]
    reused: bool
    findings: list[DamageFindingResponse]
