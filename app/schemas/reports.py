from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class AssessmentReportResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    status: str
    summary: str | None
    report_data: dict[str, Any] | None
    generated_at: datetime | None
    created_at: datetime
