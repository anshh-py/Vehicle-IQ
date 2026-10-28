from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class UploadedImageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    assessment_id: UUID
    original_filename: str | None
    content_type: str | None
    byte_size: int | None
    sha256: str | None
    captured_at: datetime | None
    uploaded_at: datetime
