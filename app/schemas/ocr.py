from uuid import UUID

from pydantic import BaseModel


class OCRRecognitionResponse(BaseModel):
    id: UUID
    assessment_id: UUID
    image_id: UUID
    provider: str
    result_type: str
    status: str
    text: str
    confidence: float | None = None
