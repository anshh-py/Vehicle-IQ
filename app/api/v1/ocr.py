from uuid import UUID
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import OCRResult, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.images.validation import InvalidImageError, inspect_image
from app.ocr.client import (
    BaiduUnlimitedOCRClient,
    HuggingFaceGradioOCRClient,
    OCRClient,
    OCRInferenceFailed,
    OCRInvalidResponse,
    OCRServiceUnavailable,
    OCRUnreadableText,
    PaddleOCRLocalClient,
)
from app.schemas.ocr import OCRRecognitionResponse
from app.storage.images import ImageStorage, ImageStorageError, get_image_storage


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment OCR"])
logger = logging.getLogger(__name__)


def get_ocr_client() -> OCRClient | None:
    if settings.ocr_service_provider == "paddleocr_local":
        return PaddleOCRLocalClient(
            settings.ocr_paddle_detection_model_dir,
            settings.ocr_paddle_recognition_model_dir,
        )
    if settings.ocr_service_provider == "huggingface_gradio":
        return HuggingFaceGradioOCRClient(
            settings.ocr_gradio_space_url,
            mode=settings.ocr_gradio_mode,
        )
    if not settings.ocr_service_url:
        return None
    return BaiduUnlimitedOCRClient(settings.ocr_service_url)


@router.post(
    "/{assessment_id}/images/{image_id}/ocr",
    response_model=OCRRecognitionResponse,
    response_model_exclude_none=True,
)
def recognize_assessment_image(
    assessment_id: UUID,
    image_id: UUID,
    db: Session = Depends(get_db),
    storage: ImageStorage = Depends(get_image_storage),
    client: OCRClient | None = Depends(get_ocr_client),
) -> OCRRecognitionResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        image = db.scalar(
            select(UploadedImage).where(
                UploadedImage.id == image_id,
                UploadedImage.assessment_id == assessment_id,
            )
        )
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment image.") from None

    if image is None:
        raise HTTPException(status_code=404, detail="Image not found for assessment.")
    if client is None:
        raise HTTPException(status_code=503, detail="OCR service is not configured.")

    try:
        content = storage.read(image.object_key)
    except ImageStorageError:
        raise HTTPException(status_code=500, detail="Stored image is unavailable.") from None

    try:
        extension, content_type = inspect_image(content)
    except InvalidImageError:
        raise HTTPException(status_code=422, detail="Stored image is corrupt or unsupported.") from None

    try:
        filename = f"image.{extension}"
        if isinstance(client, HuggingFaceGradioOCRClient):
            filename = image.original_filename or filename
        result = client.recognize(content, filename, content_type)
    except OCRServiceUnavailable as exc:
        if isinstance(client, PaddleOCRLocalClient):
            logger.warning("Local PaddleOCR unavailable: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from None
        raise HTTPException(status_code=503, detail="OCR service is unavailable.") from None
    except OCRInvalidResponse:
        raise HTTPException(status_code=502, detail="OCR service returned an invalid response.") from None
    except OCRUnreadableText:
        raise HTTPException(status_code=422, detail="No readable text was found in the image.") from None
    except OCRInferenceFailed as exc:
        detail = str(exc).strip() or "OCR inference failed. The OCR service returned no error details."
        logger.warning("OCR inference failed: %s", detail)
        raise HTTPException(status_code=502, detail=detail) from None

    ocr_result = OCRResult(
        assessment_id=assessment_id,
        image_id=image_id,
        result_type="image_ocr",
        status="completed",
        provider=client.provider_name,
        raw_text=result.text,
        result_data={"confidence": result.confidence} if result.confidence is not None else None,
    )
    try:
        db.add(ocr_result)
        db.flush()
        db.refresh(ocr_result)
        response = OCRRecognitionResponse(
            id=ocr_result.id,
            assessment_id=ocr_result.assessment_id,
            image_id=image_id,
            provider=client.provider_name,
            result_type=ocr_result.result_type,
            status=ocr_result.status,
            text=result.text,
            confidence=result.confidence,
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save OCR result.") from None
    return response
