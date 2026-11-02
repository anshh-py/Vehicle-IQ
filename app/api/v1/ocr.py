import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import EvidenceRecord, OCRResult, UploadedImage, VehicleAssessment
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
from app.ocr.preprocessing import preprocess_ocr_image
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
            read_timeout_seconds=settings.ocr_timeout_seconds,
        )
    if not settings.ocr_service_url:
        return None
    return BaiduUnlimitedOCRClient(settings.ocr_service_url, timeout_seconds=settings.ocr_timeout_seconds)


def _finish_ocr_attempt(
    db: Session,
    attempt: EvidenceRecord,
    *,
    state: str,
    error_code: str | None = None,
    message: str | None = None,
) -> None:
    data = dict(attempt.evidence_data or {})
    data.update({"status": state, "error_code": error_code, "message": message})
    attempt.evidence_data = data
    attempt.observed_at = datetime.now(timezone.utc)
    try:
        db.add(attempt)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save OCR attempt status.") from None


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
    attempt = EvidenceRecord(
        assessment_id=assessment_id,
        image_id=image_id,
        evidence_type="ocr_attempt",
        source_reference=getattr(client, "provider_name", None),
        evidence_data={
            "status": "processing",
            "provider": getattr(client, "provider_name", None),
            "error_code": None,
            "message": None,
        },
        observed_at=datetime.now(timezone.utc),
    )
    try:
        db.add(attempt)
        db.commit()
        db.refresh(attempt)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to start OCR attempt.") from None

    if client is None:
        _finish_ocr_attempt(
            db, attempt, state="unavailable", error_code="provider_not_configured",
            message="No OCR provider is configured.",
        )
        raise HTTPException(status_code=503, detail="OCR service is not configured.")

    try:
        content = storage.read(image.object_key)
    except ImageStorageError:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="stored_image_unavailable",
            message="The stored image could not be read.",
        )
        raise HTTPException(status_code=500, detail="Stored image is unavailable.") from None

    try:
        inspect_image(content)
    except InvalidImageError:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="invalid_image",
            message="The stored image is corrupt or unsupported.",
        )
        raise HTTPException(status_code=422, detail="Stored image is corrupt or unsupported.") from None

    try:
        ocr_content = preprocess_ocr_image(content)
        filename = "image.png"
        if isinstance(client, HuggingFaceGradioOCRClient):
            original_name = (image.original_filename or "image").rsplit(".", 1)[0]
            filename = f"{original_name or 'image'}.png"
        result = client.recognize(ocr_content, filename, "image/png")
    except ValueError:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="image_preprocessing_failed",
            message="The image could not be prepared for OCR.",
        )
        raise HTTPException(status_code=422, detail="Stored image could not be prepared for OCR.") from None
    except OCRServiceUnavailable as exc:
        safe_exception_text = str(exc).casefold()
        timed_out = "timeout" in safe_exception_text or "timed out" in safe_exception_text
        code = "provider_timeout" if timed_out else "provider_unavailable"
        safe_message = "The OCR provider timed out." if timed_out else "The OCR provider is unavailable."
        _finish_ocr_attempt(db, attempt, state="unavailable", error_code=code, message=safe_message)
        if isinstance(client, PaddleOCRLocalClient):
            logger.warning("Local PaddleOCR unavailable (%s).", type(exc).__name__)
            raise HTTPException(status_code=503, detail=str(exc)) from None
        raise HTTPException(status_code=503, detail="OCR service is unavailable.") from None
    except OCRInvalidResponse:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="invalid_provider_response",
            message="The OCR provider returned an invalid response.",
        )
        raise HTTPException(status_code=502, detail="OCR service returned an invalid response.") from None
    except OCRUnreadableText:
        _finish_ocr_attempt(
            db, attempt, state="no_text", error_code="no_text_detected",
            message="The OCR provider completed but found no readable text.",
        )
        raise HTTPException(status_code=422, detail="No readable text was found in the image.") from None
    except OCRInferenceFailed as exc:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="inference_failed",
            message="The OCR provider reported an inference failure.",
        )
        logger.warning("OCR inference failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=502, detail="OCR inference failed.") from None
    except Exception as exc:
        _finish_ocr_attempt(
            db, attempt, state="failed", error_code="unexpected_provider_error",
            message="The OCR attempt failed unexpectedly.",
        )
        logger.warning("OCR attempt failed (%s).", type(exc).__name__)
        raise HTTPException(status_code=502, detail="OCR inference failed.") from None

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
        attempt.evidence_data = {
            "status": "completed",
            "provider": client.provider_name,
            "error_code": None,
            "message": None,
        }
        attempt.observed_at = datetime.now(timezone.utc)
        db.add(attempt)
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
        try:
            _finish_ocr_attempt(
                db, attempt, state="failed", error_code="result_persistence_failed",
                message="The OCR result could not be saved.",
            )
        except HTTPException:
            pass
        raise HTTPException(status_code=500, detail="Unable to save OCR result.") from None
    return response
