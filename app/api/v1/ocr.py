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
    HybridOCRClient,
    HuggingFaceGradioOCRClient,
    OCRClient,
    OCRInferenceFailed,
    OCRInvalidResponse,
    OCRServiceUnavailable,
    OCRUnreadableText,
    PaddleOCRLocalClient,
)
from app.ocr.preprocessing import preprocess_ocr_image
from app.ocr.sanitization import sanitize_ocr_text
from app.schemas.ocr import OCRRecognitionResponse
from app.storage.images import ImageContentUnavailable, read_image_content


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment OCR"])
logger = logging.getLogger(__name__)


def _persist_ocr_attempt_status(
    db: Session,
    assessment_id: UUID,
    image_id: UUID,
    provider: str | None,
    status: str,
    error_code: str,
    diagnostic: dict | None = None,
) -> None:
    """Store per-image OCR attempt state using the existing OCRResult table."""
    try:
        db.add(
            OCRResult(
                assessment_id=assessment_id,
                image_id=image_id,
                result_type="image_ocr",
                status=status,
                provider=provider,
                result_data={"error_code": error_code, **({"diagnostic": diagnostic} if diagnostic else {})},
            )
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning(
            "OCR attempt status could not be persisted (exception_class=%s).",
            type(exc).__name__,
        )


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
    if settings.ocr_service_provider == "hybrid":
        return HybridOCRClient(
            PaddleOCRLocalClient(
                settings.ocr_paddle_detection_model_dir,
                settings.ocr_paddle_recognition_model_dir,
            ),
            HuggingFaceGradioOCRClient(
                settings.ocr_gradio_space_url,
                mode=settings.ocr_gradio_mode,
                read_timeout_seconds=settings.ocr_timeout_seconds,
            ),
        )
    if not settings.ocr_service_url:
        return None
    return BaiduUnlimitedOCRClient(settings.ocr_service_url)


def _diagnostic(provider: str | None, category: str, message: str) -> dict:
    return {"provider": provider or "unconfigured", "operation": "recognize", "category": category, "message": message}


@router.post(
    "/{assessment_id}/images/{image_id}/ocr",
    response_model=OCRRecognitionResponse,
    response_model_exclude_none=True,
)
def recognize_assessment_image(
    assessment_id: UUID,
    image_id: UUID,
    db: Session = Depends(get_db),
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
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, None, "unavailable", "provider_unconfigured",
            _diagnostic(None, "provider_unconfigured", "No OCR provider is configured."),
        )
        raise HTTPException(status_code=503, detail="OCR service is not configured.")

    def attempt_provider_name() -> str | None:
        return getattr(client, "last_attempted_provider", None) or client.provider_name

    try:
        content = read_image_content(image)
    except ImageContentUnavailable:
        raise HTTPException(status_code=500, detail="Stored image is unavailable.") from None
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment image.") from None

    try:
        inspect_image(content)
    except InvalidImageError:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "failed", "invalid_stored_image",
            _diagnostic(attempt_provider_name(), "invalid_image", "Stored image could not be decoded."),
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
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "failed", "image_preprocessing_failed",
            _diagnostic(attempt_provider_name(), "preprocessing_failed", "Image preparation for OCR failed."),
        )
        raise HTTPException(status_code=422, detail="Stored image could not be prepared for OCR.") from None
    except OCRServiceUnavailable as exc:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "unavailable", "provider_unavailable",
            _diagnostic(attempt_provider_name(), "provider_unavailable", "Provider unavailable; check service availability, timeout, or quota."),
        )
        if isinstance(client, PaddleOCRLocalClient):
            logger.warning("Local PaddleOCR unavailable (exception_class=%s).", type(exc).__name__)
            raise HTTPException(status_code=503, detail="Local OCR provider is unavailable.") from None
        raise HTTPException(status_code=503, detail="OCR service is unavailable.") from None
    except OCRInvalidResponse:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "failed", "invalid_provider_response",
            _diagnostic(attempt_provider_name(), "invalid_response", "Provider returned an invalid response."),
        )
        raise HTTPException(status_code=502, detail="OCR service returned an invalid response.") from None
    except OCRUnreadableText:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "unreadable", "no_readable_text",
            _diagnostic(attempt_provider_name(), "empty_text", "Provider completed but returned no readable text."),
        )
        raise HTTPException(status_code=422, detail="No readable text was found in the image.") from None
    except OCRInferenceFailed as exc:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, attempt_provider_name(), "failed", "inference_failed",
            _diagnostic(attempt_provider_name(), "inference_failed", "Provider reported that OCR inference did not complete."),
        )
        detail = "OCR inference failed. The provider did not complete recognition."
        logger.warning("OCR inference failed (provider=%s, category=inference_failed).", attempt_provider_name())
        raise HTTPException(status_code=502, detail=detail) from None

    result_provider = result.provider or client.provider_name
    display_text = sanitize_ocr_text(result.text)
    if not display_text:
        _persist_ocr_attempt_status(
            db, assessment_id, image_id, result_provider, "unreadable", "no_readable_text",
            _diagnostic(result_provider, "empty_text", "Provider completed but returned no readable text."),
        )
        raise HTTPException(status_code=422, detail="No readable text was found in the image.")
    ocr_result = OCRResult(
        assessment_id=assessment_id,
        image_id=image_id,
        result_type="image_ocr",
        status="completed",
        provider=result_provider,
        raw_text=result.text,
        result_data={
            **({"confidence": result.confidence} if result.confidence is not None else {}),
            "display_text": display_text,
            "raw_provider_evidence_available": True,
        },
    )
    try:
        db.add(ocr_result)
        db.flush()
        db.refresh(ocr_result)
        response = OCRRecognitionResponse(
            id=ocr_result.id,
            assessment_id=ocr_result.assessment_id,
            image_id=image_id,
            provider=result_provider,
            result_type=ocr_result.result_type,
            status=ocr_result.status,
            text=display_text,
            confidence=result.confidence,
        )
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save OCR result.") from None
    return response
