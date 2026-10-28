from __future__ import annotations

import hashlib
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.images.validation import InvalidImageError, inspect_image
from app.schemas.images import UploadedImageResponse
from app.storage.images import ImageStorage, get_image_storage


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/assessments/{assessment_id}/images", tags=["assessment images"])

HTTP_413_CONTENT_TOO_LARGE = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
HTTP_422_UNPROCESSABLE_CONTENT = getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422)


def get_image_upload_limit() -> int:
    return settings.max_image_upload_bytes


def _safe_original_filename(filename: str | None, extension: str) -> str:
    basename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    safe_name = "".join(char for char in basename if char.isprintable() and char not in "\r\n\t").strip()
    return (safe_name or f"upload.{extension}")[:255]


def _cleanup_file(storage: ImageStorage, key: str) -> None:
    try:
        storage.delete(key)
    except Exception:
        logger.exception("Failed to clean up image after database failure.")


@router.post("", response_model=UploadedImageResponse, status_code=status.HTTP_201_CREATED)
def upload_assessment_image(
    assessment_id: UUID,
    file: Annotated[UploadFile, File(...)],
    db: Session = Depends(get_db),
    storage: ImageStorage = Depends(get_image_storage),
    max_bytes: int = Depends(get_image_upload_limit),
) -> UploadedImageResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment.") from None
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    try:
        content = file.file.read(max_bytes + 1)
    except OSError:
        raise HTTPException(status_code=400, detail="Unable to read uploaded image.") from None
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image exceeds the {max_bytes}-byte upload limit.",
        )
    if not content:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Upload must contain a valid JPEG, PNG, or WebP image.",
        )

    try:
        extension, content_type = inspect_image(content)
    except InvalidImageError:
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Upload must contain a valid JPEG, PNG, or WebP image.",
        ) from None
    filename = _safe_original_filename(file.filename, extension)
    digest = hashlib.sha256(content).hexdigest()
    key: str | None = None

    try:
        key = storage.save(content, extension)
        image = UploadedImage(
            assessment_id=assessment.id,
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key=key,
            original_filename=filename,
            content_type=content_type,
            byte_size=len(content),
            sha256=digest,
            captured_at=None,
        )
        db.add(image)
        db.flush()
        db.refresh(image)
        response = UploadedImageResponse.model_validate(image)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        if key is not None:
            _cleanup_file(storage, key)
        raise HTTPException(status_code=500, detail="Unable to save uploaded image.") from None
    except Exception:
        db.rollback()
        if key is not None:
            _cleanup_file(storage, key)
        raise HTTPException(status_code=500, detail="Unable to save uploaded image.") from None
    return response


@router.get("", response_model=list[UploadedImageResponse])
def list_assessment_images(assessment_id: UUID, db: Session = Depends(get_db)) -> list[UploadedImageResponse]:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        images = db.scalars(
            select(UploadedImage)
            .where(UploadedImage.assessment_id == assessment_id)
            .order_by(UploadedImage.uploaded_at.desc(), UploadedImage.id.desc())
        ).all()
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to list assessment images.") from None
    return [UploadedImageResponse.model_validate(image) for image in images]
