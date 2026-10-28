from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.damage.inference import DamageInferenceResult, DamageInputError
from app.damage.worker import (
    MODEL_NAME,
    DamageProvider,
    DamageWorkerInferenceFailed,
    DamageWorkerInvalidResponse,
    DamageWorkerTimeout,
    DamageWorkerUnavailable,
    IsolatedDamageProvider,
)
from app.db.models import DamageFinding, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.images.validation import InvalidImageError, inspect_image
from app.schemas.damage import DamageFindingResponse, ImageDamageResponse
from app.storage.images import ImageStorage, ImageStorageError, get_image_storage


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment damage detection"])
logger = logging.getLogger(__name__)


def get_damage_provider() -> DamageProvider:
    return IsolatedDamageProvider(
        settings.damage_inference_python,
        settings.damage_checkpoint_path,
        timeout_seconds=settings.damage_inference_timeout_seconds,
    )


def _finding_response(finding: DamageFinding) -> DamageFindingResponse:
    return DamageFindingResponse(
        id=finding.id,
        assessment_id=finding.assessment_id,
        image_id=finding.image_id,
        component=finding.component,
        damage_type=finding.damage_type,
        confidence=finding.confidence,
        bounding_box=finding.bounding_box,
        model_name=finding.model_name,
    )


def _analysis_response(
    assessment_id: UUID,
    image_id: UUID,
    findings: list[DamageFinding],
    *,
    reused: bool,
) -> ImageDamageResponse:
    ordered_findings = sorted(
        findings,
        key=lambda finding: (
            -(finding.confidence if finding.confidence is not None else -1.0),
            finding.damage_type,
            str(finding.id),
        ),
    )
    return ImageDamageResponse(
        assessment_id=assessment_id,
        image_id=image_id,
        provider=IsolatedDamageProvider.provider_name,
        status="completed",
        reused=reused,
        findings=[_finding_response(finding) for finding in ordered_findings],
    )


def _existing_findings(db: Session, assessment_id: UUID, image_id: UUID) -> list[DamageFinding]:
    return list(
        db.scalars(
            select(DamageFinding)
            .where(
                DamageFinding.assessment_id == assessment_id,
                DamageFinding.image_id == image_id,
                DamageFinding.model_name == MODEL_NAME,
            )
            .order_by(DamageFinding.created_at, DamageFinding.id)
        ).all()
    )


def _persist_findings(
    db: Session,
    assessment_id: UUID,
    image_id: UUID,
    result: DamageInferenceResult,
) -> ImageDamageResponse:
    if not result.detections:
        return _analysis_response(assessment_id, image_id, [], reused=False)

    try:
        db.rollback()
        with db.begin():
            locked_image_id = db.scalar(
                select(UploadedImage.id)
                .where(UploadedImage.id == image_id, UploadedImage.assessment_id == assessment_id)
                .with_for_update()
            )
            if locked_image_id is None:
                raise HTTPException(status_code=404, detail="Image not found for assessment.")

            existing = _existing_findings(db, assessment_id, image_id)
            if existing:
                return _analysis_response(assessment_id, image_id, existing, reused=True)

            findings: list[DamageFinding] = []
            for detection in result.detections:
                x1, y1, x2, y2 = detection.bbox_xyxy
                finding = DamageFinding(
                    assessment_id=assessment_id,
                    image_id=image_id,
                    component=None,
                    damage_type=detection.class_name,
                    confidence=detection.confidence,
                    bounding_box={
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "coordinate_space": "original_image_pixels",
                    },
                    model_name=MODEL_NAME,
                )
                db.add(finding)
                findings.append(finding)
            db.flush()
            response = _analysis_response(assessment_id, image_id, findings, reused=False)
        return response
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save damage findings.") from None


@router.post(
    "/{assessment_id}/images/{image_id}/damage",
    response_model=ImageDamageResponse,
)
def inspect_assessment_image_damage(
    assessment_id: UUID,
    image_id: UUID,
    db: Session = Depends(get_db),
    storage: ImageStorage = Depends(get_image_storage),
    provider: DamageProvider = Depends(get_damage_provider),
) -> ImageDamageResponse:
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
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found for assessment.")
        object_key = image.object_key
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment image.") from None

    try:
        content = storage.read(object_key)
    except ImageStorageError:
        raise HTTPException(status_code=500, detail="Stored image is unavailable.") from None

    try:
        extension, _content_type = inspect_image(content)
    except InvalidImageError:
        raise HTTPException(status_code=422, detail="Stored image is corrupt or unsupported.") from None

    try:
        existing = _existing_findings(db, assessment_id, image_id)
        if existing:
            db.rollback()
            return _analysis_response(assessment_id, image_id, existing, reused=True)
        db.rollback()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to retrieve existing damage findings.") from None

    try:
        result = provider.detect(content, extension)
    except DamageInputError:
        raise HTTPException(status_code=422, detail="Stored image is corrupt or unsupported.") from None
    except DamageWorkerUnavailable:
        raise HTTPException(status_code=503, detail="Damage inference is unavailable.") from None
    except DamageWorkerTimeout:
        raise HTTPException(status_code=504, detail="Damage inference timed out.") from None
    except DamageWorkerInvalidResponse:
        raise HTTPException(status_code=502, detail="Damage inference returned an invalid response.") from None
    except DamageWorkerInferenceFailed as exc:
        logger.warning("Damage inference worker failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Damage inference failed.") from None
    except Exception as exc:
        logger.warning("Damage inference provider failed: %s", type(exc).__name__)
        raise HTTPException(status_code=502, detail="Damage inference failed.") from None

    return _persist_findings(db, assessment_id, image_id, result)
