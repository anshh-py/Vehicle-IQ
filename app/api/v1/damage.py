from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.damage.inference import DamageInferenceResult, DamageInputError
from app.damage.worker import (
    MODEL_NAME,
    DamageInferenceDisabled,
    DamageProvider,
    DamageWorkerInferenceFailed,
    DamageWorkerInvalidResponse,
    DamageWorkerTimeout,
    DamageWorkerUnavailable,
    IsolatedDamageProvider,
)
from app.db.models import DamageFinding, EvidenceRecord, OCRResult, UploadedImage, VehicleAssessment
from app.images.classification import (
    ImageEligibilityDecision,
    cached_decision,
    classify_image,
    classification_cache_key,
    store_classification,
)
from app.db.session import get_db
from app.images.validation import InvalidImageError, inspect_image
from app.schemas.damage import (
    AssessmentDamageRequest,
    AssessmentDamageResponse,
    AssessmentImageDamageResult,
    DamageImageType,
    DamageInspectionStatus,
    DamageFindingResponse,
    ImageDamageResponse,
)
from app.storage.images import ImageStorage, ImageStorageError, get_image_storage


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment damage detection"])
logger = logging.getLogger(__name__)


def get_damage_provider() -> DamageProvider:
    enabled = settings.damage_inference_enabled
    if enabled is None:
        enabled = settings.app_env.casefold() != "production"
    if not enabled:
        return _DisabledDamageProvider()
    return IsolatedDamageProvider(
        settings.damage_inference_python,
        settings.damage_checkpoint_path,
        timeout_seconds=settings.damage_inference_timeout_seconds,
    )


class _DisabledDamageProvider:
    provider_name = IsolatedDamageProvider.provider_name

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        raise DamageInferenceDisabled("Damage inference is disabled by configuration.")


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
    status: DamageInspectionStatus | None = None,
    image_type: str | None = None,
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
        status=status or (
            "completed_with_findings" if ordered_findings else "completed_no_findings"
        ),
        reused=reused,
        findings=[_finding_response(finding) for finding in ordered_findings],
        image_type=image_type,
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


def _existing_scan(db: Session, assessment_id: UUID, image_id: UUID) -> EvidenceRecord | None:
    return db.scalar(
        select(EvidenceRecord)
        .where(
            EvidenceRecord.assessment_id == assessment_id,
            EvidenceRecord.image_id == image_id,
            EvidenceRecord.evidence_type == "damage_detection",
            EvidenceRecord.source_reference == MODEL_NAME,
        )
        .order_by(EvidenceRecord.created_at.desc(), EvidenceRecord.id.desc())
    )


def _existing_eligibility_scan(db: Session, assessment_id: UUID, image_id: UUID) -> EvidenceRecord | None:
    return db.scalar(
        select(EvidenceRecord).where(
            EvidenceRecord.assessment_id == assessment_id,
            EvidenceRecord.image_id == image_id,
            EvidenceRecord.evidence_type == "damage_eligibility",
        )
    )


_COMPLETED_DAMAGE_STATUSES = {
    "completed_with_findings",
    "completed_no_findings",
}


def _stored_damage_status(
    scan: EvidenceRecord | None,
    findings: list[DamageFinding],
) -> DamageInspectionStatus:
    if findings:
        return "completed_with_findings"
    if scan is None or not isinstance(scan.evidence_data, dict):
        return "unavailable"
    data = scan.evidence_data
    status = data.get("status")
    if status == "completed_with_findings":
        return "unavailable"
    if status == "completed_no_findings" and data.get("detection_count", 0) == 0:
        return "completed_no_findings"
    if status == "completed":
        if data.get("detection_count") == 0:
            return "completed_no_findings"
        return "unavailable"
    if status == "unavailable":
        return "unavailable"
    if status == "failed":
        return "failed"
    return "unavailable"


def _record_inspection_outcome(
    db: Session,
    assessment_id: UUID,
    image_id: UUID,
    status: DamageInspectionStatus,
    *,
    reason: str | None = None,
    image_type: str | None = None,
    detection_count: int = 0,
    reason_code: str | None = None,
) -> None:
    try:
        db.rollback()
        with db.begin():
            locked_image_id = db.scalar(
                select(UploadedImage.id)
                .where(
                    UploadedImage.id == image_id,
                    UploadedImage.assessment_id == assessment_id,
                )
                .with_for_update()
            )
            if locked_image_id is None:
                raise HTTPException(status_code=404, detail="Image not found for assessment.")
            record = _existing_scan(db, assessment_id, image_id)
            evidence_data = {
                "status": status,
                "detection_count": detection_count,
            }
            if reason:
                evidence_data["reason"] = reason
            if image_type:
                evidence_data["image_type"] = image_type
            if reason_code:
                evidence_data["reason_code"] = reason_code
            if record is None:
                record = EvidenceRecord(
                    assessment_id=assessment_id,
                    image_id=image_id,
                    evidence_type="damage_detection",
                    source_reference=MODEL_NAME,
                )
                db.add(record)
            record.evidence_data = evidence_data
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save damage inspection status.") from None


def _persist_findings(
    db: Session,
    assessment_id: UUID,
    image_id: UUID,
    result: DamageInferenceResult,
    image_type: str | None = None,
) -> ImageDamageResponse:
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
            scan = _existing_scan(db, assessment_id, image_id)
            if _stored_damage_status(scan, existing) in _COMPLETED_DAMAGE_STATUSES:
                return _analysis_response(
                    assessment_id,
                    image_id,
                    existing,
                    reused=True,
                    status=_stored_damage_status(scan, existing),
                    image_type=(scan.evidence_data or {}).get("image_type") if scan and isinstance(scan.evidence_data, dict) else image_type,
                )

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
            if scan is None:
                scan = EvidenceRecord(
                    assessment_id=assessment_id,
                    image_id=image_id,
                    evidence_type="damage_detection",
                    source_reference=MODEL_NAME,
                )
                db.add(scan)
            scan.evidence_data = {
                "status": (
                    "completed_with_findings"
                    if findings
                    else "completed_no_findings"
                ),
                "detection_count": len(findings),
                "inference_time_seconds": result.inference_seconds,
                "image_type": image_type,
            }
            db.flush()

            response = _analysis_response(
                assessment_id, image_id, findings, reused=False, image_type=image_type
            )
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
    image_type: DamageImageType | None = None,
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
        reason = "Stored image is unavailable."
        _record_inspection_outcome(
            db, assessment_id, image_id, "unavailable", reason=reason,
            reason_code="stored_image_unavailable",
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider=IsolatedDamageProvider.provider_name,
            status="unavailable",
            reused=False,
            findings=[],
            reason=reason,
            reason_code="stored_image_unavailable",
        )

    try:
        extension, _content_type = inspect_image(content)
    except InvalidImageError:
        reason = "Stored image is corrupt or unsupported."
        _record_inspection_outcome(
            db, assessment_id, image_id, "unavailable", reason=reason,
            reason_code="invalid_image",
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider=IsolatedDamageProvider.provider_name,
            status="unavailable",
            reused=False,
            findings=[],
            reason=reason,
            reason_code="invalid_image",
        )

    ocr_results = list(
        db.scalars(
            select(OCRResult).where(
                OCRResult.assessment_id == assessment_id,
                OCRResult.image_id == image_id,
            )
        ).all()
    )

    # Fail-safe classification: any classification error makes the image
    # ineligible, so YOLO is never called for an unclassified image.
    classification_failed = False
    try:
        decision = cached_decision(image, content, ocr_results, image_type)
        if decision is None:
            decision = classify_image(
                content=content,
                ocr_results=ocr_results,
                original_filename=image.original_filename,
                declared_image_type=image_type,
            )
            store_classification(image, decision)
            db.add(image)
            db.flush()
    except Exception as exc:
        classification_failed = True
        logger.warning("Image classification failed: %s", type(exc).__name__)
        decision = ImageEligibilityDecision(
            image_type="unknown",
            classification_method="classification_exception",
            eligible_for_damage_inference=False,
            rejection_reason=f"classification_failed: {type(exc).__name__}",
            classified_at=datetime.now(timezone.utc),
            signals={},
            cache_key=classification_cache_key(content, ocr_results, image_type),
        )
        store_classification(image, decision)
        try:
            db.add(image)
            db.flush()
        except SQLAlchemyError:
            db.rollback()

    if classification_failed:
        reason = "Image eligibility classification failed; damage inference was not run."
        _record_inspection_outcome(
            db,
            assessment_id,
            image_id,
            "failed",
            reason=reason,
            image_type=decision.image_type,
            reason_code="eligibility_classification_failed",
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider="image_classification",
            status="failed",
            reused=False,
            findings=[],
            image_type=decision.image_type,
            reason=reason,
            reason_code="eligibility_classification_failed",
        )

    if not decision.eligible_for_damage_inference:
        existing_eligibility = _existing_eligibility_scan(db, assessment_id, image_id)
        reused = existing_eligibility is not None
        if existing_eligibility is None:
            try:
                db.rollback()
                with db.begin():
                    locked_image_id = db.scalar(
                        select(UploadedImage.id).where(
                            UploadedImage.id == image_id,
                            UploadedImage.assessment_id == assessment_id,
                        ).with_for_update()
                    )
                    if locked_image_id is None:
                        raise HTTPException(status_code=404, detail="Image not found for assessment.")
                    if _existing_eligibility_scan(db, assessment_id, image_id) is None:
                        db.add(
                            EvidenceRecord(
                                assessment_id=assessment_id,
                                image_id=image_id,
                                evidence_type="damage_eligibility",
                                source_reference="image_classification",
                                evidence_data={
                                    "status": "unavailable",
                                    "image_type": decision.image_type,
                                    "reason": decision.rejection_reason,
                                    "classified_at": decision.classified_at.isoformat(),
                                },
                            )
                        )
                db.commit()
            except HTTPException:
                raise
            except SQLAlchemyError:
                db.rollback()
                raise HTTPException(status_code=500, detail="Unable to save image eligibility.") from None
        _record_inspection_outcome(
            db,
            assessment_id,
            image_id,
            "unavailable",
            reason=decision.rejection_reason,
            image_type=decision.image_type,
            reason_code=(
                "eligibility_not_submitted_insufficient_classification_evidence"
                if decision.rejection_reason == "insufficient_classification_evidence"
                else "image_rejected_by_eligibility"
            ),
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider="image_classification",
            status="unavailable",
            reused=reused,
            findings=[],
            image_type=decision.image_type,
            reason=decision.rejection_reason,
            reason_code=(
                "eligibility_not_submitted_insufficient_classification_evidence"
                if decision.rejection_reason == "insufficient_classification_evidence"
                else "image_rejected_by_eligibility"
            ),
        )

    try:
        existing = _existing_findings(db, assessment_id, image_id)
        scan = _existing_scan(db, assessment_id, image_id)
        stored_status = _stored_damage_status(scan, existing)
        if stored_status in _COMPLETED_DAMAGE_STATUSES:
            db.rollback()
            return _analysis_response(
                assessment_id, image_id, existing, reused=True, status=stored_status
            )
        db.rollback()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to retrieve existing damage findings.") from None

    try:
        result = provider.detect(content, extension)
    except DamageInputError:
        status_value: DamageInspectionStatus = "unavailable"
        reason = "Stored image is corrupt or unsupported."
        reason_code = "invalid_image"
    except DamageWorkerUnavailable as exc:
        status_value = "unavailable"
        reason_code = getattr(exc, "reason_code", "inference_unavailable")
        reason = (
            "Damage inference is disabled by configuration."
            if reason_code == "inference_disabled"
            else "The damage model checkpoint is unavailable."
            if reason_code == "checkpoint_unavailable"
            else "The damage inference runtime is unavailable."
            if reason_code in {"runtime_unavailable", "runner_unavailable"}
            else "Damage inference is unavailable."
        )
    except DamageWorkerTimeout:
        status_value = "unavailable"
        reason = "Damage inference timed out."
        reason_code = "inference_timeout"
    except DamageWorkerInvalidResponse:
        status_value = "unavailable"
        reason = "Damage inference returned an invalid response."
        reason_code = "invalid_worker_response"
    except DamageWorkerInferenceFailed as exc:
        logger.warning("Damage inference worker failed: %s", type(exc).__name__)
        status_value = "failed"
        reason = "Damage inference failed."
        reason_code = "inference_failed"
    except Exception as exc:
        logger.warning("Damage inference provider failed: %s", type(exc).__name__)
        status_value = "failed"
        reason = "Damage inference failed."
        reason_code = "inference_failed"
    else:
        return _persist_findings(db, assessment_id, image_id, result, decision.image_type)

    _record_inspection_outcome(
        db, assessment_id, image_id, status_value, reason=reason, reason_code=reason_code
    )
    return ImageDamageResponse(
        assessment_id=assessment_id,
        image_id=image_id,
        provider=IsolatedDamageProvider.provider_name,
        status=status_value,
        reused=False,
        findings=[],
        reason=reason,
        reason_code=reason_code,
    )


@router.post(
    "/{assessment_id}/damage",
    response_model=AssessmentDamageResponse,
    responses={404: {"description": "Assessment not found."}},
)
def inspect_assessment_damage(
    assessment_id: UUID,
    payload: AssessmentDamageRequest,
    db: Session = Depends(get_db),
    storage: ImageStorage = Depends(get_image_storage),
    provider: DamageProvider = Depends(get_damage_provider),
) -> AssessmentDamageResponse:
    """Run the existing per-image detector for up to ten selected assessment images."""
    try:
        if db.get(VehicleAssessment, assessment_id) is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment.") from None

    results: list[AssessmentImageDamageResult] = []
    for image_id in payload.image_ids:
        declared_image_type = payload.image_types.get(image_id)
        try:
            item = inspect_assessment_image_damage(
                assessment_id=assessment_id,
                image_id=image_id,
                db=db,
                storage=storage,
                provider=provider,
                image_type=declared_image_type,
            )
            results.append(
                AssessmentImageDamageResult(
                    image_id=image_id,
                    status=item.status,
                    reused=item.reused,
                    findings=item.findings,
                    error=item.reason,
                    image_type=item.image_type,
                    reason=item.reason,
                    reason_code=item.reason_code,
                )
            )
        except HTTPException as exc:
            db.rollback()
            results.append(
                AssessmentImageDamageResult(
                    image_id=image_id,
                    status="failed",
                    error=str(exc.detail),
                    reason=str(exc.detail),
                    reason_code="image_processing_failed",
                )
            )
        except Exception as exc:
            db.rollback()
            logger.warning(
                "Damage processing failed for image (%s).", type(exc).__name__
            )
            results.append(
                AssessmentImageDamageResult(
                    image_id=image_id,
                    status="failed",
                    error="Damage inspection failed.",
                    reason="Damage inspection failed.",
                    reason_code="image_processing_failed",
                )
            )

    failed_count = sum(result.status == "failed" for result in results)
    unavailable_count = sum(result.status == "unavailable" for result in results)
    if any(result.status == "completed_with_findings" for result in results):
        status_value = "completed_with_findings"
    elif failed_count:
        status_value = "failed"
    elif unavailable_count:
        status_value = "unavailable"
    else:
        status_value = "completed_no_findings"
    return AssessmentDamageResponse(
        assessment_id=assessment_id,
        status=status_value,
        processed_count=sum(
            result.status in _COMPLETED_DAMAGE_STATUSES for result in results
        ),
        failed_count=failed_count,
        unavailable_count=unavailable_count,
        results=results,
    )
