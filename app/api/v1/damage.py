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
from app.pricing.condition import assess_condition
from app.schemas.damage import (
    AssessmentDamageRequest,
    AssessmentDamageResponse,
    AssessmentImageDamageResult,
    DamageInspectionStatus,
    DamageFindingResponse,
    ImageDamageResponse,
)
from app.storage.images import ImageContentUnavailable, read_image_content


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
    status: DamageInspectionStatus | None = None,
    image_type: str | None = None,
    inspection_outcome: str | None = None,
    condition_category: str | None = None,
    requires_human_review: bool = False,
    decision: ImageEligibilityDecision | None = None,
    inference_seconds: float | None = None,
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
        inspection_outcome=inspection_outcome or (
            "completed_with_findings" if ordered_findings else "completed_no_findings"
        ),
        condition_category=condition_category,
        requires_human_review=requires_human_review,
        classification_source=decision.classification_source if decision else None,
        damage_eligible=decision.eligible_for_damage_inference if decision else None,
        damage_rejection_reason=decision.rejection_reason if decision else None,
        classification_method=decision.classification_method if decision else None,
        classification_evidence=decision.signals if decision else None,
        inference_seconds=inference_seconds,
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
        ).order_by(EvidenceRecord.created_at.desc(), EvidenceRecord.id.desc())
    )


def _persist_eligibility_decision(
    db: Session, assessment_id: UUID, image_id: UUID,
    decision: ImageEligibilityDecision,
) -> None:
    """Persist classification evidence in its own transaction boundary."""
    try:
        db.rollback()
        with db.begin():
            image = db.scalar(select(UploadedImage).where(
                UploadedImage.id == image_id,
                UploadedImage.assessment_id == assessment_id,
            ).with_for_update())
            if image is None:
                raise HTTPException(status_code=404, detail="Image not found for assessment.")
            record = _existing_eligibility_scan(db, assessment_id, image_id)
            if record is None:
                record = EvidenceRecord(
                    assessment_id=assessment_id,
                    image_id=image_id,
                    evidence_type="damage_eligibility",
                    source_reference="image_classification",
                )
                db.add(record)
            state = "eligible" if decision.eligible_for_damage_inference else (
                "classification_uncertain" if decision.image_type == "unknown" else "ineligible"
            )
            record.evidence_data = {
                **decision.as_metadata(),
                "status": state,
                "inspection_outcome": state,
                "classification_source": decision.classification_source,
                "damage_eligible": decision.eligible_for_damage_inference,
                "reason": decision.rejection_reason,
            }
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save image eligibility.") from None


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
    decision: ImageEligibilityDecision | None = None,
    inference_seconds: float | None = None,
    inspection_outcome: str | None = None,
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
                "inspection_outcome": inspection_outcome or (
                    "inference_failed" if status == "failed" else
                    "completed_no_findings" if status == "completed_no_findings" else
                    "completed_with_findings" if status == "completed_with_findings" else
                    "classification_uncertain" if image_type == "unknown" else
                    "ineligible" if status == "unavailable" else status
                ),
            }
            if decision is not None:
                evidence_data.update({
                    "image_type": decision.image_type,
                    "classification_source": decision.classification_source,
                    "damage_eligible": decision.eligible_for_damage_inference,
                    "damage_rejection_reason": decision.rejection_reason,
                    "classification_method": decision.classification_method,
                })
            if inference_seconds is not None:
                evidence_data["inference_seconds"] = inference_seconds
            if reason:
                evidence_data["reason"] = reason
            if image_type:
                evidence_data["image_type"] = image_type
            outcome_condition = assess_condition(
                [{"damage_eligible": True, "status": status}], []
            ) if status in _COMPLETED_DAMAGE_STATUSES else {
                "category": "inspection_failed" if status == "failed" else "inspection_unavailable",
                "requires_human_review": True,
            }
            evidence_data["condition_category"] = outcome_condition["category"]
            evidence_data["requires_human_review"] = outcome_condition["requires_human_review"]
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
    decision: ImageEligibilityDecision | None = None,
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
                "inspection_outcome": "completed_with_findings" if findings else "completed_no_findings",
                "damage_types": [finding.damage_type for finding in findings],
                "inference_seconds": result.inference_seconds,
                "image_type": decision.image_type if decision else None,
                "classification_source": decision.classification_source if decision else None,
                "damage_eligible": decision.eligible_for_damage_inference if decision else None,
            }
            condition = assess_condition(
                [{"damage_eligible": True, "status": "completed_with_findings" if findings else "completed_no_findings"}],
                [{"damage_type": finding.damage_type} for finding in findings],
            )
            scan.evidence_data["condition_category"] = condition["category"]
            scan.evidence_data["requires_human_review"] = condition["requires_human_review"]
            db.flush()

            response = _analysis_response(
                assessment_id, image_id, findings, reused=False,
                image_type=decision.image_type if decision else None,
                inspection_outcome="completed_with_findings" if findings else "completed_no_findings",
                condition_category=condition["category"],
                requires_human_review=condition["requires_human_review"],
                decision=decision,
                inference_seconds=result.inference_seconds,
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
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment image.") from None

    try:
        content = read_image_content(image)
    except ImageContentUnavailable:
        reason = "Stored image is unavailable."
        _record_inspection_outcome(
            db, assessment_id, image_id, "unavailable", reason=reason
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider=IsolatedDamageProvider.provider_name,
            status="unavailable",
            reused=False,
            findings=[],
            reason=reason,
        )
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment image.") from None

    try:
        extension, _content_type = inspect_image(content)
    except InvalidImageError:
        reason = "Stored image is corrupt or unsupported."
        _record_inspection_outcome(
            db, assessment_id, image_id, "unavailable", reason=reason
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider=IsolatedDamageProvider.provider_name,
            status="unavailable",
            reused=False,
            findings=[],
            reason=reason,
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
        metadata_decision = (image.image_metadata or {}).get("damage_eligibility")
        if isinstance(metadata_decision, dict) and metadata_decision.get("classification_source") == "user_input":
            decision = classify_image(content=content, selected_type=metadata_decision.get("image_type"))
        else:
            decision = cached_decision(image, content, ocr_results)
        if decision is None:
            decision = classify_image(
                content=content,
                ocr_results=ocr_results,
                original_filename=image.original_filename,
            )
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
            cache_key=classification_cache_key(content, ocr_results),
        )
    store_classification(image, decision)
    db.add(image)
    try:
        # A commit, rather than a flush, is required: later status helpers
        # intentionally roll back their read transaction.
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save image classification.") from None

    if classification_failed:
        reason = "Image eligibility classification failed; damage inference was not run."
        _record_inspection_outcome(
            db,
            assessment_id,
            image_id,
            "unavailable",
            reason=reason,
            image_type=decision.image_type,
            decision=decision,
            inspection_outcome="classification_uncertain",
        )
        return ImageDamageResponse(
            assessment_id=assessment_id,
            image_id=image_id,
            provider="image_classification",
            status="unavailable",
            reused=False,
            findings=[],
            image_type=decision.image_type,
            reason=reason,
            inspection_outcome="classification_uncertain",
            condition_category="inspection_unavailable",
            requires_human_review=True,
            classification_source=decision.classification_source,
            damage_eligible=False,
            damage_rejection_reason=decision.rejection_reason,
            classification_method=decision.classification_method,
            classification_evidence=decision.signals,
        )

    existing_eligibility = _existing_eligibility_scan(db, assessment_id, image_id)
    eligibility_reused = existing_eligibility is not None
    try:
        _persist_eligibility_decision(db, assessment_id, image_id, decision)
    except HTTPException:
        raise

    if not decision.eligible_for_damage_inference:
        reused = eligibility_reused
        _record_inspection_outcome(
            db,
            assessment_id,
            image_id,
            "unavailable",
            reason=decision.rejection_reason,
            image_type=decision.image_type,
            decision=decision,
            inspection_outcome="classification_uncertain" if decision.image_type == "unknown" else "ineligible",
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
            inspection_outcome="classification_uncertain" if decision.image_type == "unknown" else "ineligible",
            classification_source=decision.classification_source,
            damage_eligible=False,
            damage_rejection_reason=decision.rejection_reason,
            classification_method=decision.classification_method,
            classification_evidence=decision.signals,
        )

    try:
        existing = _existing_findings(db, assessment_id, image_id)
        scan = _existing_scan(db, assessment_id, image_id)
        stored_status = _stored_damage_status(scan, existing)
        if stored_status in _COMPLETED_DAMAGE_STATUSES:
            stored_evidence = scan.evidence_data if scan and isinstance(scan.evidence_data, dict) else {}
            stored_classification = (image.image_metadata or {}).get("damage_eligibility", {})
            db.rollback()
            return _analysis_response(
                assessment_id, image_id, existing, reused=True, status=stored_status,
                image_type=stored_evidence.get("image_type") or stored_classification.get("image_type"),
                inspection_outcome=stored_evidence.get("inspection_outcome"),
                condition_category=stored_evidence.get("condition_category"),
                requires_human_review=bool(stored_evidence.get("requires_human_review")),
                decision=decision,
                inference_seconds=stored_evidence.get("inference_seconds"),
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
    except DamageWorkerUnavailable:
        status_value = "unavailable"
        reason = "Damage inference is unavailable."
    except DamageWorkerTimeout:
        status_value = "failed"
        reason = "Damage inference timed out."
    except DamageWorkerInvalidResponse:
        status_value = "failed"
        reason = "Damage inference returned an invalid response."
    except DamageWorkerInferenceFailed as exc:
        logger.warning("Damage inference worker failed: %s", type(exc).__name__)
        status_value = "failed"
        reason = "Damage inference failed."
    except Exception as exc:
        logger.warning("Damage inference provider failed: %s", type(exc).__name__)
        status_value = "failed"
        reason = "Damage inference failed."
    else:
        return _persist_findings(db, assessment_id, image_id, result, decision)

    _record_inspection_outcome(
        db, assessment_id, image_id, status_value, reason=reason,
        decision=decision,
        inspection_outcome="inference_failed" if status_value == "failed" else "inference_unavailable",
    )
    return ImageDamageResponse(
        assessment_id=assessment_id,
        image_id=image_id,
        provider=IsolatedDamageProvider.provider_name,
        status=status_value,
        reused=False,
        findings=[],
        reason=reason,
        image_type=decision.image_type,
        inspection_outcome="inference_failed" if status_value == "failed" else "inference_unavailable",
        condition_category="inspection_failed" if status_value == "failed" else "inspection_unavailable",
        requires_human_review=True,
        classification_source=decision.classification_source,
        damage_eligible=True,
        damage_rejection_reason=decision.rejection_reason,
        classification_method=decision.classification_method,
        classification_evidence=decision.signals,
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
        try:
            item = inspect_assessment_image_damage(
                assessment_id=assessment_id,
                image_id=image_id,
                db=db,
                provider=provider,
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
                    inspection_outcome=item.inspection_outcome,
                    condition_category=item.condition_category,
                    requires_human_review=item.requires_human_review,
                    classification_source=item.classification_source,
                    damage_eligible=item.damage_eligible,
                    damage_rejection_reason=item.damage_rejection_reason,
                    classification_method=item.classification_method,
                    classification_evidence=item.classification_evidence,
                    inference_seconds=item.inference_seconds,
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
