from uuid import UUID
from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.v1.damage import get_damage_provider, inspect_assessment_damage
from app.api.v1.ocr import get_ocr_client, recognize_assessment_image
from app.api.v1.pricing import (
    _create_price_estimate,
    _persist_manual_pricing_specifications,
)
from app.api.v1.reports import _generate_assessment_report_snapshot, get_market_listing_provider
from app.api.v1.vehicle_lookup import (
    _challan_state,
    _fastag_state,
    _store_service_result,
    get_challan_provider,
    get_fastag_provider,
    get_vehicle_registration_provider,
    lookup_registration_from_manual,
    lookup_registration_from_ocr,
    normalize_manual_registration,
)
from app.core.config import settings
from app.db.models import AssessmentReport, EvidenceRecord, OCRResult, VehicleAssessment
from app.db.session import get_db
from app.pricing.model_defaults import get_model_defaults
from app.pricing.provider import get_price_provider
from app.vehicle_specs.resolver import VehicleSpecificationResolver
from app.vehicle_specs.sources import is_approved_source
from app.schemas.damage import AssessmentDamageRequest
from app.schemas.pricing import (
    PriceEstimateRequest,
    PricingUserInput,
    VehiclePriceInput,
)
from app.schemas.reports import AssessmentReportResponse
from app.schemas.workflow import (
    AssessmentWorkflowRequest,
    AssessmentWorkflowResponse,
    WorkflowStageError,
    WorkflowStageResult,
)
from app.schemas.vehicle_services import ChallanDetails, FastagDetails
from app.storage.images import ImageStorage, get_image_storage
from app.ocr.registration import build_registration_ocr_assessment, normalize_registration_candidate


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment workflow"])

_PRICE_INPUT_FIELDS = set(VehiclePriceInput.model_fields)
_XGBOOST_REQUIRED_FIELDS = (
    "make",
    "model",
    "seller_type",
    "fuel_type",
    "transmission",
    "registration_year",
    "odometer_km",
    "mileage_kmpl",
    "engine_cc",
    "max_power_bhp",
    "seats",
)
_INDIAN_REGISTRATION_PREFIXES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML",
    "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN",
    "TR", "TS", "UA", "UK", "UP", "WB",
}


def _is_indian_registration_candidate(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().upper()
    return normalized.startswith("BH") or normalized[:2] in _INDIAN_REGISTRATION_PREFIXES


def _lookup_price_features(details) -> dict:
    source = details.model_dump(mode="python", exclude_none=True)
    mapped = {
        key: source[key]
        for key in ("make", "model", "variant", "fuel_type", "engine_cc", "seats", "max_power_bhp", "transmission", "owner_count", "manufacturing_year", "mileage_kmpl", "seats", "rto", "vehicle_class", "registration_valid_until", "fitness_expiry", "puc_expiry", "insurance_expiry")
        if source.get(key) is not None
    }
    if isinstance(mapped.get("fuel_type"), str):
        mapped["fuel_type"] = mapped["fuel_type"].strip().casefold()
    if isinstance(mapped.get("transmission"), str):
        mapped["transmission"] = mapped["transmission"].strip().casefold()
    # The price model needs first registration year. Manufacturing year is a
    # different fact and is deliberately not substituted for it.
    registration_date = source.get("registration_date")
    if registration_date is not None and hasattr(registration_date, "year"):
        mapped["registration_year"] = registration_date.year
    return mapped


def _saved_lookup_for_candidate(db: Session, assessment_id: UUID, candidate: str):
    rows = db.scalars(
        select(OCRResult)
        .where(
            OCRResult.assessment_id == assessment_id,
            OCRResult.result_type == "registration_lookup",
            OCRResult.status == "completed",
        )
        .order_by(OCRResult.created_at.desc(), OCRResult.id.desc())
    ).all()
    for row in rows:
        source = row.result_data if isinstance(row.result_data, dict) else {}
        registration = row.raw_text or source.get("registration_number")
        if normalize_registration_candidate(registration) != candidate:
            continue
        try:
            from app.schemas.vehicle_lookup import VehicleRegistrationDetails

            return VehicleRegistrationDetails.model_validate(source), row.provider
        except ValidationError:
            continue
    return None


def _workflow_lookup_stage(
    assessment_id: UUID,
    db: Session,
    ocr_results: list[dict],
    provider,
    manual_registration: str | None = None,
) -> tuple[WorkflowStageResult, object | None]:
    if manual_registration is None:
        assessment = build_registration_ocr_assessment(ocr_results)
        if assessment["status"] != "candidate":
            if assessment["status"] == "needs_review":
                return WorkflowStageResult(
                    status="unavailable",
                    result={
                        "registration_number_candidate": assessment[
                            "normalized_candidate"
                        ],
                        "confidence": assessment["confidence"],
                        "requires_manual_correction": True,
                    },
                    errors=[WorkflowStageError(
                        code="registration_needs_review",
                        detail="OCR registration candidates need review; vehicle lookup was skipped.",
                    )],
                ), None
            return WorkflowStageResult(
                status="unavailable",
                errors=[WorkflowStageError(
                    code="registration_not_found",
                    detail="No readable registration candidate was available for vehicle lookup.",
                )],
            ), None
        candidate = assessment["normalized_candidate"]
    else:
        candidate = manual_registration

    if manual_registration is None and not _is_indian_registration_candidate(candidate):
        return WorkflowStageResult(
            status="unavailable",
            errors=[WorkflowStageError(
                code="registration_not_found",
                detail="No supported Indian registration candidate was available for vehicle lookup.",
            )],
        ), None

    saved = _saved_lookup_for_candidate(db, assessment_id, candidate)
    if saved is not None:
        details, provider_name = saved
        state = "available" if details.make and details.model else "partial"
        return WorkflowStageResult(
            status=state,
            result={
                "status": state,
                "provider": provider_name,
                "details": details.model_dump(mode="json"),
                "field_provenance": {
                    key: "vehicle_api"
                    for key, value in details.model_dump(exclude_none=True).items()
                    if value is not None
                },
            },
        ), details

    # The provider factory already enforces this setting. Recheck the flag here
    # so dependency overrides or future wiring cannot bypass explicit opt-in.
    if not settings.vehicle_lookup_enabled or provider is None:
        return WorkflowStageResult(
            status="unavailable",
            errors=[WorkflowStageError(
                code="provider_disabled",
                detail="Vehicle registration lookup is disabled or not configured; no lookup was made.",
            )],
        ), None

    try:
        if manual_registration is not None:
            details = lookup_registration_from_manual(
                assessment_id=assessment_id,
                registration_number=manual_registration,
                db=db,
                provider=provider,
            )
            provider_name = provider.provider_name
        else:
            matches = [
                item for item in ocr_results
                if item.get("status") == "completed"
                and normalize_registration_candidate(item.get("text")) == candidate
                and item.get("id")
            ]
            if not matches:
                return WorkflowStageResult(
                    status="unavailable",
                    errors=[WorkflowStageError(
                        code="registration_needs_review",
                        detail="No matching completed OCR result was available; vehicle lookup was skipped.",
                    )],
                ), None
            selected = max(matches, key=lambda item: float(item.get("confidence") or 0))
            lookup = lookup_registration_from_ocr(
                assessment_id=assessment_id,
                ocr_result_id=UUID(selected["id"]),
                db=db,
                provider=provider,
            )
            details = lookup.details
            provider_name = lookup.provider
    except HTTPException as exc:
        return WorkflowStageResult(
            status="unavailable" if exc.status_code == 503 else "failed",
            errors=[WorkflowStageError(
                code={503: "provider_unavailable", 422: "invalid_registration", 502: "provider_failed"}.get(
                    exc.status_code, "lookup_failed"
                ),
                detail=str(exc.detail),
            )],
        ), None
    except Exception:
        db.rollback()
        return WorkflowStageResult(
            status="failed",
            errors=[WorkflowStageError(code="lookup_failed", detail="Vehicle registration lookup failed.")],
        ), None

    state = "available" if details.make and details.model else "partial"
    return WorkflowStageResult(
        status=state,
        result={
            "status": state,
            "provider": provider_name,
            "details": details.model_dump(mode="json"),
            "field_provenance": {
                key: "vehicle_api"
                for key, value in details.model_dump(exclude_none=True).items()
                if value is not None
            },
        },
    ), details


def _compose_price_input(
    assessment: VehicleAssessment,
    lookup_details,
    user_details,
    db=None,
) -> tuple[VehiclePriceInput | None, list[str], list[str], dict[str, str]]:
    provenance: dict[str, str] = {}
    merged: dict = {}

    # Priority 1: validated registration-provider fields.
    if lookup_details is not None:
        lookup_features = _lookup_price_features(lookup_details)
        for key, value in lookup_features.items():
            if key in _PRICE_INPUT_FIELDS and value is not None and key not in merged:
                merged[key] = value
                provenance[key] = "vehicle_api"

    # Assessment JSON is user supplied. Only explicitly permitted manual fields
    # may be reused from it.
    stored = assessment.vehicle_details if isinstance(assessment.vehicle_details, dict) else {}
    for key in (
        "make", "model", "variant", "registration_year", "odometer_km", "fuel_type",
        "seller_type", "transmission", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats",
    ):
        value = stored.get(key)
        if value is not None:
            merged[key] = value
            provenance[key] = "user_input"

    # Explicit manual inputs override API values, and are persisted for retries.
    if user_details is not None:
        user_dump = user_details.model_dump(exclude_unset=True)
        for key in (
            "make",
            "model",
            "variant",
            "registration_year",
            "odometer_km",
            "fuel_type",
            "seller_type",
            "transmission",
            "mileage_kmpl",
            "engine_cc",
            "max_power_bhp",
            "seats",
        ):
            value = user_dump.get(key)
            if value is not None:
                merged[key] = value
                provenance[key] = "user_input"

    # Priority 2: exact, explicitly curated model defaults. No defaults are
    # currently enabled because the available captured catalog is unverified.
    model_defaults = get_model_defaults(
        merged.get("make"),
        merged.get("model"),
        merged.get("variant"),
    )
    if model_defaults is not None:
        for field, value in model_defaults.items():
            if field in _PRICE_INPUT_FIELDS and merged.get(field) is None and value is not None:
                merged[field] = value
                provenance[field] = "model_default"

    # Verified database specifications may also provide model-specific defaults.
    if (
        db is not None
        and merged.get("make")
        and merged.get("model")
        and merged.get("variant")
        and lookup_details is not None
    ):
        from app.vehicle_specs.resolver import PRICING_FIELDS

        resolution = VehicleSpecificationResolver(db).resolve(
            make=merged["make"],
            model=merged["model"],
            variant=merged["variant"],
            fuel_type=merged.get("fuel_type"),
            manufacturing_year=getattr(lookup_details, "manufacturing_year", None),
        )
        resolved = resolution.resolved
        if (
            resolved is not None
            and resolved.variant is not None
            and resolved.variant.casefold() == merged["variant"].casefold()
            and resolved.verification_status == "verified"
            and is_approved_source(resolved.source_provider)
        ):
            for field in PRICING_FIELDS:
                value = getattr(resolved, field, None)
                source = resolution.provenance.get(field) or resolved.source_provider
                if (
                    merged.get(field) is None
                    and value is not None
                    and is_approved_source(source)
                ):
                    merged[field] = value
                    provenance[field] = "model_default"

    manually_correctable = (
        "make", "model", "variant", "registration_year", "odometer_km", "fuel_type",
        "seller_type", "transmission", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats",
    )
    try:
        PricingUserInput.model_validate({
            field: merged[field]
            for field in manually_correctable
            if merged.get(field) is not None
        })
    except ValidationError as exc:
        invalid = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
        return None, [], invalid, provenance

    missing = [field for field in _XGBOOST_REQUIRED_FIELDS if merged.get(field) is None]

    if missing:
        return None, missing, [], provenance

    try:
        return VehiclePriceInput.model_validate(merged), [], [], provenance
    except ValidationError as exc:
        invalid = sorted({str(error["loc"][0]) for error in exc.errors() if error.get("loc")})
        return None, [], invalid, provenance


def _stage_error(exc: Exception, *, image_id: UUID | None = None) -> WorkflowStageError:
    if isinstance(exc, HTTPException):
        if exc.status_code == 422 and exc.detail == "No readable text was found in the image.":
            return WorkflowStageError(image_id=image_id, code="no_text", detail="No readable text was detected.")
        code = {
            404: "not_found",
            422: "invalid_input",
            503: "provider_unavailable",
            504: "provider_timeout",
        }.get(exc.status_code, "stage_failed")
        detail = str(exc.detail)
    else:
        code = "stage_failed"
        detail = "The workflow stage failed."
    return WorkflowStageError(image_id=image_id, code=code, detail=detail)


def _stage_status(success_count: int, errors: list[WorkflowStageError]) -> str:
    if not errors:
        return "available"
    if success_count:
        return "partial"
    if all(error.code == "no_text" for error in errors):
        return "no_text"
    if all(error.code in {"provider_not_configured", "provider_unavailable", "provider_timeout"} for error in errors):
        return "unavailable"
    return "failed"


def _ocr_stage_error(db: Session, assessment_id: UUID, image_id: UUID, exc: Exception) -> WorkflowStageError:
    """Prefer the persisted attempt outcome over inferring from a generic HTTP response."""
    try:
        attempt = db.scalar(
            select(EvidenceRecord)
            .where(
                EvidenceRecord.assessment_id == assessment_id,
                EvidenceRecord.image_id == image_id,
                EvidenceRecord.evidence_type == "ocr_attempt",
            )
            .order_by(EvidenceRecord.created_at.desc(), EvidenceRecord.id.desc())
        )
        evidence = attempt.evidence_data if attempt and isinstance(attempt.evidence_data, dict) else {}
        code = evidence.get("error_code")
        if code in {
            "provider_not_configured", "provider_timeout", "provider_unavailable",
            "stored_image_unavailable", "invalid_image", "image_preprocessing_failed",
            "invalid_provider_response", "no_text_detected", "inference_failed",
            "unexpected_provider_error", "result_persistence_failed",
        }:
            safe_details = {
                "provider_not_configured": "No OCR provider is configured.",
                "provider_timeout": "The OCR provider timed out.",
                "provider_unavailable": "The OCR provider is unavailable.",
                "stored_image_unavailable": "The stored image could not be read.",
                "invalid_image": "The image is corrupt or unsupported.",
                "image_preprocessing_failed": "The image could not be prepared for OCR.",
                "invalid_provider_response": "The OCR provider returned an invalid response.",
                "no_text_detected": "No readable text was detected.",
                "inference_failed": "The OCR provider reported an inference failure.",
                "unexpected_provider_error": "The OCR attempt failed unexpectedly.",
                "result_persistence_failed": "The OCR result could not be saved.",
            }
            stage_code = "no_text" if code == "no_text_detected" else code
            return WorkflowStageError(image_id=image_id, code=stage_code, detail=safe_details[code])
    except SQLAlchemyError:
        db.rollback()
    return _stage_error(exc, image_id=image_id)


@router.post("/{assessment_id}/workflow", response_model=AssessmentWorkflowResponse)
def run_assessment_workflow(
    assessment_id: UUID,
    payload: AssessmentWorkflowRequest,
    db: Session = Depends(get_db),
    storage: ImageStorage = Depends(get_image_storage),
    ocr_client=Depends(get_ocr_client),
    damage_provider=Depends(get_damage_provider),
    vehicle_lookup_provider=Depends(get_vehicle_registration_provider),
    price_provider=Depends(get_price_provider),
    market_provider=Depends(get_market_listing_provider),
    challan_provider=Depends(get_challan_provider),
    fastag_provider=Depends(get_fastag_provider),
) -> AssessmentWorkflowResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment.") from None
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    # Serialize concurrent workflow executions for the same assessment using
    # database-level row locking. This prevents duplicate workflow executions
    # (the same assessment workflow endpoint being called twice for the same
    # assessment ID) while preserving legitimate retries after failures:
    # a workflow that already completed or partially failed will return the
    # stored progress instead of starting over.
    try:
        assessment = db.scalars(
            select(VehicleAssessment)
            .where(VehicleAssessment.id == assessment_id)
            .with_for_update()
        ).one()
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to serialize workflow execution.") from None
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    manual_registration = (
        normalize_manual_registration(assessment.registration_number)
        if assessment.registration_number
        else None
    )
    if manual_registration is not None and assessment.registration_number != manual_registration:
        assessment.registration_number = manual_registration
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail="Unable to save the normalized registration number.",
            ) from None

    stages: dict[str, WorkflowStageResult] = {}

    ocr_results: list[dict] = []
    ocr_errors: list[WorkflowStageError] = []
    # Idempotency: skip rerunning completed stages unless manual input needs
    # to be applied to this workflow execution.
    try:
        existing_report = db.scalar(
            select(AssessmentReport).where(AssessmentReport.assessment_id == assessment_id)
        )
    except SQLAlchemyError:
        existing_report = None

    if (
        existing_report is not None
        and isinstance(existing_report.report_data, dict)
        and isinstance(existing_report.report_data.get("stages"), dict)
    ):
        stored_stages = existing_report.report_data["stages"]
        pricing_stage = stored_stages.get("pricing", {})
        stored_market = existing_report.report_data.get("market_valuation", {})
        market_needs_vehicle_refresh = (
            isinstance(stored_market, dict)
            and stored_market.get("refresh_status") == "insufficient_vehicle_details"
        )
        manual_specification_submitted = payload.vehicle is not None and any(
            getattr(payload.vehicle, field) is not None
            for field in (
                "make", "model", "variant", "registration_year", "odometer_km", "fuel_type",
                "seller_type", "transmission", "mileage_kmpl", "engine_cc", "max_power_bhp", "seats",
            )
        )
        if (
            isinstance(pricing_stage, dict)
            and pricing_stage.get("status") in {"available", "failed"}
            and not market_needs_vehicle_refresh
            and not manual_specification_submitted
            and manual_registration is None
        ):
            # Return the stored workflow response; pricing is the final stage and
            # its result is already persisted. The frontend can reuse the report
            # without re-running the entire workflow.
            cached_stages = dict(stored_stages)
            cached_report = AssessmentReportResponse.model_validate(existing_report).model_dump(mode="json")
            cached_stages["report"] = {"status": "available", "result": cached_report}
            market = existing_report.report_data.get("market_valuation", {})
            cached_stages["market_valuation"] = {
                "status": "available" if isinstance(market, dict) and market.get("status") == "available" else "unavailable",
                "result": {
                    "status": market.get("status") if isinstance(market, dict) else "unavailable",
                    "refresh_status": market.get("refresh_status") if isinstance(market, dict) else None,
                    "evidence_count": market.get("evidence_count", 0) if isinstance(market, dict) else 0,
                    "market_asking_price_range": market.get("market_asking_price_range") if isinstance(market, dict) else None,
                },
            }
            service_rows = db.scalars(
                select(OCRResult).where(
                    OCRResult.assessment_id == assessment_id,
                    OCRResult.result_type.in_(("challan_lookup", "fastag_lookup")),
                )
            ).all()
            for result_type in ("challan_lookup", "fastag_lookup"):
                row = next((item for item in reversed(service_rows) if item.result_type == result_type), None)
                data = row.result_data if row and isinstance(row.result_data, dict) else {}
                state = data.get("state") or "not_run"
                cached_stages[result_type] = {
                    "status": "completed" if state == "available" else state,
                    "result": {"status": state, "provider": row.provider if row else None,
                               "details": data.get("details"), "error_code": data.get("error_code")},
                }
            statuses = [
                item.get("status")
                for key, item in cached_stages.items()
                if key not in {"report", "market_valuation"}
                and (key != "vehicle_details_lookup" or settings.vehicle_lookup_enabled)
            ]
            has_success = any(item in {"available", "completed", "partial", "completed_with_findings", "completed_no_findings", "disabled", "not_run"} for item in statuses)
            has_problem = any(item in {"partial", "unavailable", "failed", "no_text"} for item in statuses)
            return AssessmentWorkflowResponse(
                assessment_id=assessment_id,
                status="partial" if has_problem and has_success else "failed" if not has_success else "completed",
                stages=cached_stages,
            )
    try:
        if _persist_manual_pricing_specifications(assessment, payload.vehicle):
            db.commit()
            db.refresh(assessment)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to save manually supplied vehicle specifications.",
        ) from None
    if manual_registration is None:
        for image_id in payload.image_ids:
            try:
                result = recognize_assessment_image(
                    assessment_id=assessment_id,
                    image_id=image_id,
                    db=db,
                    storage=storage,
                    client=ocr_client,
                )
                ocr_results.append(result.model_dump(mode="json"))
            except Exception as exc:
                db.rollback()
                ocr_errors.append(_ocr_stage_error(db, assessment_id, image_id, exc))
        ocr_status = _stage_status(len(ocr_results), ocr_errors)
        stages["ocr"] = WorkflowStageResult(
            status=ocr_status,
            result=ocr_results,
            errors=ocr_errors,
        )
    else:
        stages["ocr"] = WorkflowStageResult(
            status="skipped",
            result=[],
        )

    try:
        lookup_stage, lookup_details = _workflow_lookup_stage(
            assessment_id,
            db,
            ocr_results,
            vehicle_lookup_provider,
            manual_registration=manual_registration,
        )
    except SQLAlchemyError:
        db.rollback()
        lookup_stage = WorkflowStageResult(
            status="failed",
            errors=[WorkflowStageError(code="lookup_storage_failed", detail="Unable to read vehicle lookup results.")],
        )
        lookup_details = None
    stages["vehicle_details_lookup"] = lookup_stage

    # Vehicle services only run for a single, normalized registration
    # candidate. OCR disagreement/low confidence never selects a winner.
    registration_candidate = (
        manual_registration
        if _is_indian_registration_candidate(manual_registration)
        else None
    )
    if registration_candidate is None:
        registration_assessment = build_registration_ocr_assessment(ocr_results)
        candidate = registration_assessment.get("normalized_candidate")
        if registration_assessment.get("status") == "candidate" and _is_indian_registration_candidate(candidate):
            registration_candidate = candidate

    def vehicle_service_stage(result_type, enabled, provider, lookup_fn, result_model):
        if registration_candidate is None:
            outcome = SimpleNamespace(status="not_run", provider=None, details=None, error_code="registration_unavailable")
        elif not enabled:
            outcome = SimpleNamespace(status="disabled", provider=None, details=None, error_code="disabled")
        else:
            saved = db.scalar(
                select(OCRResult)
                .where(
                    OCRResult.assessment_id == assessment_id,
                    OCRResult.result_type == result_type,
                    OCRResult.raw_text == registration_candidate,
                )
                .order_by(OCRResult.created_at.desc(), OCRResult.id.desc())
            )
            saved_data = saved.result_data if saved and isinstance(saved.result_data, dict) else {}
            saved_status = saved_data.get("state")
            if saved is not None and saved_status in {"available", "partial", "unavailable", "failed"}:
                try:
                    details = result_model.model_validate(saved_data.get("details")) if saved_data.get("details") is not None else None
                except ValidationError:
                    details = None
                outcome = SimpleNamespace(
                    status=saved_status,
                    provider=saved.provider,
                    details=details,
                    error_code=saved_data.get("error_code"),
                )
            else:
                try:
                    outcome = lookup_fn(provider, registration_candidate)
                except Exception:
                    outcome = SimpleNamespace(status="failed", provider=None, details=None, error_code="provider_error")
        image_id = next(
            (
                UUID(item["image_id"])
                for item in ocr_results
                if registration_candidate
                and item.get("image_id")
                and normalize_registration_candidate(item.get("text")) == registration_candidate
            ),
            None,
        )
        try:
            _store_service_result(
                db,
                assessment_id=assessment_id,
                image_id=image_id,
                result_type=result_type,
                registration_number=registration_candidate or "",
                result=outcome,
            )
            db.commit()
        except SQLAlchemyError:
            db.rollback()
            return WorkflowStageResult(status="failed", errors=[WorkflowStageError(code="result_persistence_failed", detail="Vehicle service results could not be saved.")])
        result_status = outcome.status
        stage_status = "completed" if result_status == "available" else result_status
        if result_status in {"available", "partial"}:
            return WorkflowStageResult(status=stage_status, result=outcome.model_dump(mode="json") if hasattr(outcome, "model_dump") else {
                "status": result_status,
                "provider": outcome.provider,
                "details": outcome.details.model_dump(mode="json") if outcome.details is not None else None,
            })
        return WorkflowStageResult(
            status=stage_status,
            result={"status": stage_status, "error_code": outcome.error_code},
            errors=[] if stage_status in {"disabled", "not_run"} else [WorkflowStageError(code=outcome.error_code or "provider_unavailable", detail="Vehicle service lookup did not return usable data.")],
        )

    try:
        stages["challan_lookup"] = vehicle_service_stage(
            "challan_lookup", settings.challan_lookup_enabled, challan_provider, _challan_state,
            ChallanDetails,
        )
    except Exception:
        db.rollback()
        stages["challan_lookup"] = WorkflowStageResult(status="failed", errors=[WorkflowStageError(code="provider_error", detail="Challan lookup failed.")])
    try:
        stages["fastag_lookup"] = vehicle_service_stage(
            "fastag_lookup", settings.fastag_lookup_enabled, fastag_provider, _fastag_state,
            FastagDetails,
        )
    except Exception:
        db.rollback()
        stages["fastag_lookup"] = WorkflowStageResult(status="failed", errors=[WorkflowStageError(code="provider_error", detail="FASTag lookup failed.")])

    try:
        damage_response = inspect_assessment_damage(
            assessment_id=assessment_id,
            payload=AssessmentDamageRequest(image_ids=payload.image_ids, image_types=payload.image_types),
            db=db,
            storage=storage,
            provider=damage_provider,
        )
        damage_errors = [
            WorkflowStageError(
                image_id=item.image_id,
                code=item.reason_code or (
                    "provider_unavailable" if item.status == "unavailable" else "inference_failed"
                ),
                detail=item.error or item.reason or "Damage inspection did not complete.",
            )
            for item in damage_response.results
            if item.status in {"unavailable", "failed"}
        ]
        stages["damage_detection"] = WorkflowStageResult(
            status=damage_response.status,
            result=damage_response.model_dump(mode="json"),
            errors=damage_errors,
        )
    except Exception as exc:
        db.rollback()
        stages["damage_detection"] = WorkflowStageResult(
            status="unavailable" if isinstance(exc, HTTPException) and exc.status_code == 503 else "failed",
            errors=[_stage_error(exc)],
        )

    try:
        vehicle_input, missing_fields, invalid_fields, price_provenance = _compose_price_input(
            assessment, lookup_details, payload.vehicle, db
        )
    except SQLAlchemyError:
        db.rollback()
        vehicle_input, missing_fields, invalid_fields, price_provenance = None, [], [], {}
        stages["pricing"] = WorkflowStageResult(
            status="failed",
            errors=[WorkflowStageError(
                code="pricing_input_storage_failed",
                detail="Vehicle specifications could not be resolved from stored evidence.",
            )],
        )
    if missing_fields:
        stages["pricing"] = WorkflowStageResult(
            status="unavailable",
            result={"missing_fields": missing_fields, "input_provenance": price_provenance},
            errors=[
                WorkflowStageError(
                    code="missing_vehicle_attributes",
                    detail="Supply the missing vehicle attributes to request a price estimate: "
                    + ", ".join(missing_fields)
                    + ".",
                )
            ],
        )
    elif invalid_fields:
        stages["pricing"] = WorkflowStageResult(
            status="failed",
            result={"invalid_fields": invalid_fields, "input_provenance": price_provenance},
            errors=[WorkflowStageError(
                code="invalid_vehicle_attributes",
                detail="Some stored vehicle attributes are invalid; correct them before pricing: "
                + ", ".join(invalid_fields)
                + ".",
            )],
        )
    elif "pricing" not in stages:
        try:
            price_response = _create_price_estimate(
                assessment_id=assessment_id,
                payload=PriceEstimateRequest(vehicle=vehicle_input),
                db=db,
                provider=price_provider,
                provenance=price_provenance,
            )
            stages["pricing"] = WorkflowStageResult(
                status="available",
                result=price_response.model_dump(mode="json"),
            )
        except Exception as exc:
            db.rollback()
            is_unavailable = isinstance(exc, HTTPException) and exc.status_code == 503
            error_code = "provider_unavailable" if is_unavailable else (
                "price_provider_failed" if isinstance(exc, HTTPException) and exc.status_code == 502 else "price_stage_failed"
            )
            stages["pricing"] = WorkflowStageResult(
                status="unavailable" if is_unavailable else "failed",
                errors=[WorkflowStageError(code=error_code, detail=str(exc.detail) if isinstance(exc, HTTPException) else "Price prediction failed.")],
            )

    try:
        pricing_error = stages["pricing"].errors[0] if stages["pricing"].errors else None
        pricing_state = {
            "status": "available" if stages["pricing"].status == "available" else (
                "skipped" if pricing_error and pricing_error.code == "missing_vehicle_attributes" else
                "unavailable" if stages["pricing"].status == "unavailable" else "failed"
            ),
            "error_code": pricing_error.code if pricing_error else None,
            "missing_fields": missing_fields,
            "invalid_fields": invalid_fields,
            "input_provenance": price_provenance,
        }
        lookup_error = lookup_stage.errors[0] if lookup_stage.errors else None
        lookup_state = {
            "status": lookup_stage.status,
            "error_code": lookup_error.code if lookup_error else None,
            "error_message": lookup_error.detail if lookup_error else None,
        }
        report_response = _generate_assessment_report_snapshot(
            assessment_id=assessment_id,
            db=db,
            market_provider=market_provider,
            pricing_workflow_state=pricing_state,
            vehicle_lookup_workflow_state=lookup_state,
        )
        stages["report"] = WorkflowStageResult(
            status="available",
            result=report_response.model_dump(mode="json"),
        )
        market_result = report_response.report_data.get("market_valuation", {}) if isinstance(report_response.report_data, dict) else {}
        market_refresh = market_result.get("refresh_status") if isinstance(market_result, dict) else None
        market_status = market_result.get("status") if isinstance(market_result, dict) else None
        stages["market_valuation"] = WorkflowStageResult(
            status=("available" if market_status == "available" else "unavailable"),
            result={"status": market_status, "refresh_status": market_refresh,
                    "evidence_count": market_result.get("evidence_count", 0),
                    "market_asking_price_range": market_result.get("market_asking_price_range")},
        )
    except Exception as exc:
        db.rollback()
        stages["report"] = WorkflowStageResult(status="failed", errors=[_stage_error(exc)])

    relevant_stages = {
        key: value
        for key, value in stages.items()
        if key not in {"market_valuation"}
        and (key != "vehicle_details_lookup" or settings.vehicle_lookup_enabled)
    }
    successful_stages = sum(
        stage.status in {
            "available",
            "completed",
            "completed_with_findings",
            "completed_no_findings",
            "no_text",
            "disabled",
            "not_run",
        }
        for stage in relevant_stages.values()
    )
    has_partial_or_failed = any(
        stage.status in {"partial", "unavailable", "failed", "no_text"} for stage in relevant_stages.values()
    )
    overall_status = (
        "failed" if successful_stages == 0 else "partial" if has_partial_or_failed else "completed"
    )
    return AssessmentWorkflowResponse(
        assessment_id=assessment_id,
        status=overall_status,
        stages=stages,
    )
