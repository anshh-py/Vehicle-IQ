from uuid import UUID

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
    get_vehicle_registration_provider,
    lookup_registration_from_manual,
    lookup_registration_from_ocr,
    normalize_manual_registration,
)
from app.core.config import settings
from app.db.models import AssessmentReport, OCRResult, VehicleAssessment
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
from app.schemas.workflow import (
    AssessmentWorkflowRequest,
    AssessmentWorkflowResponse,
    WorkflowStageError,
    WorkflowStageResult,
)
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

            details_source = {key: value for key, value in source.items() if key != "source"}
            return VehicleRegistrationDetails.model_validate(details_source), row.provider
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
        "odometer_km",
        "seller_type",
        "transmission",
        "mileage_kmpl",
        "max_power_bhp",
    ):
        value = stored.get(key)
        if value is not None:
            merged[key] = value
            provenance[key] = "user_input"

    # Explicit manual inputs override API values, and are persisted for retries.
    if user_details is not None:
        user_dump = user_details.model_dump(exclude_unset=True)
        for key in (
            "odometer_km",
            "seller_type",
            "transmission",
            "mileage_kmpl",
            "max_power_bhp",
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

    manually_correctable = ("transmission", "mileage_kmpl", "max_power_bhp")
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
    if all(error.code in {"provider_unavailable", "provider_timeout"} for error in errors):
        return "unavailable"
    return "failed"


@router.post("/{assessment_id}/workflow", response_model=AssessmentWorkflowResponse)
def run_assessment_workflow(
    assessment_id: UUID,
    payload: AssessmentWorkflowRequest,
    db: Session = Depends(get_db),
    ocr_client=Depends(get_ocr_client),
    damage_provider=Depends(get_damage_provider),
    vehicle_lookup_provider=Depends(get_vehicle_registration_provider),
    price_provider=Depends(get_price_provider),
    market_provider=Depends(get_market_listing_provider),
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
            for field in ("transmission", "mileage_kmpl", "max_power_bhp")
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
            # Reconstruct all stages including vehicle_details_lookup and report
            # from the stored report data.
            reconstructed_stages = dict(stored_stages)
            # Add vehicle_details_lookup from report_data
            vehicle_lookup = existing_report.report_data.get("vehicle_details_lookup")
            if vehicle_lookup is not None:
                reconstructed_stages["vehicle_details_lookup"] = vehicle_lookup
            # Add report stage from the stored report
            reconstructed_stages["report"] = {
                "status": "available",
                "result": {
                    "id": str(existing_report.id),
                    "assessment_id": str(existing_report.assessment_id),
                    "status": existing_report.status,
                    "summary": existing_report.summary,
                    "report_data": existing_report.report_data,
                    "generated_at": existing_report.generated_at.isoformat() if existing_report.generated_at else None,
                    "created_at": existing_report.created_at.isoformat() if existing_report.created_at else None,
                },
            }
            # Compute workflow status from all stages
            relevant_stages = {
                key: value for key, value in reconstructed_stages.items()
                if key != "vehicle_details_lookup" or settings.vehicle_lookup_enabled
            }
            successful_stages = sum(
                (value.get("status") if isinstance(value, dict) else "")
                in {
                    "available",
                    "completed_with_findings",
                    "completed_no_findings",
                }
                for value in relevant_stages.values()
            )
            has_partial_or_failed = any(
                (value.get("status") if isinstance(value, dict) else "")
                in {"partial", "unavailable", "failed"}
                for value in relevant_stages.values()
            )
            overall_status = (
                "failed" if successful_stages == 0 else "partial" if has_partial_or_failed else "completed"
            )
            return AssessmentWorkflowResponse(
                assessment_id=assessment_id,
                status=overall_status,
                stages=reconstructed_stages,
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
                    client=ocr_client,
                )
                ocr_results.append(result.model_dump(mode="json"))
            except Exception as exc:
                db.rollback()
                ocr_errors.append(_stage_error(exc, image_id=image_id))
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

    try:
        damage_response = inspect_assessment_damage(
            assessment_id=assessment_id,
            payload=AssessmentDamageRequest(image_ids=payload.image_ids),
            db=db,
            provider=damage_provider,
        )
        damage_errors = [
            WorkflowStageError(
                image_id=item.image_id,
                code=(
                    "provider_unavailable"
                    if item.status == "unavailable"
                    else "inference_failed"
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
        decision = (report_response.report_data or {}).get("pricing_decision", {})
        if (
            stages.get("pricing") is not None
            and stages["pricing"].status == "available"
            and isinstance(decision, dict)
            and decision.get("pricing_eligible") is not True
        ):
            # The model prediction remains recorded in the report's explicitly
            # audit-only section. Do not expose it as the workflow's resale
            # estimate when the deterministic condition policy withholds it.
            pricing_result = dict(stages["pricing"].result or {})
            pricing_result.update({
                "amount": None,
                "lower_bound": None,
                "upper_bound": None,
                "valuation_status": decision.get("status"),
                "withholding_reason": decision.get("withholding_reason"),
                "withholding_message": decision.get("withholding_message"),
                "customer_facing_estimate_inr": None,
            })
            stages["pricing"] = WorkflowStageResult(
                status="available",
                result=pricing_result,
            )
        stages["report"] = WorkflowStageResult(
            status="available",
            result=report_response.model_dump(mode="json"),
        )
    except Exception as exc:
        db.rollback()
        stages["report"] = WorkflowStageResult(status="failed", errors=[_stage_error(exc)])

    relevant_stages = {key: value for key, value in stages.items() if key != "vehicle_details_lookup" or settings.vehicle_lookup_enabled}
    successful_stages = sum(
        stage.status in {
            "available",
            "completed_with_findings",
            "completed_no_findings",
        }
        for stage in relevant_stages.values()
    )
    has_partial_or_failed = any(
        stage.status in {"partial", "unavailable", "failed"} for stage in relevant_stages.values()
    )
    overall_status = (
        "failed" if successful_stages == 0 else "partial" if has_partial_or_failed else "completed"
    )
    return AssessmentWorkflowResponse(
        assessment_id=assessment_id,
        status=overall_status,
        stages=stages,
    )
