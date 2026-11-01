from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import OCRResult, PriceEstimate, VehicleAssessment
from app.db.session import get_db
from app.pricing.provider import PricePredictionProvider, get_price_provider
from app.schemas.pricing import (
    PriceEstimateRequest,
    PriceEstimateResponse,
    PriceEstimateSubmission,
    PricePrediction,
)
from app.schemas.vehicle_lookup import VehicleRegistrationDetails


router = APIRouter(prefix="/api/v1/assessments", tags=["price estimates"])
_MANUAL_SPECIFICATION_FIELDS = ("transmission", "mileage_kmpl", "max_power_bhp")


def _persist_manual_pricing_specifications(assessment, user_details) -> bool:
    if user_details is None:
        return False
    supplied = user_details.model_dump(mode="json", exclude_unset=True)
    updates = {
        field: supplied[field]
        for field in _MANUAL_SPECIFICATION_FIELDS
        if supplied.get(field) is not None
    }
    if not updates:
        return False
    existing_details = assessment.vehicle_details
    details = dict(existing_details) if isinstance(existing_details, dict) else {}
    details.update(updates)
    assessment.vehicle_details = details
    return True


def _save_price_estimate(
    assessment_id: UUID,
    payload: PriceEstimateRequest,
    db: Session,
    provider: PricePredictionProvider | None,
    provenance: dict[str, str] | None = None,
) -> PriceEstimateResponse:
    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Price prediction is not configured. No estimate was created.",
            headers={"Retry-After": "3600"},
        )

    try:
        prediction = PricePrediction.model_validate(provider.predict(payload.vehicle))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The price provider did not return a valid estimate.",
        ) from None

    estimate = PriceEstimate(
        assessment_id=assessment_id,
        amount=prediction.amount,
        currency=prediction.currency,
        lower_bound=prediction.lower_bound,
        upper_bound=prediction.upper_bound,
        method=prediction.method,
        model_name=prediction.model_name,
        estimate_data={"vehicle": payload.vehicle.model_dump(mode="json"), "provenance": provenance or {}},
    )
    try:
        db.add(estimate)
        db.commit()
        db.refresh(estimate)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save price estimate.") from None

    estimate_data = estimate.estimate_data if isinstance(estimate.estimate_data, dict) else {}
    return PriceEstimateResponse(
        id=estimate.id,
        assessment_id=estimate.assessment_id,
        amount=estimate.amount,
        currency=estimate.currency,
        lower_bound=estimate.lower_bound,
        upper_bound=estimate.upper_bound,
        method=estimate.method,
        model_name=estimate.model_name,
        created_at=estimate.created_at,
        vehicle=estimate_data.get("vehicle", {}),
        provenance=estimate_data.get("provenance", {}),
    )


def _create_price_estimate(
    assessment_id: UUID,
    payload: PriceEstimateRequest,
    db: Session,
    provider: PricePredictionProvider | None,
    provenance: dict[str, str] | None = None,
) -> PriceEstimateResponse:
    """Internal helper used by the workflow. Validates assessment and provider, then saves."""
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment.") from None
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")

    if provider is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Price prediction is not configured. No estimate was created.",
            headers={"Retry-After": "3600"},
        )

    return _save_price_estimate(assessment_id, payload, db, provider, provenance)


def _latest_saved_vehicle_lookup(
    assessment_id: UUID,
    db: Session,
) -> VehicleRegistrationDetails | None:
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
        if not isinstance(row.result_data, dict):
            continue
        try:
            return VehicleRegistrationDetails.model_validate(row.result_data)
        except ValidationError:
            continue
    return None


@router.post(
    "/{assessment_id}/price-estimates",
    response_model=PriceEstimateResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Assessment not found."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Price prediction is not configured; no estimate was created."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "The configured provider returned no valid estimate."},
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Pricing is unavailable when required, sourced model features are missing."
        },
    },
)
def create_price_estimate(
    assessment_id: UUID,
    payload: PriceEstimateSubmission,
    db: Session = Depends(get_db),
    provider: PricePredictionProvider | None = Depends(get_price_provider),
) -> PriceEstimateResponse:
    """Create an estimate from stored API facts and permitted manual inputs."""
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Price prediction is not configured. No estimate was created.",
                headers={"Retry-After": "3600"},
            )
        should_commit_assessment = _persist_manual_pricing_specifications(
            assessment, payload.vehicle
        )
        if payload.vehicle is not None and payload.vehicle.seller_asking_price is not None:
            assessment.asking_price_inr = payload.vehicle.seller_asking_price
            should_commit_assessment = True
        if should_commit_assessment:
            db.commit()
            db.refresh(assessment)
        lookup_details = _latest_saved_vehicle_lookup(assessment_id, db)
        from app.api.v1.workflow import _compose_price_input

        vehicle, missing, invalid, provenance = _compose_price_input(
            assessment,
            lookup_details,
            payload.vehicle,
            db,
        )
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to resolve pricing inputs.") from None

    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "status": "unavailable",
                "missing_features": missing,
                "message": "Required vehicle specifications were not returned by the registration API "
                "or a verified model-specific source.",
            },
        )
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "status": "failed",
                "invalid_features": invalid,
                "message": "Resolved pricing inputs contain invalid values.",
            },
        )
    if vehicle is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"status": "unavailable", "message": "Required pricing inputs are unavailable."},
        )
    return _create_price_estimate(
        assessment_id,
        PriceEstimateRequest(vehicle=vehicle),
        db,
        provider,
        provenance,
    )
