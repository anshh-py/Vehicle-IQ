from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import PriceEstimate, VehicleAssessment
from app.db.session import get_db
from app.pricing.provider import PricePredictionProvider, get_price_provider
from app.schemas.pricing import PriceEstimateRequest, PriceEstimateResponse, PricePrediction


router = APIRouter(prefix="/api/v1/assessments", tags=["price estimates"])


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
    },
)
def create_price_estimate(
    assessment_id: UUID,
    payload: PriceEstimateRequest,
    db: Session = Depends(get_db),
    provider: PricePredictionProvider | None = Depends(get_price_provider),
) -> PriceEstimateResponse:
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
        estimate_data={"vehicle": payload.vehicle.model_dump(mode="json")},
    )
    try:
        db.add(estimate)
        db.commit()
        db.refresh(estimate)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to save price estimate.") from None

    return PriceEstimateResponse.model_validate(estimate)
