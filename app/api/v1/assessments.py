from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import VehicleAssessment
from app.db.session import get_db
from app.schemas.assessments import (
    AssessmentCreate,
    AssessmentLimit,
    AssessmentListResponse,
    AssessmentOffset,
    AssessmentPatch,
    AssessmentResponse,
)


router = APIRouter(prefix="/api/v1/assessments", tags=["assessments"])


def _database_error(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


def _vehicle_details_without_asking_price(details: dict | None) -> dict | None:
    if details is None:
        return None
    sanitized = dict(details)
    sanitized.pop("asking_price_inr", None)
    return sanitized


@router.post("", response_model=AssessmentResponse, status_code=status.HTTP_201_CREATED)
def create_assessment(payload: AssessmentCreate, db: Session = Depends(get_db)) -> AssessmentResponse:
    assessment = VehicleAssessment(
        vin=payload.vin,
        registration_number=payload.registration_number,
        vehicle_details=_vehicle_details_without_asking_price(payload.vehicle_details),
    )
    if payload.asking_price_inr is not None:
        assessment.asking_price_inr = payload.asking_price_inr
    try:
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Assessment contains invalid data.",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        raise _database_error("Unable to create assessment.") from None
    return AssessmentResponse.model_validate(assessment)


@router.get("", response_model=AssessmentListResponse)
def list_assessments(
    limit: AssessmentLimit = Query(default=20),
    offset: AssessmentOffset = Query(default=0),
    db: Session = Depends(get_db),
) -> AssessmentListResponse:
    try:
        total = db.scalar(select(func.count()).select_from(VehicleAssessment)) or 0
        items = db.scalars(
            select(VehicleAssessment)
            .order_by(VehicleAssessment.created_at.desc(), VehicleAssessment.id.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    except SQLAlchemyError:
        raise _database_error("Unable to list assessments.") from None
    return AssessmentListResponse(items=items, total=total, limit=limit, offset=offset)


@router.get("/{assessment_id}", response_model=AssessmentResponse)
def get_assessment(assessment_id: UUID, db: Session = Depends(get_db)) -> AssessmentResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
    except SQLAlchemyError:
        raise _database_error("Unable to retrieve assessment.") from None
    if assessment is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")
    return AssessmentResponse.model_validate(assessment)


@router.patch("/{assessment_id}", response_model=AssessmentResponse)
def update_assessment(
    assessment_id: UUID,
    payload: AssessmentPatch,
    db: Session = Depends(get_db),
) -> AssessmentResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")

        updates = payload.model_dump(exclude_unset=True)
        asking_price_was_set = "asking_price_inr" in updates
        previous_asking_price = assessment.asking_price_inr
        for field, value in updates.items():
            if field == "asking_price_inr":
                continue
            if field == "vehicle_details":
                value = _vehicle_details_without_asking_price(value)
                if not asking_price_was_set and previous_asking_price is not None:
                    value = dict(value or {})
                    value["asking_price_inr"] = format(previous_asking_price, "f")
            setattr(assessment, field, value.value if field == "status" else value)
        if asking_price_was_set:
            assessment.asking_price_inr = updates["asking_price_inr"]

        db.commit()
        db.refresh(assessment)
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Assessment contains invalid data.",
        ) from None
    except SQLAlchemyError:
        db.rollback()
        raise _database_error("Unable to update assessment.") from None
    return AssessmentResponse.model_validate(assessment)
