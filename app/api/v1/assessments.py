import json
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.observability import request_id_context
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
logger = logging.getLogger("app.assessments")


def _database_error(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


def _log_database_error(operation: str, exc: SQLAlchemyError) -> None:
    """Log useful database diagnostics without forwarding driver-provided details."""
    original = getattr(exc, "orig", None)
    diagnostic = getattr(original, "diag", None)
    sqlstate = (
        getattr(original, "sqlstate", None)
        or getattr(original, "pgcode", None)
        or getattr(diagnostic, "sqlstate", None)
    )
    if not isinstance(sqlstate, str) or len(sqlstate) != 5 or not sqlstate.isascii() or not sqlstate.isalnum():
        sqlstate = None
    raw_message = str(original) if original is not None else str(exc)
    error_category, safe_message = _safe_database_message(raw_message)
    logger.error(
        json.dumps(
            {
                "event": "assessment_database_error",
                "operation": operation,
                "request_id": request_id_context.get(),
                "exception_class": type(exc).__name__,
                "underlying_exception_class": type(original).__name__ if original is not None else None,
                "sqlstate": sqlstate,
                "error_category": error_category,
                "error_message": safe_message,
            },
            separators=(",", ":"),
        )
    )


def _safe_database_message(message: str) -> tuple[str, str]:
    """Map common PostgreSQL connection failures to fixed, non-sensitive diagnostics."""
    normalized = message.casefold()
    categories = (
        (
            "authentication",
            ("password authentication failed", "authentication failed", "no password supplied", "invalid password"),
            "Database authentication failed.",
        ),
        (
            "dns",
            ("could not translate host name", "name or service not known", "name resolution", "getaddrinfo", "nodename nor servname"),
            "Database host name could not be resolved.",
        ),
        ("connection_refused", ("connection refused",), "Database connection was refused."),
        (
            "timeout",
            ("timed out", "timeout expired", "operation timed out"),
            "Database connection timed out.",
        ),
        ("ssl_tls", ("ssl", "tls", "certificate verify failed"), "Database TLS/SSL negotiation failed."),
        (
            "network",
            ("network is unreachable", "no route to host", "connection reset", "server closed the connection"),
            "Database network connection failed.",
        ),
        (
            "capacity",
            ("too many connections", "remaining connection slots", "connection limit"),
            "Database connection capacity was exceeded.",
        ),
    )
    for category, indicators, safe_message in categories:
        if any(indicator in normalized for indicator in indicators):
            return category, safe_message
    return "other", "Database operation failed; driver details were omitted."


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
    except SQLAlchemyError as exc:
        db.rollback()
        _log_database_error("create_assessment", exc)
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
    except SQLAlchemyError as exc:
        _log_database_error("list_assessments", exc)
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
