from datetime import datetime, timezone
from copy import deepcopy
import math
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AssessmentReport,
    DamageFinding,
    OCRResult,
    PriceEstimate,
    UploadedImage,
    VehicleAssessment,
)
from app.db.session import get_db
from app.pricing.market import (
    MarketListingProvider,
    MarketVehicleCriteria,
    calculate_market_comparables,
    get_market_listing_provider,
)
from app.schemas.reports import AssessmentReportResponse


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment reports"])
_DAMAGE_COMPLETION_LIMITATION = (
    "The database stores positive damage findings only; an empty finding list cannot "
    "distinguish a completed scan with no detections from a scan that has not run."
)


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _ocr_confidence(result: OCRResult) -> float | None:
    raw_value = (result.result_data or {}).get("confidence")
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        confidence = float(raw_value)
        if math.isfinite(confidence):
            return confidence
    return None


def _market_inputs(assessment: VehicleAssessment, estimates: list[PriceEstimate]):
    details = assessment.vehicle_details if isinstance(assessment.vehicle_details, dict) else {}
    latest = estimates[-1] if estimates else None
    estimate_details = latest.estimate_data if latest and isinstance(latest.estimate_data, dict) else {}
    vehicle = estimate_details.get("vehicle") if isinstance(estimate_details.get("vehicle"), dict) else {}
    source = {**details, **vehicle}

    asking_price = assessment.asking_price_inr
    ml_prediction = (
        {
            "status": "available",
            "amount_inr": format(latest.amount, "f"),
            "currency": latest.currency,
            "method": latest.method,
            "model_name": latest.model_name,
            "created_at": _iso(latest.created_at),
        }
        if latest is not None
        else {"status": "pending", "amount_inr": None}
    )
    make = source.get("make", source.get("brand"))
    model = source.get("model")
    if not isinstance(make, str) or not make.strip() or not isinstance(model, str) or not model.strip():
        return None, asking_price, ml_prediction
    try:
        criteria = MarketVehicleCriteria(
            make=make,
            model=model,
            variant=source.get("variant"),
            year=source.get("registration_year", source.get("year")),
            fuel_type=source.get("fuel_type"),
            transmission=source.get("transmission", source.get("transmission_type")),
            mileage_km=source.get("odometer_km", source.get("mileage_km")),
            location=source.get("location", source.get("city", source.get("state"))),
        )
    except Exception:
        criteria = None
    return criteria, asking_price, ml_prediction


def _compare_price_to_market(amount, label: str, price_range: dict[str, Any] | None):
    if amount is None or not price_range:
        return {"status": "unavailable", "combined_estimate": None}
    from decimal import Decimal

    compared_amount = Decimal(str(amount))
    lower = Decimal(price_range["lower_inr"])
    upper = Decimal(price_range["upper_inr"])
    if compared_amount < lower:
        return {
            "status": f"{label}_below_market_asking_range",
            "difference_from_range_inr": format(lower - compared_amount, "f"),
            "combined_estimate": None,
        }
    if compared_amount > upper:
        return {
            "status": f"{label}_above_market_asking_range",
            "difference_from_range_inr": format(compared_amount - upper, "f"),
            "combined_estimate": None,
        }
    return {"status": f"{label}_within_market_asking_range", "combined_estimate": None}


def _ml_comparison(ml_prediction: dict[str, Any], price_range: dict[str, Any] | None):
    amount = ml_prediction.get("amount_inr") if ml_prediction.get("status") == "available" else None
    return _compare_price_to_market(amount, "ml", price_range)


def _market_section(
    provider: MarketListingProvider | None,
    criteria: MarketVehicleCriteria | None,
    asking_price,
    ml_prediction: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    if provider is None or criteria is None:
        if previous:
            section = deepcopy(previous)
            section.update(
                seller_asking_price_inr=format(asking_price, "f") if asking_price is not None else None,
                ml_prediction=ml_prediction,
                data_origin="stored_previous",
                refresh_status="provider_unavailable" if provider is None else "insufficient_vehicle_data",
                discrepancy=_ml_comparison(ml_prediction, section.get("market_asking_price_range")),
                seller_asking_price_comparison=_compare_price_to_market(
                    asking_price, "seller_asking", section.get("market_asking_price_range")
                ),
            )
            section["limitations"] = list(section.get("limitations", []))
            section["limitations"].append("A fresh comparable-listing result was unavailable; showing the saved market snapshot.")
            return section
        reason = "no_authorized_listing_provider_configured" if provider is None else "insufficient_vehicle_data_for_matching"
        return {
            "status": "unavailable",
            "valuation_date": now.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "market_asking_price_range": None,
            "evidence_count": 0,
            "minimum_evidence_count": 5,
            "freshness_threshold_days": 30,
            "data_freshness_days": None,
            "maximum_listing_age_days": None,
            "sources": [],
            "comparables": [],
            "rejected_listing_counts": {},
            "data_origin": "unavailable",
            "refresh_status": reason,
            "discrepancy": _ml_comparison(ml_prediction, None),
            "seller_asking_price_comparison": _compare_price_to_market(asking_price, "seller_asking", None),
            "limitations": [
                "No comparable-listing price range is available; no numerical market range was inferred.",
                "The ML estimate and any seller asking price are kept separate from comparable asking prices.",
                "Advertised prices are not verified transaction prices.",
            ],
        }

    try:
        raw_listings = provider.fetch_listings(criteria)
        result = calculate_market_comparables(
            raw_listings, criteria, valuation_date=datetime.now(timezone.utc)
        )
        result_data = result.model_dump(mode="json")
        market_range = result_data.get("price_range")
        sources = sorted({listing.source_name for listing in result.listings})
        return {
            "status": result.status,
            "valuation_date": result.valuation_date.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "market_asking_price_range": market_range,
            "evidence_count": result.evidence_count,
            "minimum_evidence_count": result.minimum_evidence_count,
            "freshness_threshold_days": result.freshness_threshold_days,
            "data_freshness_days": result.data_freshness_days,
            "maximum_listing_age_days": result.maximum_listing_age_days,
            "sources": sources,
            "comparables": [listing.model_dump(mode="json") for listing in result.listings],
            "rejected_listing_counts": result.rejected_counts,
            "data_origin": "freshly_retrieved",
            "refresh_status": "completed",
            "discrepancy": _ml_comparison(ml_prediction, market_range),
            "seller_asking_price_comparison": _compare_price_to_market(asking_price, "seller_asking", market_range),
            "limitations": [
                "Comparable prices are advertised asking prices, not verified transaction prices.",
                "The market range is the interquartile range after 1.5×IQR outlier filtering.",
                "The market range and ML prediction are reported separately and are not averaged.",
            ],
        }
    except Exception:
        if previous:
            section = deepcopy(previous)
            section.update(
                seller_asking_price_inr=format(asking_price, "f") if asking_price is not None else None,
                ml_prediction=ml_prediction,
                data_origin="stored_previous",
                refresh_status="refresh_failed",
                discrepancy=_ml_comparison(ml_prediction, section.get("market_asking_price_range")),
                seller_asking_price_comparison=_compare_price_to_market(
                    asking_price, "seller_asking", section.get("market_asking_price_range")
                ),
            )
            section["limitations"] = list(section.get("limitations", []))
            section["limitations"].append("The latest market refresh failed; showing the saved market snapshot.")
            return section
        return {
            "status": "unavailable",
            "valuation_date": now.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "market_asking_price_range": None,
            "evidence_count": 0,
            "minimum_evidence_count": 5,
            "freshness_threshold_days": 30,
            "data_freshness_days": None,
            "maximum_listing_age_days": None,
            "sources": [],
            "comparables": [],
            "rejected_listing_counts": {},
            "data_origin": "unavailable",
            "refresh_status": "provider_error",
            "discrepancy": _ml_comparison(ml_prediction, None),
            "seller_asking_price_comparison": _compare_price_to_market(asking_price, "seller_asking", None),
            "limitations": [
                "The market provider did not return usable listing evidence; no numerical market range was inferred.",
                "Advertised prices are not verified transaction prices.",
            ],
        }


def _build_report_data(
    assessment: VehicleAssessment,
    images: list[UploadedImage],
    ocr_results: list[OCRResult],
    damage_findings: list[DamageFinding],
    price_estimates: list[PriceEstimate],
    market_valuation: dict[str, Any],
) -> dict[str, Any]:
    image_ids = {image.id for image in images}
    completed_ocr_image_ids = {
        result.image_id
        for result in ocr_results
        if result.status == "completed" and result.image_id is not None and result.image_id in image_ids
    }
    if image_ids:
        completed_ocr_count = len(completed_ocr_image_ids)
        if completed_ocr_count == len(image_ids):
            ocr_status = "available"
        elif completed_ocr_count:
            ocr_status = "partial"
        else:
            ocr_status = "pending"
    elif any(result.status == "completed" for result in ocr_results):
        ocr_status = "available"
    elif ocr_results:
        ocr_status = "pending"
    else:
        ocr_status = "unavailable"

    if damage_findings:
        damage_status = "results_available"
    else:
        damage_status = "pending_or_no_findings" if images else "unavailable"

    report_data = {
        "assessment": {
            "id": str(assessment.id),
            "status": assessment.status,
            "vin": assessment.vin,
            "registration_number": assessment.registration_number,
            "vehicle_details": {
                key: value
                for key, value in (assessment.vehicle_details or {}).items()
                if key != "asking_price_inr"
            },
            "asking_price_inr": (
                format(assessment.asking_price_inr, "f")
                if assessment.asking_price_inr is not None
                else None
            ),
            "created_at": _iso(assessment.created_at),
            "updated_at": _iso(assessment.updated_at),
        },
        "images": [
            {
                "id": str(image.id),
                "original_filename": image.original_filename,
                "content_type": image.content_type,
                "byte_size": image.byte_size,
                "uploaded_at": _iso(image.uploaded_at),
            }
            for image in images
        ],
        "stages": {
            "ocr": {
                "status": ocr_status,
                "result_count": len(ocr_results),
                "completed_image_count": len(completed_ocr_image_ids),
            },
            "damage_detection": {
                "status": damage_status,
                "finding_count": len(damage_findings),
                "images_with_findings": len(
                    {finding.image_id for finding in damage_findings if finding.image_id is not None}
                ),
            },
            "pricing": {
                "status": "available" if price_estimates else "pending",
                "estimate_count": len(price_estimates),
            },
        },
        "ocr_results": [
            {
                "id": str(result.id),
                "image_id": str(result.image_id) if result.image_id else None,
                "result_type": result.result_type,
                "status": result.status,
                "provider": result.provider,
                "text": result.raw_text,
                "confidence": _ocr_confidence(result),
                "created_at": _iso(result.created_at),
            }
            for result in ocr_results
        ],
        "damage_findings": [
            {
                "id": str(finding.id),
                "image_id": str(finding.image_id) if finding.image_id else None,
                "component": finding.component,
                "damage_type": finding.damage_type,
                "severity": finding.severity,
                "confidence": finding.confidence,
                "description": finding.description,
                "bounding_box": finding.bounding_box,
                "model_name": finding.model_name,
                "created_at": _iso(finding.created_at),
            }
            for finding in damage_findings
        ],
        "price_estimates": [
            {
                "id": str(estimate.id),
                "amount": str(estimate.amount),
                "currency": estimate.currency,
                "lower_bound": str(estimate.lower_bound) if estimate.lower_bound is not None else None,
                "upper_bound": str(estimate.upper_bound) if estimate.upper_bound is not None else None,
                "method": estimate.method,
                "model_name": estimate.model_name,
                "created_at": _iso(estimate.created_at),
            }
            for estimate in price_estimates
        ],
        "market_valuation": market_valuation,
        "limitations": [_DAMAGE_COMPLETION_LIMITATION],
    }
    report_data["buyer_recommendation"] = _buyer_recommendation(
        assessment.asking_price_inr,
        price_estimates[-1] if price_estimates else None,
        damage_findings,
        ocr_status,
        market_valuation,
    )
    return report_data


def _buyer_recommendation(
    asking_price,
    estimate: PriceEstimate | None,
    damage_findings: list[DamageFinding],
    ocr_status: str,
    market_valuation: dict[str, Any],
) -> dict[str, Any]:
    """Build deterministic buyer guidance; ML is a comparison reference, not market value."""
    from decimal import Decimal, InvalidOperation

    def positive_decimal(value) -> Decimal | None:
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None
        return parsed if parsed.is_finite() and parsed > 0 else None

    asking = positive_decimal(asking_price)
    predicted = positive_decimal(estimate.amount) if estimate is not None else None
    difference = abs(asking - predicted) if asking is not None and predicted is not None else None
    percent = difference * Decimal(100) / predicted if difference is not None else None
    tolerance = Decimal(settings.buyer_price_near_threshold_percent)
    market_available = (
        market_valuation.get("status") == "available"
        and isinstance(market_valuation.get("market_asking_price_range"), dict)
        and market_valuation.get("evidence_count", 0)
        >= market_valuation.get("minimum_evidence_count", 5)
    )

    if damage_findings:
        recommendation = "further_inspection"
        explanation = (
            f"{len(damage_findings)} stored damage finding(s) are present. Have the vehicle "
            "inspected by a qualified person before purchase; no repair cost or severity "
            "deduction has been inferred."
        )
        next_steps = ["Review the recorded findings and obtain an in-person inspection and repair assessment."]
    elif asking is None or predicted is None:
        recommendation = "insufficient_evidence"
        missing = []
        if asking is None:
            missing.append("a valid seller asking price")
        if predicted is None:
            missing.append("a valid positive ML estimate")
        explanation = "Cannot compare price because " + " and ".join(missing) + " are unavailable."
        next_steps = ["Provide the missing price input and refresh the assessment report."]
    elif percent <= tolerance:
        recommendation = "reasonably_priced"
        explanation = (
            f"The asking price is within the configured {tolerance}% comparison tolerance "
            "of the ML estimate. This is only a model-relative comparison, not confirmation "
            "of fair market value."
        )
        next_steps = ["Verify vehicle condition, ownership and documents before agreeing to buy."]
    else:
        recommendation = "negotiate"
        direction = "above" if asking > predicted else "below"
        explanation = (
            f"The asking price is {percent.quantize(Decimal('0.01'))}% {direction} the ML estimate, "
            f"exceeding the configured {tolerance}% tolerance. The estimate is not verified "
            "market value."
        )
        next_steps = ["Discuss the price with the seller and verify condition and comparable offers."]

    if market_available:
        market_note = "Comparable-market evidence is available separately in market_valuation; it is not merged with the ML estimate."
    else:
        market_note = "Live comparable-market evidence was not available for this report."
    if ocr_status not in {"available", "partial"}:
        evidence_note = "OCR evidence is unavailable or incomplete; registration details may need manual verification."
    else:
        evidence_note = "OCR results are recorded as evidence only and do not validate vehicle identity."

    return {
        "status": recommendation,
        "comparison_basis": "ml_estimate_only",
        "comparison_tolerance_percent": int(tolerance),
        "asking_price_inr": format(asking, "f") if asking is not None else None,
        "ml_estimate_inr": format(predicted, "f") if predicted is not None else None,
        "absolute_difference_inr": format(difference.quantize(Decimal("0.01")), "f") if difference is not None else None,
        "difference_percent_of_ml_estimate": format(percent.quantize(Decimal("0.01")), "f") if percent is not None else None,
        "price_comparison": (
            "unavailable" if difference is None else
            "within_tolerance" if percent <= tolerance else
            "asking_above_estimate" if asking > predicted else "asking_below_estimate"
        ),
        "market_evidence_available": market_available,
        "explanation": explanation,
        "limitations": [
            "The ML estimate is not verified market value and is not a guarantee of sale price.",
            market_note,
            evidence_note,
            "No repair costs or unsupported damage deductions are included.",
        ],
        "next_steps": next_steps,
    }


def _load_stage_records(db: Session, assessment_id: UUID):
    images = list(
        db.scalars(
            select(UploadedImage)
            .where(UploadedImage.assessment_id == assessment_id)
            .order_by(UploadedImage.uploaded_at, UploadedImage.id)
        ).all()
    )
    ocr_results = list(
        db.scalars(
            select(OCRResult)
            .where(OCRResult.assessment_id == assessment_id)
            .order_by(OCRResult.created_at, OCRResult.id)
        ).all()
    )
    damage_findings = list(
        db.scalars(
            select(DamageFinding)
            .where(DamageFinding.assessment_id == assessment_id)
            .order_by(DamageFinding.created_at, DamageFinding.id)
        ).all()
    )
    price_estimates = list(
        db.scalars(
            select(PriceEstimate)
            .where(PriceEstimate.assessment_id == assessment_id)
            .order_by(PriceEstimate.created_at, PriceEstimate.id)
        ).all()
    )
    return images, ocr_results, damage_findings, price_estimates


@router.post(
    "/{assessment_id}/report",
    response_model=AssessmentReportResponse,
    status_code=status.HTTP_200_OK,
)
def generate_assessment_report(
    assessment_id: UUID,
    db: Session = Depends(get_db),
    market_provider: MarketListingProvider | None = Depends(get_market_listing_provider),
) -> AssessmentReportResponse:
    """Create or refresh a draft snapshot from results already stored for an assessment."""
    try:
        preview_assessment = db.get(VehicleAssessment, assessment_id)
        if preview_assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        preview_estimates = list(
            db.scalars(
                select(PriceEstimate)
                .where(PriceEstimate.assessment_id == assessment_id)
                .order_by(PriceEstimate.created_at, PriceEstimate.id)
            ).all()
        )
        preview_report = db.scalar(
            select(AssessmentReport).where(AssessmentReport.assessment_id == assessment_id)
        )
        previous_market = None
        if preview_report and isinstance(preview_report.report_data, dict):
            candidate = preview_report.report_data.get("market_valuation")
            previous_market = deepcopy(candidate) if isinstance(candidate, dict) else None
        criteria, asking_price, ml_prediction = _market_inputs(preview_assessment, preview_estimates)
        db.rollback()
        market_valuation = _market_section(
            market_provider, criteria, asking_price, ml_prediction, previous_market
        )

        with db.begin():
            assessment = db.scalar(
                select(VehicleAssessment)
                .where(VehicleAssessment.id == assessment_id)
                .with_for_update()
            )
            if assessment is None:
                raise HTTPException(status_code=404, detail="Assessment not found.")

            images, ocr_results, damage_findings, price_estimates = _load_stage_records(db, assessment_id)
            report = db.scalar(
                select(AssessmentReport)
                .where(AssessmentReport.assessment_id == assessment_id)
                .with_for_update()
            )
            report_data = _build_report_data(
                assessment, images, ocr_results, damage_findings, price_estimates, market_valuation
            )
            if report is None:
                report = AssessmentReport(assessment_id=assessment_id)
                db.add(report)
            report.status = "draft"
            report.summary = "Draft snapshot of currently stored assessment results."
            report.report_data = report_data
            report.generated_at = datetime.now(timezone.utc)
            db.flush()
    except HTTPException:
        raise
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to generate assessment report.") from None

    try:
        db.refresh(report)
        return AssessmentReportResponse.model_validate(report)
    except SQLAlchemyError:
        db.rollback()
        raise HTTPException(status_code=500, detail="Unable to retrieve generated assessment report.") from None


@router.get(
    "/{assessment_id}/report",
    response_model=AssessmentReportResponse,
)
def get_assessment_report(
    assessment_id: UUID,
    db: Session = Depends(get_db),
) -> AssessmentReportResponse:
    try:
        assessment = db.get(VehicleAssessment, assessment_id)
        if assessment is None:
            raise HTTPException(status_code=404, detail="Assessment not found.")
        report = db.scalar(
            select(AssessmentReport).where(AssessmentReport.assessment_id == assessment_id)
        )
    except HTTPException:
        raise
    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Unable to retrieve assessment report.") from None
    if report is None:
        raise HTTPException(status_code=404, detail="Assessment report not found.")
    return AssessmentReportResponse.model_validate(report)
