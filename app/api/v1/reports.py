from datetime import datetime, timezone
from copy import deepcopy
import json
import logging
import math
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import (
    AssessmentReport,
    DamageFinding,
    EvidenceRecord,
    OCRResult,
    PriceEstimate,
    UploadedImage,
    VehicleAssessment,
)
from app.db.session import get_db
from app.ocr.registration import build_registration_ocr_assessment
from app.pricing.market import (
    MIN_COMPARABLE_COUNT,
    MarketListingProvider,
    MarketVehicleCriteria,
    calculate_market_comparables,
    get_market_listing_provider,
)
from app.llm.report import (
    damage_condition_summary,
    damage_narrative_explanation,
    generate_llm_narrative,
    report_data_hash,
)
from app.schemas.reports import AssessmentReportResponse


router = APIRouter(prefix="/api/v1/assessments", tags=["assessment reports"])
logger = logging.getLogger(__name__)
_DAMAGE_COMPLETION_LIMITATION = (
    "Legacy damage records may have no scan-completion evidence; an empty finding list "
    "for those records cannot distinguish a completed scan from a scan that has not run."
)
_PRICING_PROVENANCE_LABELS = {
    "vehicle_api",
    "model_default",
    "generic_fallback",
    "user_input",
}
_USER_PRICING_FIELDS = {"odometer_km", "seller_type"}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _valid_market_source_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urlsplit(value)
        return (
            parts.scheme.casefold() in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
        )
    except ValueError:
        return False


def _valid_market_price(value: Any) -> bool:
    if not isinstance(value, (str, int, float, Decimal)) or isinstance(value, bool):
        return False
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return False
    return amount.is_finite() and amount > 0


def _ocr_confidence(result: OCRResult) -> float | None:
    raw_value = (result.result_data or {}).get("confidence")
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        confidence = float(raw_value)
        if math.isfinite(confidence):
            return confidence
    return None


def _latest_vehicle_service(results: list[OCRResult], result_type: str, allowed_fields: set[str]):
    matches = [result for result in results if result.result_type == result_type]
    if not matches:
        return {"status": "unavailable", "provider": None, "details": None, "error_code": "not_run"}
    latest = matches[-1]
    data = latest.result_data if isinstance(latest.result_data, dict) else {}
    details = data.get("details") if isinstance(data.get("details"), dict) else None
    safe_details = (
        {key: value for key, value in details.items() if key in allowed_fields}
        if details is not None
        else None
    )
    state = data.get("state")
    if state not in {"available", "unavailable", "partial", "failed"}:
        state = latest.status if latest.status in {"unavailable", "partial", "failed"} else "failed"
    return {
        "status": state,
        "provider": latest.provider,
        "details": safe_details,
        "error_code": data.get("error_code") if isinstance(data.get("error_code"), str) else None,
    }


def _vehicle_details_lookup(results: list[OCRResult]) -> dict[str, Any]:
    matches = [result for result in results if result.result_type == "registration_lookup"]
    if not matches:
        return {
            "status": "unavailable",
            "result_count": 0,
            "details": None,
            "field_provenance": {},
        }
    latest = matches[-1]
    source = latest.result_data if isinstance(latest.result_data, dict) else {}
    allowed_fields = {
        "registration_number",
        "make",
        "model",
        "variant",
        "fuel_type",
        "registration_date",
        "manufacturing_year",
        "vehicle_class",
        "registration_valid_until",
        "rto",
        "fitness_expiry",
        "puc_expiry",
        "insurance_expiry",
        "blacklisted",
        "engine_cc",
        "seats",
        "max_power_bhp",
        "transmission",
        "owner_count",
        "mileage_kmpl",
    }
    details = {key: value for key, value in source.items() if key in allowed_fields}
    registration_date = details.get("registration_date")
    if isinstance(registration_date, str) and len(registration_date) >= 4:
        details["registration_year"] = registration_date[:4]
    if latest.status != "completed":
        result_status = "failed"
    elif details.get("make") and details.get("model"):
        result_status = "available"
    else:
        result_status = "partial"
    return {
        "status": result_status,
        "result_count": len(matches),
        "provider": latest.provider,
        "source": source.get("source") or ("ocr" if latest.image_id is not None else None),
        "details": details or None,
        "field_provenance": {
            key: "vehicle_api"
            for key, value in details.items()
            if value is not None
        },
    }


def _market_inputs(
    assessment: VehicleAssessment,
    estimates: list[PriceEstimate],
    lookup_results: list[OCRResult] | None = None,
):
    details = assessment.vehicle_details if isinstance(assessment.vehicle_details, dict) else {}
    lookup_details = (
        _vehicle_details_lookup(lookup_results)["details"]
        if lookup_results
        else {}
    )
    latest = estimates[-1] if estimates else None
    estimate_details = latest.estimate_data if latest and isinstance(latest.estimate_data, dict) else {}
    vehicle = estimate_details.get("vehicle") if isinstance(estimate_details.get("vehicle"), dict) else {}
    source = {**lookup_details, **details, **vehicle}

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
    location_source = {**vehicle, **lookup_details, **details}
    location = location_source.get("location")
    if not isinstance(location, str) or not location.strip():
        location_parts = [
            value.strip()
            for value in (location_source.get("city"), location_source.get("state"))
            if isinstance(value, str) and value.strip()
        ]
        location = ", ".join(dict.fromkeys(location_parts)) or None
    elif isinstance(location_source.get("state"), str) and location_source["state"].strip():
        state = location_source["state"].strip()
        if state.casefold() not in location.casefold():
            location = f"{location.strip()}, {state}"
    try:
        criteria = MarketVehicleCriteria(
            make=make,
            model=model,
            variant=source.get("variant"),
            year=source.get("registration_year", source.get("year")),
            fuel_type=source.get("fuel_type"),
            transmission=source.get("transmission", source.get("transmission_type")),
            mileage_km=source.get("odometer_km", source.get("mileage_km")),
            location=location,
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
    if provider is None:
        reason = "Tavily web search is not configured. Set TAVILY_API_KEY in the backend environment."
        return {
            "status": "unavailable",
            "unavailable_reason": reason,
            "valuation_date": now.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "vehicle_researched": (
                {"make": criteria.make, "model": criteria.model} if criteria else None
            ),
            "search_location": criteria.location if criteria else None,
            "geographic_scope": [],
            "relevant_result_count": 0,
            "priced_comparable_count": 0,
            "exact_vehicle_match_count": 0,
            "different_variant_count": 0,
            "different_year_count": 0,
            "general_market_information_count": 0,
            "priced_comparables": [],
            "market_research_results": [],
            "market_asking_price_range": None,
            "evidence_count": 0,
            "exact_match_count": 0,
            "close_match_count": 0,
            "approximate_match_count": 0,
            "minimum_evidence_count": MIN_COMPARABLE_COUNT,
            "freshness_threshold_days": 30,
            "data_freshness_days": None,
            "maximum_listing_age_days": None,
            "sources": [],
            "comparables": [],
            "rejected_listing_counts": {},
            "market_research_rejection_counts": {},
            "search_observability": {
                "queries_issued": 0,
                "raw_results_returned": 0,
                "listings_extracted": 0,
                "extraction_rejected": 0,
                "relevance_rejected": 0,
                "market_research_rejected": 0,
                "duplicates_removed": 0,
                "outliers_removed": 0,
                "final_exact_count": 0,
                "final_close_count": 0,
                "final_approximate_count": 0,
            },
            "data_origin": "unavailable",
            "refresh_status": "provider_not_configured",
            "discrepancy": _ml_comparison(ml_prediction, None),
            "seller_asking_price_comparison": _compare_price_to_market(
                asking_price, "seller_asking", None
            ),
            "limitations": [reason],
        }

    if criteria is None:
        return {
            "status": "unavailable",
            "valuation_date": now.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "vehicle_researched": None,
            "search_location": None,
            "geographic_scope": [],
            "relevant_result_count": 0,
            "priced_comparable_count": 0,
            "exact_vehicle_match_count": 0,
            "different_variant_count": 0,
            "different_year_count": 0,
            "general_market_information_count": 0,
            "priced_comparables": [],
            "market_research_results": [],
            "market_asking_price_range": None,
            "evidence_count": 0,
            "exact_match_count": 0,
            "close_match_count": 0,
            "approximate_match_count": 0,
            "minimum_evidence_count": MIN_COMPARABLE_COUNT,
            "freshness_threshold_days": 30,
            "data_freshness_days": None,
            "maximum_listing_age_days": None,
            "sources": [],
            "comparables": [],
            "rejected_listing_counts": {},
            "market_research_rejection_counts": {},
            "search_observability": {
                "queries_issued": 0,
                "raw_results_returned": 0,
                "listings_extracted": 0,
                "extraction_rejected": 0,
                "relevance_rejected": 0,
                "market_research_rejected": 0,
                "duplicates_removed": 0,
                "outliers_removed": 0,
                "final_exact_count": 0,
                "final_close_count": 0,
                "final_approximate_count": 0,
            },
            "data_origin": "unavailable",
            "refresh_status": "insufficient_vehicle_details",
            "unavailable_reason": "Vehicle make and model details are required to find comparable listings.",
            "discrepancy": _ml_comparison(ml_prediction, None),
            "seller_asking_price_comparison": _compare_price_to_market(
                asking_price, "seller_asking", None
            ),
            "limitations": [
                "Insufficient vehicle details to generate market estimates.",
            ],
        }

    try:
        raw_listings = provider.fetch_listings(criteria)
        result = calculate_market_comparables(
            raw_listings, criteria, valuation_date=datetime.now(timezone.utc)
        )
        result_data = result.model_dump(mode="json")
        market_range = result_data.get("price_range")
        provider_rejections = getattr(provider, "rejected_listing_counts", {})
        rejected_counts = dict(result.rejected_counts)
        if isinstance(provider_rejections, dict):
            for reason, count in provider_rejections.items():
                if isinstance(reason, str) and isinstance(count, int) and count > 0:
                    rejected_counts[reason] = rejected_counts.get(reason, 0) + count
        exact_count = sum(
            listing.match_type == "exact_match" for listing in result.listings
        )
        close_count = sum(
            listing.match_type == "close_match" for listing in result.listings
        )
        approximate_count = sum(
            listing.match_type == "approximate_match" for listing in result.listings
        )
        raw_research_results = getattr(provider, "market_research_results", [])
        if not isinstance(raw_research_results, list):
            raw_research_results = []
        provider_research_rejections = getattr(
            provider, "market_research_rejection_counts", {}
        )
        if not isinstance(provider_research_rejections, dict):
            provider_research_rejections = {}
        market_research_rejection_counts = {
            reason: count
            for reason, count in provider_research_rejections.items()
            if isinstance(reason, str)
            and isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
        }
        market_research_results = [
            {
                "source_url": item["source_url"],
                "asking_price_inr": format(Decimal(str(item["asking_price_inr"])), "f"),
                "variant": (
                    item["variant"].strip()
                    if isinstance(item.get("variant"), str) and item["variant"].strip()
                    else None
                ),
                "year": (
                    item["year"]
                    if isinstance(item.get("year"), int)
                    and not isinstance(item.get("year"), bool)
                    else None
                ),
            }
            for item in raw_research_results
            if isinstance(item, dict)
            and item.get("priced_used_listing") is True
            and _valid_market_price(item.get("asking_price_inr"))
            and _valid_market_source_url(item.get("source_url"))
        ]
        if not market_research_results:
            market_research_results = [
                {
                    "source_url": str(listing.source_url),
                    "variant": listing.variant,
                    "year": listing.year,
                    "asking_price_inr": format(listing.advertised_price_inr, "f"),
                }
                for listing in result.listings
            ]
        priced_comparables = market_research_results
        sources = sorted({listing.source_name for listing in result.listings})
        raw_search_scopes = getattr(provider, "search_scopes", [])
        if not isinstance(raw_search_scopes, list):
            raw_search_scopes = []
        search_scopes = list(
            dict.fromkeys(
                scope.strip()
                for scope in raw_search_scopes
                if isinstance(scope, str) and scope.strip()
            )
        )
        if not search_scopes:
            search_scopes = [
                f"{criteria.location} and nearby regions" if criteria.location else "India"
            ]
        provider_observability = getattr(provider, "search_observability", {})
        if not isinstance(provider_observability, dict):
            provider_observability = {}
        relevance_rejected = sum(
            count
            for reason, count in result.rejected_counts.items()
            if reason not in {"duplicate_listing", "price_outlier"}
        )
        search_observability = {
            "queries_issued": provider_observability.get("queries_issued", 0),
            "raw_results_returned": provider_observability.get("raw_results_returned", 0),
            "listings_extracted": provider_observability.get("listings_extracted", 0),
            "extraction_rejected": provider_observability.get("extraction_rejected", 0),
            "relevance_rejected": provider_observability.get("relevance_rejected", 0)
            + relevance_rejected,
            "market_research_rejected": provider_observability.get(
                "market_research_rejected", 0
            ),
            "duplicates_removed": provider_observability.get("duplicates_removed", 0)
            + result.rejected_counts.get("duplicate_listing", 0),
            "outliers_removed": provider_observability.get("outliers_removed", 0)
            + result.rejected_counts.get("price_outlier", 0),
            "final_exact_count": exact_count,
            "final_close_count": close_count,
            "final_approximate_count": approximate_count,
        }
        search_observability = {
            key: value if isinstance(value, int) and not isinstance(value, bool) else 0
            for key, value in search_observability.items()
        }
        unavailable_reason = None
        if result.status == "unavailable":
            if market_research_results:
                unavailable_reason = (
                    "Relevant market research was found, but no reliable priced comparables "
                    "met the range criteria. No market price range was calculated."
                )
            else:
                unavailable_reason = "The provider returned no valid comparable listings."
        elif result.status == "low_evidence":
            unavailable_reason = (
                f"Only {result.evidence_count} same-fuel, year-compatible comparable "
                "listings were found; "
                f"at least {result.minimum_evidence_count} are required."
            )
        return {
            "status": result.status,
            "unavailable_reason": unavailable_reason,
            "valuation_date": result.valuation_date.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "vehicle_researched": {"make": criteria.make, "model": criteria.model},
            "search_location": criteria.location,
            "geographic_scope": search_scopes,
            "relevant_result_count": len(market_research_results),
            "priced_comparable_count": len(priced_comparables),
            "exact_vehicle_match_count": 0,
            "different_variant_count": 0,
            "different_year_count": 0,
            "general_market_information_count": 0,
            "priced_comparables": priced_comparables,
            "market_research_results": market_research_results,
            "market_asking_price_range": market_range,
            "evidence_count": result.evidence_count,
            "exact_match_count": exact_count,
            "close_match_count": close_count,
            "approximate_match_count": approximate_count,
            "minimum_evidence_count": result.minimum_evidence_count,
            "freshness_threshold_days": result.freshness_threshold_days,
            "data_freshness_days": result.data_freshness_days,
            "maximum_listing_age_days": result.maximum_listing_age_days,
            "sources": sources,
            "comparables": [listing.model_dump(mode="json") for listing in result.listings],
            "rejected_listing_counts": rejected_counts,
            "market_research_rejection_counts": market_research_rejection_counts,
            "search_observability": search_observability,
            "data_origin": "freshly_retrieved",
            "refresh_status": "completed",
            "discrepancy": _ml_comparison(ml_prediction, market_range),
            "seller_asking_price_comparison": _compare_price_to_market(asking_price, "seller_asking", market_range),
            "limitations": [
                "Web-search results are discovered advertisements, not independently verified transactions or confirmation that a listing remains active.",
                "The market range is the interquartile range after 1.5×IQR outlier filtering.",
                "The market range and ML prediction are reported separately and are not averaged.",
                *(
                    [
                        "A price range is shown only when the existing minimum evidence and comparable-quality checks are met; individual discovered prices remain separately labelled."
                    ]
                    if priced_comparables and market_range is None
                    else []
                ),
                *(
                    [
                        "Approximate matches were included because exact and close matches "
                        "did not meet the evidence minimum; see each listing's differences."
                    ]
                    if approximate_count
                    else []
                ),
                *([unavailable_reason] if unavailable_reason else []),
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
            "unavailable_reason": "The market provider could not return usable listing evidence.",
            "valuation_date": now.isoformat(),
            "seller_asking_price_inr": format(asking_price, "f") if asking_price is not None else None,
            "ml_prediction": ml_prediction,
            "vehicle_researched": {"make": criteria.make, "model": criteria.model},
            "search_location": criteria.location,
            "geographic_scope": [],
            "relevant_result_count": 0,
            "priced_comparable_count": 0,
            "exact_vehicle_match_count": 0,
            "different_variant_count": 0,
            "different_year_count": 0,
            "general_market_information_count": 0,
            "priced_comparables": [],
            "market_research_results": [],
            "market_asking_price_range": None,
            "evidence_count": 0,
            "exact_match_count": 0,
            "close_match_count": 0,
            "approximate_match_count": 0,
            "minimum_evidence_count": MIN_COMPARABLE_COUNT,
            "freshness_threshold_days": 30,
            "data_freshness_days": None,
            "maximum_listing_age_days": None,
            "sources": [],
            "comparables": [],
            "rejected_listing_counts": {},
            "market_research_rejection_counts": {},
            "search_observability": {
                "queries_issued": 0,
                "raw_results_returned": 0,
                "listings_extracted": 0,
                "extraction_rejected": 0,
                "relevance_rejected": 0,
                "market_research_rejected": 0,
                "duplicates_removed": 0,
                "outliers_removed": 0,
                "final_exact_count": 0,
                "final_close_count": 0,
                "final_approximate_count": 0,
            },
            "data_origin": "unavailable",
            "refresh_status": "provider_error",
            "discrepancy": _ml_comparison(ml_prediction, None),
            "seller_asking_price_comparison": _compare_price_to_market(asking_price, "seller_asking", None),
            "limitations": [
                "The market provider could not return usable listing evidence; no numerical market range was inferred.",
                "Advertised prices are not verified transaction prices.",
            ],
        }


def _build_report_data(
    assessment: VehicleAssessment,
    images: list[UploadedImage],
    ocr_results: list[OCRResult],
    damage_findings: list[DamageFinding],
    damage_scans: list[EvidenceRecord],
    price_estimates: list[PriceEstimate],
    market_valuation: dict[str, Any],
    pricing_workflow_state: dict[str, Any] | None = None,
    vehicle_lookup_workflow_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_ids = {image.id for image in images}
    latest_ocr_by_image: dict[UUID, OCRResult] = {}
    for result in sorted(
        (item for item in ocr_results if item.result_type == "image_ocr"),
        key=lambda item: (item.created_at or datetime.min.replace(tzinfo=timezone.utc), str(item.id)),
    ):
        if result.image_id is not None and result.image_id in image_ids:
            latest_ocr_by_image[result.image_id] = result
    completed_ocr_image_ids = {
        image_id for image_id, result in latest_ocr_by_image.items() if result.status == "completed"
    }
    latest_ocr_statuses = {result.status for result in latest_ocr_by_image.values()}

    def no_success_ocr_status() -> str:
        if "failed" in latest_ocr_statuses:
            return "failed"
        if latest_ocr_statuses and latest_ocr_statuses <= {"unavailable"}:
            return "unavailable"
        if latest_ocr_statuses:
            return "unavailable"
        return "pending" if image_ids else "unavailable"

    if image_ids:
        completed_ocr_count = len(completed_ocr_image_ids)
        if completed_ocr_count == len(image_ids):
            ocr_status = "available"
        elif completed_ocr_count:
            ocr_status = "partial"
        else:
            ocr_status = no_success_ocr_status()
    elif any(result.result_type == "image_ocr" and result.status == "completed" for result in ocr_results):
        ocr_status = "available"
    else:
        ocr_status = no_success_ocr_status()

    findings_by_image: dict[UUID, int] = {}
    for finding in damage_findings:
        if finding.image_id is not None:
            findings_by_image[finding.image_id] = findings_by_image.get(finding.image_id, 0) + 1

    latest_damage_scans: dict[UUID, EvidenceRecord] = {}
    latest_eligibility_scans: dict[UUID, EvidenceRecord] = {}
    for scan in damage_scans:
        if scan.image_id is None or scan.image_id not in image_ids:
            continue
        destination = (
            latest_damage_scans
            if scan.evidence_type == "damage_detection"
            else latest_eligibility_scans
        )
        previous = destination.get(scan.image_id)
        if previous is None or (scan.created_at, str(scan.id)) > (
            previous.created_at,
            str(previous.id),
        ):
            destination[scan.image_id] = scan

    def damage_outcome(image: UploadedImage) -> dict[str, Any]:
        finding_count = findings_by_image.get(image.id, 0)
        scan = latest_damage_scans.get(image.id)
        data = scan.evidence_data if scan and isinstance(scan.evidence_data, dict) else {}
        status_value = data.get("status")
        if finding_count:
            outcome = "completed_with_findings"
        elif (
            status_value == "completed_no_findings"
            and data.get("detection_count", 0) == 0
        ):
            outcome = "completed_no_findings"
        elif status_value == "completed" and data.get("detection_count") == 0:
            outcome = "completed_no_findings"
        elif status_value == "failed":
            outcome = "failed"
        elif status_value == "unavailable":
            outcome = "unavailable"
        else:
            outcome = "unavailable"
            eligibility = latest_eligibility_scans.get(image.id)
            if eligibility and isinstance(eligibility.evidence_data, dict):
                if eligibility.evidence_data.get("status") == "failed":
                    outcome = "failed"
                if isinstance(eligibility.evidence_data.get("reason"), str):
                    data = eligibility.evidence_data
        reason = data.get("reason") if isinstance(data.get("reason"), str) else None
        return {"image_id": str(image.id), "status": outcome, "reason": reason}

    damage_image_results = [damage_outcome(image) for image in images]
    damage_statuses = [item["status"] for item in damage_image_results]
    if damage_findings or "completed_with_findings" in damage_statuses:
        damage_status = "completed_with_findings"
    elif damage_statuses and all(
        outcome == "completed_no_findings" for outcome in damage_statuses
    ):
        damage_status = "completed_no_findings"
    elif "failed" in damage_statuses:
        damage_status = "failed"
    else:
        damage_status = "unavailable"

    report_ocr_results = []
    selected_ocr_results = list(latest_ocr_by_image.values())
    selected_ocr_results.extend(
        result for result in ocr_results
        if result.result_type == "image_ocr" and result.image_id is None
    )
    for result in sorted(
        selected_ocr_results,
        key=lambda item: (item.created_at or datetime.min.replace(tzinfo=timezone.utc), str(item.id)),
    ):
        report_ocr_results.append({
            "id": str(result.id),
            "image_id": str(result.image_id) if result.image_id else None,
            "result_type": result.result_type,
            "status": result.status,
            "provider": result.provider,
            "text": result.raw_text,
            "confidence": _ocr_confidence(result),
            "created_at": _iso(result.created_at),
        })
    vehicle_lookup = _vehicle_details_lookup(ocr_results)
    if vehicle_lookup_workflow_state is not None:
        vehicle_lookup["workflow_status"] = vehicle_lookup_workflow_state.get("status")
        error_code = vehicle_lookup_workflow_state.get("error_code")
        if isinstance(error_code, str):
            vehicle_lookup["error_code"] = error_code
            vehicle_lookup["error_message"] = vehicle_lookup_workflow_state.get("error_message")
    lookup_details = vehicle_lookup.get("details") if isinstance(vehicle_lookup.get("details"), dict) else {}
    stored_vehicle_details = (
        assessment.vehicle_details
        if isinstance(assessment.vehicle_details, dict)
        else {}
    )
    assessment_details = dict(stored_vehicle_details)
    assessment_details.pop("asking_price_inr", None)
    registration_number_source = assessment_details.pop("registration_number_source", None) or vehicle_lookup.get("source")
    assessment_details.update(lookup_details)
    assessment_details.pop("asking_price_inr", None)
    vehicle_details_provenance = {
        key: "user_input"
        for key, value in stored_vehicle_details.items()
        if value is not None and key not in {"asking_price_inr", "registration_number_source"}
    }
    vehicle_details_provenance.update(
        vehicle_lookup.get("field_provenance", {})
    )
    report_pricing = {
        "status": "available" if price_estimates else "pending",
        "estimate_count": len(price_estimates),
        "input_provenance": {},
    }
    if pricing_workflow_state is not None:
        report_pricing.update(pricing_workflow_state)
        report_pricing["estimate_count"] = len(price_estimates)
        raw_input_provenance = pricing_workflow_state.get("input_provenance")
        if isinstance(raw_input_provenance, dict):
            report_pricing["input_provenance"] = {
                key: value
                for key, value in raw_input_provenance.items()
                if isinstance(value, str) and value in _PRICING_PROVENANCE_LABELS
            }

    report_price_estimates = []
    for estimate in price_estimates:
        estimate_data = estimate.estimate_data if isinstance(estimate.estimate_data, dict) else {}
        raw_vehicle = estimate_data.get("vehicle")
        raw_provenance = estimate_data.get("provenance")
        raw_vehicle = raw_vehicle if isinstance(raw_vehicle, dict) else {}
        raw_provenance = raw_provenance if isinstance(raw_provenance, dict) else {}
        estimate_provenance = {
            key: value
            for key, value in raw_provenance.items()
            if isinstance(value, str)
            and value in {"vehicle_api", "model_default", "user_input"}
            and (
                (key in _USER_PRICING_FIELDS and value == "user_input")
                or (key not in _USER_PRICING_FIELDS and value in {"vehicle_api", "model_default"})
            )
        }
        estimate_vehicle = {
            key: value
            for key, value in raw_vehicle.items()
            if key in {
                "make", "model", "variant", "registration_year", "odometer_km",
                "fuel_type", "transmission", "seller_type", "mileage_kmpl", "engine_cc",
                "max_power_bhp", "seats", "owner_count",
            }
            and key in estimate_provenance
        }
        report_price_estimates.append({
            "id": str(estimate.id),
            "amount": str(estimate.amount),
            "currency": estimate.currency,
            "lower_bound": str(estimate.lower_bound) if estimate.lower_bound is not None else None,
            "upper_bound": str(estimate.upper_bound) if estimate.upper_bound is not None else None,
            "method": estimate.method,
            "model_name": estimate.model_name,
            "vehicle": estimate_vehicle,
            "provenance": estimate_provenance,
            "created_at": _iso(estimate.created_at),
        })

    report_data = {
        "assessment": {
            "id": str(assessment.id),
            "status": assessment.status,
            "vin": assessment.vin,
            "registration_number": assessment.registration_number or lookup_details.get("registration_number"),
            "registration_number_source": registration_number_source,
            "vehicle_details": assessment_details,
            "vehicle_details_provenance": vehicle_details_provenance,
            "asking_price_inr": (
                format(assessment.asking_price_inr, "f")
                if assessment.asking_price_inr is not None
                else None
            ),
            "asking_price_inr_provenance": (
                "user_input" if assessment.asking_price_inr is not None else None
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
                "image_type": (image.image_metadata or {}).get("damage_eligibility", {}).get("image_type"),
                "damage_eligible": (image.image_metadata or {}).get("damage_eligibility", {}).get("damage_eligible"),
                "damage_rejection_reason": (image.image_metadata or {}).get("damage_eligibility", {}).get("damage_rejection_reason"),
            }
            for image in images
        ],
        "stages": {
            "ocr": {
                "status": ocr_status,
                "result_count": sum(result.status == "completed" for result in latest_ocr_by_image.values()),
                "completed_image_count": len(completed_ocr_image_ids),
            },
            "damage_detection": {
                "status": damage_status,
                "finding_count": len(damage_findings),
                "images_with_findings": len(
                    {finding.image_id for finding in damage_findings if finding.image_id is not None}
                ),
                "excluded_image_count": sum(
                    1
                    for image in images
                    if (image.image_metadata or {}).get("damage_eligibility", {}).get("damage_eligible") is False
                ),
                "excluded_images": [
                    {
                        "image_id": str(image.id),
                        "image_type": (image.image_metadata or {}).get("damage_eligibility", {}).get("image_type"),
                        "reason": (image.image_metadata or {}).get("damage_eligibility", {}).get("damage_rejection_reason"),
                    }
                    for image in images
                    if (image.image_metadata or {}).get("damage_eligibility", {}).get("damage_eligible") is False
                ],
                "completed_image_count": sum(
                    item["status"] in {
                        "completed_with_findings",
                        "completed_no_findings",
                    }
                    for item in damage_image_results
                ),
                "unavailable_count": sum(
                    item["status"] == "unavailable" for item in damage_image_results
                ),
                "failed_count": sum(
                    item["status"] == "failed" for item in damage_image_results
                ),
                "image_results": damage_image_results,
            },
            "pricing": report_pricing,
        },
        "ocr_results": report_ocr_results,
        "registration_ocr_assessment": build_registration_ocr_assessment(report_ocr_results),
        "registration_lookup_results": [
            {
                "id": str(result.id),
                "image_id": str(result.image_id) if result.image_id else None,
                "status": result.status,
                "provider": result.provider,
                "source": (result.result_data or {}).get("source"),
                "details": {
                    key: value
                    for key, value in (result.result_data or {}).items()
                    if key in {
                        "registration_number",
                        "make",
                        "model",
                        "variant",
                        "fuel_type",
                        "registration_date",
                        "manufacturing_year",
                        "vehicle_class",
                        "registration_valid_until",
                        "rto",
                        "fitness_expiry",
                        "puc_expiry",
                        "insurance_expiry",
                        "blacklisted",
                    }
                },
                "created_at": _iso(result.created_at),
            }
            for result in ocr_results
            if result.result_type == "registration_lookup"
        ],
        "vehicle_details_lookup": vehicle_lookup,
        "challan_lookup": _latest_vehicle_service(
            ocr_results,
            "challan_lookup",
            {"status", "records"},
        ),
        "fastag_lookup": _latest_vehicle_service(
            ocr_results,
            "fastag_lookup",
            {"status", "issuing_bank", "issue_date", "vehicle_class"},
        ),
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
        "price_estimates": report_price_estimates,
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
    # Use the stored estimate; if the estimate is missing or invalid, report
    # ml_estimate_inr as None rather than fabricating a prediction. The frontend
    # must never show a fallback or stale prediction as a genuine ML result.
    predicted = positive_decimal(estimate.amount) if estimate is not None else None
    if predicted is None and estimate is not None and estimate.amount is not None:
        # Estimate exists but amount could not be parsed as a positive decimal;
        # treat as unavailable rather than showing a possibly stale value.
        predicted = None
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
    elif market_valuation.get("relevant_result_count", 0) > 0:
        market_note = (
            "Relevant web-search market research is available separately in market_valuation, "
            "but it did not support a comparable price range."
        )
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
        "asking_price_provenance": "user_input" if asking is not None else None,
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


def _normalize_legacy_damage_report(report_data: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(report_data)
    stages = normalized.get("stages")
    if not isinstance(stages, dict):
        return normalized
    damage_stage = stages.get("damage_detection")
    if not isinstance(damage_stage, dict):
        return normalized

    findings = normalized.get("damage_findings")
    findings = findings if isinstance(findings, list) else []
    raw_status = damage_stage.get("status")
    if findings or damage_stage.get("finding_count", 0):
        status_value = "completed_with_findings"
    elif raw_status == "completed_no_findings":
        status_value = "completed_no_findings"
    elif raw_status == "failed":
        status_value = "failed"
    else:
        status_value = "unavailable"
    damage_stage["status"] = status_value

    narrative = normalized.get("llm_narrative")
    if isinstance(narrative, dict):
        narrative["detected_damage_explanation"] = damage_narrative_explanation(
            status_value, findings
        )
        narrative["overall_condition_summary"] = damage_condition_summary(status_value)
    return normalized


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
    damage_scans = list(
        db.scalars(
            select(EvidenceRecord)
            .where(
                EvidenceRecord.assessment_id == assessment_id,
                EvidenceRecord.evidence_type.in_(("damage_detection", "damage_eligibility")),
            )
            .order_by(EvidenceRecord.created_at, EvidenceRecord.id)
        ).all()
    )
    price_estimates = list(
        db.scalars(
            select(PriceEstimate)
            .where(PriceEstimate.assessment_id == assessment_id)
            .order_by(PriceEstimate.created_at, PriceEstimate.id)
        ).all()
    )
    return images, ocr_results, damage_findings, damage_scans, price_estimates


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
    return _generate_assessment_report_snapshot(assessment_id, db, market_provider)


def _generate_assessment_report_snapshot(
    assessment_id: UUID,
    db: Session,
    market_provider: MarketListingProvider | None,
    pricing_workflow_state: dict[str, Any] | None = None,
    vehicle_lookup_workflow_state: dict[str, Any] | None = None,
) -> AssessmentReportResponse:
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
        preview_lookup_results = list(
            db.scalars(
                select(OCRResult)
                .where(
                    OCRResult.assessment_id == assessment_id,
                    OCRResult.result_type == "registration_lookup",
                    OCRResult.status == "completed",
                )
                .order_by(OCRResult.created_at, OCRResult.id)
            ).all()
        )
        preview_report = db.scalar(
            select(AssessmentReport).where(AssessmentReport.assessment_id == assessment_id)
        )
        previous_market = None
        if preview_report and isinstance(preview_report.report_data, dict):
            candidate = preview_report.report_data.get("market_valuation")
            previous_market = deepcopy(candidate) if isinstance(candidate, dict) else None
        criteria, asking_price, ml_prediction = _market_inputs(
            preview_assessment, preview_estimates, preview_lookup_results
        )
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

            images, ocr_results, damage_findings, damage_scans, price_estimates = _load_stage_records(db, assessment_id)
            report = db.scalar(
                select(AssessmentReport)
                .where(AssessmentReport.assessment_id == assessment_id)
                .with_for_update()
            )
            report_data = _build_report_data(
                assessment,
                images,
                ocr_results,
                damage_findings,
                damage_scans,
                price_estimates,
                market_valuation,
                pricing_workflow_state,
                vehicle_lookup_workflow_state,
            )
            current_hash = report_data_hash(report_data)
            previous_narrative = None
            previous_hash = None
            if report is not None and isinstance(report.report_data, dict):
                previous_narrative = report.report_data.get("llm_narrative")
                previous_hash = report.report_data.get("llm_narrative_input_hash")
            if previous_hash == current_hash and previous_narrative is not None:
                report_data["llm_narrative"] = previous_narrative
            else:
                narrative = generate_llm_narrative(report_data)
                if narrative is not None:
                    report_data["llm_narrative"] = narrative.model_dump()
            report_data["llm_narrative_input_hash"] = current_hash
            report_data = _normalize_legacy_damage_report(report_data)
            if report is None:
                report = AssessmentReport(assessment_id=assessment_id)
                db.add(report)
            report.status = "draft"
            report.summary = "Draft snapshot of currently stored assessment results."
            serialized_market_valuation = report_data.get("market_valuation", {})
            if not isinstance(serialized_market_valuation, dict):
                serialized_market_valuation = {}
            market_results = serialized_market_valuation.get(
                "market_research_results", []
            )
            comparables = serialized_market_valuation.get("priced_comparables", [])
            search_observability = serialized_market_valuation.get(
                "search_observability", {}
            )
            report_log = {
                "event": "market_report_snapshot_results",
                "market_status": serialized_market_valuation.get("status"),
                "market_unavailable_reason": serialized_market_valuation.get(
                    "unavailable_reason"
                ),
                "report_market_research_count": (
                    len(market_results) if isinstance(market_results, list) else 0
                ),
                "report_priced_comparable_count": (
                    len(comparables) if isinstance(comparables, list) else 0
                ),
                "report_serialized_priced_listing_count": (
                    len(comparables) if isinstance(comparables, list) else 0
                ),
                "market_queries_issued": (
                    search_observability.get("queries_issued", 0)
                    if isinstance(search_observability, dict)
                    else 0
                ),
                "market_raw_results_returned": (
                    search_observability.get("raw_results_returned", 0)
                    if isinstance(search_observability, dict)
                    else 0
                ),
                "market_research_rejection_counts": serialized_market_valuation.get(
                    "market_research_rejection_counts", {}
                ),
            }
            logger.info(
                json.dumps(report_log, separators=(",", ":"), default=str),
                extra=report_log,
            )
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
    response = AssessmentReportResponse.model_validate(report)
    if isinstance(response.report_data, dict):
        response.report_data = _normalize_legacy_damage_report(response.report_data)
    return response
