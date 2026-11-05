"""Deterministic condition and customer-facing valuation policy.

Damage detections are visual model evidence, not confirmed repairs or a
mechanical inspection. No price adjustment is made without a supported source.
"""

from decimal import Decimal, InvalidOperation
from typing import Any


SEVERE_WITHHOLDING_MESSAGE = (
    "Standard resale valuation withheld due to potentially severe damage. "
    "Obtain a professional inspection and repair assessment."
)
INSPECTION_WITHHOLDING_MESSAGE = (
    "Condition assessment unavailable. A reliable condition-adjusted resale estimate cannot be provided."
)


def assess_condition(images: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive a conservative assessment from persisted per-image outcomes."""
    eligible = [image for image in images if image.get("damage_eligible") is True]
    unresolved = [
        image for image in images
        if image.get("damage_eligible") is None
        or (image.get("damage_eligible") is False and image.get("image_type") == "unknown")
    ]
    states = [str(image.get("status") or "unavailable") for image in eligible]
    if any(state == "failed" for state in states):
        category, inspection_status = "inspection_failed", "failed"
    elif unresolved or not eligible or any(state != "completed_no_findings" and state != "completed_with_findings" for state in states):
        category, inspection_status = "inspection_unavailable", "unavailable"
    elif not findings:
        category, inspection_status = "no_visible_damage", "completed"
    else:
        classes = {str(item.get("damage_type") or "").casefold().replace(" ", "_") for item in findings}
        if "major_collision" in classes:
            category = "severe_damage"
        elif classes and classes <= {"dent", "scratch", "scratches", "minor_scratch"}:
            category = "minor_cosmetic"
        else:
            # Unclassified/part damage is not assigned a precise severity.
            category = "moderate_damage"
        inspection_status = "completed_with_findings"
    return {
        "category": category,
        "inspection_status": inspection_status,
        "finding_count": len(findings),
        "requires_human_review": category in {
            "minor_cosmetic", "moderate_damage", "severe_damage",
            "inspection_unavailable", "inspection_failed",
        },
        "severity_basis": (
            "model_class_major_collision" if category == "severe_damage"
            else "supported_damage_classes" if findings and category in {"minor_cosmetic", "moderate_damage"}
            else "completed_inference_without_findings" if category == "no_visible_damage"
            else "inspection_evidence_status"
        ),
        "limitations": [
            "Photographs cannot establish structural, airbag, chassis, or mechanical condition.",
            "Model detections are visual evidence and require human verification.",
        ],
    }


def pricing_decision(base_amount: Any, model_name: str | None, condition: dict[str, Any]) -> dict[str, Any]:
    """Decide whether a research ML base estimate may be shown as resale guidance."""
    try:
        amount = Decimal(str(base_amount))
        if not amount.is_finite() or amount <= 0:
            amount = None
    except (InvalidOperation, TypeError, ValueError):
        amount = None

    category = condition.get("category")
    status = "unavailable"
    reason = "base_ml_estimate_unavailable"
    customer_amount = None
    message = None
    if category == "severe_damage":
        status, reason, message = "withheld_severe_damage", "potentially_severe_damage", SEVERE_WITHHOLDING_MESSAGE
    elif category == "inspection_failed":
        status, reason, message = "withheld_inspection_failed", "inspection_failed", INSPECTION_WITHHOLDING_MESSAGE
    elif category == "inspection_unavailable":
        status, reason, message = "withheld_inspection_unavailable", "inspection_unavailable", INSPECTION_WITHHOLDING_MESSAGE
    elif category in {"minor_cosmetic", "moderate_damage"}:
        status = "withheld_insufficient_adjustment_evidence"
        reason = "no_validated_condition_adjustment_method"
        message = "Condition-adjusted valuation is unavailable until repair-cost evidence or a professional inspection and repair assessment are available. No damage deduction has been invented."
    elif category == "no_visible_damage":
        if amount is not None:
            status = "available"
            reason = None
            customer_amount = format(amount, "f")
            message = "Base research-model estimate shown after successful image inference; visual review does not guarantee mechanical condition."
        else:
            status = "unavailable"
            reason = "base_ml_estimate_unavailable"
            message = "A completed condition review is available, but no valid base model estimate is stored."

    return {
        "status": status,
        "pricing_eligible": customer_amount is not None,
        "customer_facing_estimate_inr": customer_amount,
        "customer_facing_currency": "INR" if customer_amount is not None else None,
        "withholding_reason": reason,
        "withholding_message": message,
        "condition_category": category,
        "inspection_status": condition.get("inspection_status"),
        "condition_adjustment_inr": None,
        "adjustment_methodology": None,
        "repair_cost_evidence": None,
        "human_review_required": bool(condition.get("requires_human_review")),
        "audit": {
            "base_ml_estimate_inr": format(amount, "f") if amount is not None else None,
            "model_name": model_name,
            "provenance": "research_only_xgboost_prediction" if amount is not None else None,
            "is_condition_adjusted": False,
        },
        "limitations": list(condition.get("limitations") or []) + [
            "The XGBoost output is a research-only model estimate, not verified market value.",
            "No repair cost or unsupported percentage deduction is included.",
        ],
    }
