import hashlib
import json
from typing import Any

from app.llm.client import LLMClient, LLMError, get_llm_client
from app.llm.schemas import LLMReportNarrative


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    return str(value)


def build_llm_prompt(report_data: dict[str, Any]) -> tuple[str, str]:
    vehicle_lookup = _safe(report_data.get("vehicle_details_lookup") or {})
    assessment = _safe(report_data.get("assessment") or {})
    vehicle_fields = {}
    if isinstance(assessment, dict) and isinstance(assessment.get("vehicle_details"), dict):
        vehicle_fields = assessment["vehicle_details"]
    provenance = {}
    stages = report_data.get("stages")
    damage_inspection = (
        stages.get("damage_detection")
        if isinstance(stages, dict)
        else None
    )
    for field in ("make", "model", "variant", "fuel_type", "registration_date", "manufacturing_year", "rto", "vehicle_class", "fitness_expiry", "puc_expiry", "insurance_expiry", "engine_cc", "seats", "max_power_bhp", "transmission", "mileage_kmpl", "owner_count"):
        if field in vehicle_fields and vehicle_fields.get(field) not in (None, ""):
            provenance[field] = "vehicle_api (VehicleX registration lookup; API-reported, not independently verified)"
    buyer_recommendation = _safe(report_data.get("buyer_recommendation") or {})
    market = report_data.get("market_valuation")
    if isinstance(market, dict):
        market = {
            key: _safe(market.get(key))
            for key in (
                "status", "unavailable_reason", "evidence_count",
                "minimum_evidence_count", "market_asking_price_range",
                "priced_comparables", "market_research_results",
                "search_observability", "geographic_scope",
                "rejected_listing_counts", "comparison_rejection_counts",
                "market_research_rejection_counts", "rejection_counts_by_stage",
                "limitations", "sources", "vehicle_researched",
            )
            if key in market
        }
        if isinstance(market.get("market_research_results"), list):
            # Discovery context is kept compact; the validated comparable set
            # below is the complete evidence used for any market range.
            market["market_research_results"] = market["market_research_results"][:5]
    else:
        market = {"status": "unavailable", "evidence_count": 0}

    registration_results = _safe(report_data.get("registration_lookup_results") or [])
    if isinstance(registration_results, list):
        for lookup in registration_results:
            if isinstance(lookup, dict):
                lookup.pop("registration_number", None)
    registration_ocr = _safe(report_data.get("registration_ocr_assessment") or {})
    vehicle_services = {
        "challan": _safe(report_data.get("challan_lookup") or {"status": "not_run"}),
        "fastag": _safe(report_data.get("fastag_lookup") or {"status": "not_run"}),
    }
    evidence = {
        "assessment": assessment,
        "ocr_results": _safe(report_data.get("ocr_results") or []),
        "vehicle_details_lookup": vehicle_lookup,
        "registration_lookup_results": registration_results,
        "registration_ocr_assessment": registration_ocr,
        "vehicle_services": vehicle_services,
        "vehicle_field_provenance": provenance,
        "damage_inspection": _safe(damage_inspection or {"status": "unavailable"}),
        "damage_findings": _safe(report_data.get("damage_findings") or []),
        "price_estimates": _safe(report_data.get("price_estimates") or []),
        "pricing_decision": _safe(report_data.get("pricing_decision") or {}),
        "pricing_workflow": _safe(report_data.get("pricing") or (report_data.get("stages", {}).get("pricing") if isinstance(report_data.get("stages"), dict) else {})),
        "buyer_recommendation": buyer_recommendation,
        "market_valuation": market,
        "limitations": _safe(report_data.get("limitations") or []),
    }
    decision = report_data.get("pricing_decision")
    if isinstance(decision, dict) and not decision.get("pricing_eligible"):
        # Keep suppressed base estimates in the deterministic audit record,
        # but never provide their amount to narrative generation.
        evidence["price_estimates"] = [
            {key: value for key, value in item.items() if key not in {"amount", "lower_bound", "upper_bound"}}
            for item in evidence["price_estimates"] if isinstance(item, dict)
        ]
        if isinstance(evidence.get("market_valuation"), dict):
            evidence["market_valuation"]["ml_prediction"] = {"status": "withheld_by_condition_policy"}
    system_prompt = (
        "You are an automotive inspection report assistant. "
        "Use only the supplied structured evidence. "
        "Do not invent vehicle specifications, damage findings, market prices, repair costs, mechanical condition, or inspection results. "
        "Do not recalculate or change the XGBoost price estimate. "
        "The deterministic pricing_decision controls whether a customer-facing resale estimate may be shown. Never override it. If pricing_eligible is false, do not reveal or restate the audit-only base estimate; explain only the withholding_message and required next steps. "
        "Fields with provenance vehicle_api are reported by the VehicleX registration lookup; describe them as API-reported facts, not as independently verified facts. "
        "Do not label Vehicle API fields as unverified. "
        "Distinguish API-reported facts from independently verified facts. "
        "YOLO damage detections are model-generated findings, not confirmed physical damage. "
        "Use the damage_detection stage status as the source of truth. "
        "When status is completed_no_findings, say only that no damage was detected in the supplied images; never say the vehicle is damage-free. "
        "When status is unavailable or failed, state that no conclusion about damage can be drawn. "
        "Never omit a stored damage detection. "
        "Never claim that no damage findings exist when detections are present. "
        "Summarize the supplied validated comparable listings, their source URLs, asking prices and years when present. Search results are discovered advertisements, not verified active listings or completed sales. "
        "Never invent a listing, price, source, market range, or missing vehicle field. Do not calculate or alter market statistics; repeat only the supplied deterministic range and evidence count. "
        "When market valuation is unavailable or evidence is below the stated minimum, say that no market range is available. Keep the market asking-price evidence separate from the XGBoost estimate. "
        "Describe OCR registration candidates as unverified text and vehicle API values as provider-reported, not independently verified. "
        "If evidence is insufficient, say so. "
        "Clearly identify user-provided specifications as unverified. "
        "Do not present this as a professional mechanical inspection. "
        "Return valid JSON matching the required schema only."
    )
    user_prompt = (
        "Generate a grounded inspection narrative from this evidence:\n"
        + json.dumps(evidence, sort_keys=True, default=str)
        + "\nReturn JSON with keys: vehicle_overview, overall_condition_summary, "
        "detected_damage_explanation, potential_inspection_risks, buyer_recommendations, "
        "price_estimate_explanation, missing_information_and_limitations, "
        "vehicle_identity_and_lookup_summary, market_valuation_explanation. "
        "Keep the market explanation concise and only use the supplied market_valuation fields."
    )
    return system_prompt, user_prompt


def report_data_hash(report_data: dict[str, Any]) -> str:
    payload = {key: value for key, value in report_data.items() if key not in {"llm_narrative", "llm_narrative_input_hash"}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def damage_narrative_explanation(
    status: str,
    damage_findings: list[Any],
) -> str:
    if status == "completed_with_findings":
        damage_types = [
            str(finding.get("damage_type") or finding.get("label") or "damage")
            for finding in damage_findings
            if isinstance(finding, dict)
        ]
        detail = ": " + ", ".join(damage_types) if damage_types else ""
        return (
            f"The model returned damage findings in the supplied images{detail}. "
            "These are model-generated findings, not confirmed physical damage."
        )
    if status == "completed_no_findings":
        return (
            "No damage was detected in the supplied images. This does not establish "
            "that the vehicle is damage-free."
        )
    if status == "failed":
        return (
            "Damage inspection failed before it could be completed. No conclusion "
            "about damage can be drawn from the supplied images."
        )
    return (
        "Damage inspection was unavailable for the supplied images. No conclusion "
        "about damage can be drawn."
    )


def damage_condition_summary(status: str) -> str:
    if status == "completed_with_findings":
        return (
            "The damage model returned findings from the supplied images. The "
            "findings are not confirmed physical damage and do not establish the "
            "vehicle's complete condition."
        )
    if status == "completed_no_findings":
        return (
            "No damage was detected in the supplied images. This does not establish "
            "that the vehicle is damage-free."
        )
    if status == "failed":
        return (
            "Damage inspection failed before it could be completed. No conclusion "
            "about damage can be drawn from the supplied images."
        )
    return (
        "Damage inspection was unavailable for the supplied images. No conclusion "
        "about damage can be drawn."
    )


def _grounded_market_explanation(market: Any) -> str:
    if not isinstance(market, dict):
        return "No market research result was supplied, so no market price range is available."
    status = market.get("status")
    count = market.get("evidence_count", 0)
    minimum = market.get("minimum_evidence_count", 3)
    price_range = market.get("market_asking_price_range")
    if status == "available" and isinstance(price_range, dict):
        return (
            f"An observed asking-price range was calculated from {count} comparable advertisements. "
            f"The reported interquartile range is ₹{price_range.get('lower_inr')} to "
            f"₹{price_range.get('upper_inr')}, with a median advertised price of "
            f"₹{price_range.get('median_inr')}. These are asking prices, not completed-sale values, "
            "and are reported separately from the ML estimate."
        )
    if count and isinstance(count, int):
        return (
            f"Only {count} year-compatible comparable advertisement(s) were available; "
            f"at least {minimum} are required for a market asking-price range. No range is reported."
        )
    reason = market.get("unavailable_reason")
    if isinstance(reason, str) and reason.strip():
        return f"Market valuation is unavailable. {reason.strip()} No market price was inferred."
    return "Market valuation is unavailable; no comparable-market price range was inferred."


def generate_llm_narrative(
    report_data: dict[str, Any],
    *,
    client: LLMClient | None = None,
    timeout_seconds: float | None = None,
) -> LLMReportNarrative | None:
    llm_client = client if client is not None else get_llm_client()
    if llm_client is None or not llm_client.enabled:
        return None
    system_prompt, user_prompt = build_llm_prompt(report_data)
    try:
        raw = llm_client.complete_json(system_prompt, user_prompt, timeout_seconds=timeout_seconds)
        narrative = LLMReportNarrative.model_validate_json(raw)
    except (LLMError, ValueError, TypeError):
        return None

    damage_findings = report_data.get("damage_findings") or []
    stages = report_data.get("stages")
    damage_stage = stages.get("damage_detection") if isinstance(stages, dict) else None
    damage_status = (
        damage_stage.get("status")
        if isinstance(damage_stage, dict)
        else None
    )
    if damage_status == "results_available":
        damage_status = "completed_with_findings" if damage_findings else "unavailable"
    if damage_status not in {
        "completed_with_findings",
        "completed_no_findings",
        "unavailable",
        "failed",
    }:
        damage_status = "unavailable"
    narrative.detected_damage_explanation = damage_narrative_explanation(
        damage_status,
        damage_findings if isinstance(damage_findings, list) else [],
    )
    narrative.overall_condition_summary = damage_condition_summary(damage_status)
    narrative.market_valuation_explanation = _grounded_market_explanation(
        report_data.get("market_valuation")
    )
    return narrative
