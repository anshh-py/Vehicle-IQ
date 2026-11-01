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
    if isinstance(buyer_recommendation, dict):
        buyer_recommendation.pop("market_evidence_available", None)
        limitations = buyer_recommendation.get("limitations")
        if isinstance(limitations, list):
            buyer_recommendation["limitations"] = [
                item
                for item in limitations
                if not isinstance(item, str) or "market" not in item.casefold()
            ]
    evidence = {
        "assessment": assessment,
        "ocr_results": _safe(report_data.get("ocr_results") or []),
        "vehicle_details_lookup": vehicle_lookup,
        "vehicle_field_provenance": provenance,
        "damage_inspection": _safe(damage_inspection or {"status": "unavailable"}),
        "damage_findings": _safe(report_data.get("damage_findings") or []),
        "price_estimates": _safe(report_data.get("price_estimates") or []),
        "pricing_workflow": _safe(report_data.get("pricing") or (report_data.get("stages", {}).get("pricing") if isinstance(report_data.get("stages"), dict) else {})),
        "buyer_recommendation": buyer_recommendation,
        "limitations": _safe(report_data.get("limitations") or []),
    }
    system_prompt = (
        "You are an automotive inspection report assistant. "
        "Use only the supplied structured evidence. "
        "Do not invent vehicle specifications, damage findings, market prices, repair costs, mechanical condition, or inspection results. "
        "Do not recalculate or change the XGBoost price estimate. "
        "Fields with provenance vehicle_api are reported by the VehicleX registration lookup; describe them as API-reported facts, not as independently verified facts. "
        "Do not label Vehicle API fields as unverified. "
        "Distinguish API-reported facts from independently verified facts. "
        "YOLO damage detections are model-generated findings, not confirmed physical damage. "
        "Use the damage_detection stage status as the source of truth. "
        "When status is completed_no_findings, say only that no damage was detected in the supplied images; never say the vehicle is damage-free. "
        "When status is unavailable or failed, state that no conclusion about damage can be drawn. "
        "Never omit a stored damage detection. "
        "Never claim that no damage findings exist when detections are present. "
        "Do not summarize or infer market research, used-car prices, or market-value conclusions. "
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
        "price_estimate_explanation, missing_information_and_limitations."
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
    return narrative