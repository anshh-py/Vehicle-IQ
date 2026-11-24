from app.pricing.condition import (
    INSPECTION_WITHHOLDING_MESSAGE,
    SEVERE_WITHHOLDING_MESSAGE,
    assess_condition,
    pricing_decision,
)


def completed_image(image_id="a"):
    return {"image_id": image_id, "image_type": "vehicle_exterior", "damage_eligible": True, "status": "completed_no_findings"}


def test_successful_empty_inference_permits_research_base_estimate():
    condition = assess_condition([completed_image()], [])
    decision = pricing_decision("500000", "research-model", condition)
    assert condition["category"] == "no_visible_damage"
    assert decision["status"] == "available"
    assert decision["customer_facing_estimate_inr"] == "500000"
    assert decision["audit"]["base_ml_estimate_inr"] == "500000"
    assert decision["adjustment_methodology"] is None


def test_minor_scratch_without_repair_evidence_has_no_fabricated_deduction():
    condition = assess_condition([completed_image()], [{"damage_type": "scratch", "confidence": 0.99}])
    decision = pricing_decision("500000", "research-model", condition)
    assert condition["category"] == "minor_cosmetic"
    assert decision["status"] == "withheld_insufficient_adjustment_evidence"
    assert decision["condition_adjustment_inr"] is None
    assert "adjusted_estimate_inr" not in decision
    assert decision["customer_facing_estimate_inr"] is None


def test_moderate_damage_without_validated_method_withholds_estimate():
    condition = assess_condition([completed_image()], [{"damage_type": "broken_lamp"}])
    decision = pricing_decision("500000", "research-model", condition)
    assert condition["category"] == "moderate_damage"
    assert decision["pricing_eligible"] is False
    assert decision["adjustment_methodology"] is None
    assert decision["repair_cost_evidence"] is None


def test_supported_major_collision_class_withholds_and_retains_base_for_audit():
    condition = assess_condition([completed_image()], [{"damage_type": "major_collision", "confidence": 0.4}])
    decision = pricing_decision("500000", "research-model", condition)
    assert condition["category"] == "severe_damage"
    assert condition["requires_human_review"] is True
    assert decision["status"] == "withheld_severe_damage"
    assert decision["withholding_message"] == SEVERE_WITHHOLDING_MESSAGE
    assert decision["customer_facing_estimate_inr"] is None
    assert decision["audit"]["base_ml_estimate_inr"] == "500000"


def test_inspection_unavailable_and_failed_are_not_no_damage():
    unavailable = assess_condition([{"image_type": "unknown", "damage_eligible": False, "status": "unavailable"}], [])
    failed = assess_condition([{"image_type": "vehicle_exterior", "damage_eligible": True, "status": "failed"}], [])
    assert unavailable["category"] == "inspection_unavailable"
    assert failed["category"] == "inspection_failed"
    decision_unavailable = pricing_decision("500000", "research-model", unavailable)
    decision_failed = pricing_decision("500000", "research-model", failed)
    assert decision_unavailable["status"] == "available_base_market"
    assert decision_unavailable["pricing_eligible"] is True
    assert decision_unavailable["customer_facing_estimate_inr"] == "500000"
    assert decision_failed["status"] == "available_base_market"
    assert decision_failed["pricing_eligible"] is True
    assert decision_failed["customer_facing_estimate_inr"] == "500000"


def test_confidence_alone_does_not_change_damage_condition_category():
    low = assess_condition([completed_image()], [{"damage_type": "dent", "confidence": 0.01}])
    high = assess_condition([completed_image()], [{"damage_type": "dent", "confidence": 0.99}])
    assert low["category"] == high["category"] == "minor_cosmetic"


def test_unavailable_condition_surfaces_labeled_base_market_valuation():
    """A valid base ML estimate is shown as a base (unadjusted) valuation when
    condition evidence is unavailable, instead of being fully withheld."""
    condition = assess_condition(
        [{"image_type": "unknown", "damage_eligible": False, "status": "unavailable"}],
        [],
    )
    decision = pricing_decision("625000", "research-model", condition)
    assert decision["status"] == "available_base_market"
    assert decision["pricing_eligible"] is True
    assert decision["customer_facing_estimate_inr"] == "625000"
    assert decision["customer_facing_currency"] == "INR"
    assert decision["condition_category"] == "inspection_unavailable"
    assert decision["human_review_required"] is True
    assert decision["audit"]["base_ml_estimate_inr"] == "625000"
    assert decision["audit"]["is_condition_adjusted"] is False
    assert decision["adjustment_methodology"] is None
    assert "not condition-adjusted" in decision["withholding_message"]


def test_unavailable_condition_without_base_amount_stays_unavailable():
    condition = assess_condition(
        [{"image_type": "unknown", "damage_eligible": False, "status": "unavailable"}],
        [],
    )
    decision = pricing_decision(None, "research-model", condition)
    assert decision["status"] == "unavailable"
    assert decision["pricing_eligible"] is False
    assert decision["customer_facing_estimate_inr"] is None
    assert decision["withholding_reason"] == "base_ml_estimate_unavailable"


def test_condition_available_and_clean_permits_base_estimate():
    condition = assess_condition([completed_image()], [])
    decision = pricing_decision("625000", "research-model", condition)
    assert decision["status"] == "available"
    assert decision["pricing_eligible"] is True
    assert decision["customer_facing_estimate_inr"] == "625000"


def test_actual_damage_findings_still_withhold_base_estimate():
    condition = assess_condition(
        [completed_image()],
        [{"damage_type": "broken_lamp", "confidence": 0.9}],
    )
    decision = pricing_decision("625000", "research-model", condition)
    assert decision["status"] == "withheld_insufficient_adjustment_evidence"
    assert decision["pricing_eligible"] is False
    assert decision["customer_facing_estimate_inr"] is None
