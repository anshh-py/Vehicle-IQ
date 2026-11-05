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
    assert pricing_decision("500000", "research-model", unavailable)["withholding_message"] == INSPECTION_WITHHOLDING_MESSAGE
    assert pricing_decision("500000", "research-model", failed)["status"] == "withheld_inspection_failed"


def test_confidence_alone_does_not_change_damage_condition_category():
    low = assess_condition([completed_image()], [{"damage_type": "dent", "confidence": 0.01}])
    high = assess_condition([completed_image()], [{"damage_type": "dent", "confidence": 0.99}])
    assert low["category"] == high["category"] == "minor_cosmetic"
