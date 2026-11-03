import json

import pytest

from app.llm.client import LLMClient, LLMError
from app.llm.report import build_llm_prompt, generate_llm_narrative, LLMReportNarrative


def test_valid_structured_llm_output():
    payload = {
        "vehicle_overview": "A",
        "overall_condition_summary": "B",
        "detected_damage_explanation": "C",
        "potential_inspection_risks": "D",
        "buyer_recommendations": "E",
        "price_estimate_explanation": "F",
        "missing_information_and_limitations": "G",
        "vehicle_identity_and_lookup_summary": "Identity evidence is partial.",
        "market_valuation_explanation": "No range is available.",
    }

    class FakeClient:
        enabled = True

        def complete_json(self, system_prompt, user_prompt, timeout_seconds=None):
            return json.dumps(payload)

    narrative = generate_llm_narrative({"assessment": {"id": "1"}}, client=FakeClient())
    assert isinstance(narrative, LLMReportNarrative)
    assert narrative.vehicle_overview == "A"


def test_invalid_json_returns_none():
    class FakeClient:
        enabled = True

        def complete_json(self, system_prompt, user_prompt, timeout_seconds=None):
            return "{not json"

    assert generate_llm_narrative({"assessment": {}}, client=FakeClient()) is None


def test_provider_failure_returns_none():
    class FakeClient:
        enabled = True

        def complete_json(self, system_prompt, user_prompt, timeout_seconds=None):
            raise LLMError("timeout")

    assert generate_llm_narrative({"assessment": {}}, client=FakeClient()) is None


def test_prompt_includes_all_report_evidence_and_marks_market_as_unverified():
    report_data = {
        "assessment": {"vehicle_details": {"engine_cc": 1197}},
        "stages": {"damage_detection": {"status": "unavailable"}},
        "pricing": {"result": {"amount": "100", "provenance": {"engine_cc": "user_input"}}},
        "market_valuation": {"status": "unavailable"},
        "registration_ocr_assessment": {"status": "needs_review"},
        "registration_lookup_results": [{"status": "partial", "registration_number": "SECRET-PLATE"}],
        "challan_lookup": {"status": "not_run"},
        "fastag_lookup": {"status": "not_run"},
    }
    system_prompt, user_prompt = build_llm_prompt(report_data)
    assert "user_input" in user_prompt
    assert "unavailable" in user_prompt
    assert "damage_inspection" in user_prompt
    assert "Do not invent" in system_prompt
    assert "never say the vehicle is damage-free" in system_prompt
    assert "market_valuation" in user_prompt
    assert "registration_ocr_assessment" in user_prompt
    assert "challan" in user_prompt
    assert "fastag" in user_prompt
    assert "SECRET-PLATE" not in user_prompt
    assert "completed sales" in system_prompt
    assert "market_evidence_available" not in user_prompt


def test_groq_receives_validated_comparable_sources_and_market_diagnostics():
    report_data = {
        "market_valuation": {
            "status": "available",
            "market_research_results": [{"source_url": "https://cars24.com/ad/1"}],
            "priced_comparables": [
                {"title": "Hyundai i10 2013", "source_url": "https://cars24.com/ad/1", "advertised_price_inr": "225000", "year": 2013}
            ],
            "search_observability": {"raw_results_returned": 12, "final_comparable_listings": 1},
            "geographic_scope": ["India nationwide; no city preference"],
        },
        "assessment": {"asking_price_inr": "250000", "vehicle_details": {"make": "Hyundai", "model": "i10"}},
        "pricing": {"result": {"amount_inr": "240000", "status": "research_only"}},
        "limitations": ["No market range below the evidence minimum."],
    }
    _system_prompt, user_prompt = build_llm_prompt(report_data)
    assert "https://cars24.com/ad/1" in user_prompt
    assert "225000" in user_prompt
    assert "India nationwide; no city preference" in user_prompt
    assert "asking_price_inr" in user_prompt or "asking_price" in user_prompt


def test_market_narrative_is_grounded_in_report_values_and_legacy_output_is_compatible():
    payload = {
        "vehicle_overview": "A", "overall_condition_summary": "B",
        "detected_damage_explanation": "C", "potential_inspection_risks": "D",
        "buyer_recommendations": "E", "price_estimate_explanation": "F",
        "missing_information_and_limitations": "G",
    }

    class FakeClient:
        enabled = True

        def complete_json(self, _system_prompt, _user_prompt, timeout_seconds=None):
            # Old seven-key model response remains valid; deterministic post-processing
            # supplies the additive market section.
            return json.dumps(payload)

    narrative = generate_llm_narrative(
        {"market_valuation": {
            "status": "available", "evidence_count": 3, "minimum_evidence_count": 3,
            "market_asking_price_range": {
                "lower_inr": "100000", "median_inr": "120000", "upper_inr": "140000"
            },
        }},
        client=FakeClient(),
    )
    assert "3 comparable advertisements" in narrative.market_valuation_explanation
    assert "₹100000" in narrative.market_valuation_explanation
    assert "not completed-sale values" in narrative.market_valuation_explanation


def test_no_client_returns_none(monkeypatch):
    monkeypatch.setattr("app.llm.report.get_llm_client", lambda: None)
    assert generate_llm_narrative({"assessment": {}}, client=None) is None


@pytest.mark.parametrize(
    ("status", "findings", "expected"),
    [
        (
            "completed_with_findings",
            [{"damage_type": "dent"}],
            "The model returned damage findings in the supplied images: dent. "
            "These are model-generated findings, not confirmed physical damage.",
        ),
        (
            "completed_no_findings",
            [],
            "No damage was detected in the supplied images. This does not establish "
            "that the vehicle is damage-free.",
        ),
        (
            "unavailable",
            [],
            "Damage inspection was unavailable for the supplied images. No conclusion "
            "about damage can be drawn.",
        ),
        (
            "failed",
            [],
            "Damage inspection failed before it could be completed. No conclusion "
            "about damage can be drawn from the supplied images.",
        ),
    ],
)
def test_damage_narrative_matches_inspection_status(status, findings, expected):
    payload = {
        "vehicle_overview": "Vehicle.",
        "overall_condition_summary": "Review the supplied evidence.",
        "detected_damage_explanation": "The original generated text is replaced.",
        "potential_inspection_risks": "Inspection required.",
        "buyer_recommendations": "Inspect before purchase.",
        "price_estimate_explanation": "Model output.",
        "missing_information_and_limitations": "Limited evidence.",
    }

    class FakeClient:
        enabled = True

        def complete_json(self, _system_prompt, _user_prompt, timeout_seconds=None):
            return json.dumps(payload)

    narrative = generate_llm_narrative(
        {
            "stages": {"damage_detection": {"status": status}},
            "damage_findings": findings,
        },
        client=FakeClient(),
    )

    assert narrative.detected_damage_explanation == expected
    if status == "completed_no_findings":
        assert narrative.overall_condition_summary == expected
        assert "damage-free" in narrative.overall_condition_summary
    elif status in {"unavailable", "failed"}:
        assert "No conclusion about damage" in narrative.overall_condition_summary
