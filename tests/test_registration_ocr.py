import pytest

from app.ocr.registration import (
    build_registration_ocr_assessment,
    normalize_registration_candidate,
)


def _result(image_id, text, confidence=0.99, status="completed"):
    return {
        "image_id": image_id,
        "text": text,
        "confidence": confidence,
        "status": status,
    }


def test_normalizes_separators_case_and_common_numeric_ocr_noise():
    assert normalize_registration_candidate("rj 45-ca 1879") == "RJ45CA1879"
    assert normalize_registration_candidate("RJ-4O-CA-18O9") == "RJ40CA1809"


@pytest.mark.parametrize(
    "raw_text, expected",
    [
        ("RJ45CA1879", "RJ45CA1879"),
        ("4 RJ45CA1879", "RJ45CA1879"),
        ("MH12XY0001", "MH12XY0001"),
        ("KA1A1234", "KA1A1234"),
        ("DL8CAB1234", "DL8CAB1234"),
        ("RJ 5 C 1234", "RJ5C1234"),
        ("22BH1234AB", "22BH1234AB"),
        ("R J 45 C A 1879", "RJ45CA1879"),
    ],
)
def test_extracts_supported_indian_registration_formats(raw_text, expected):
    assert normalize_registration_candidate(raw_text) == expected


@pytest.mark.parametrize(
    "raw_text",
    [
        "",
        "Dealership opening 2026",
        "RJ123CA1879",
        "RJ45CA12345",
        "4RJ45CA1879",
    ],
)
def test_rejects_unrelated_or_invalid_registration_formats(raw_text):
    assert normalize_registration_candidate(raw_text) is None


def test_character_corrections_are_limited_to_plate_positions():
    assert normalize_registration_candidate("RJ4SCA18B9") == "RJ45CA1889"
    assert normalize_registration_candidate("RJ45CA1879") == "RJ45CA1879"
    assert normalize_registration_candidate("4RJ45CA1879") is None


def test_matching_photo_candidates_are_consistent_but_unverified():
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ 45 CA 1879"), _result("img-2", "RJ45CA1879", 0.96)]
    )

    assert summary["status"] == "candidate"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    # The best confident read represents the agreed candidate; a agreeing photo
    # with a lower score must not veto a consistent candidate.
    assert summary["confidence"] == 0.99
    assert summary["raw_evidence"][0]["text"] == "RJ 45 CA 1879"
    assert summary["raw_evidence"][0]["confidence"] == 0.99
    assert "unverified candidate" in summary["note"]


def test_conflicting_photo_candidates_need_review():
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ45CA1879"), _result("img-2", "DL8CAB1234")]
    )

    assert summary["status"] == "needs_review"
    assert summary["normalized_candidate"] is None
    assert [item["normalized_candidate"] for item in summary["raw_evidence"]] == [
        "RJ45CA1879",
        "DL8CAB1234",
    ]


def test_conflicting_valid_candidates_select_unique_high_confidence_read():
    summary = build_registration_ocr_assessment(
        [
            _result("img-1", "RJ45CA1879", 0.97),
            _result("img-2", "DL8CAB1234", 0.84),
        ]
    )

    assert summary["status"] == "candidate"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    assert summary["confidence"] == 0.97


def test_low_confidence_candidate_needs_review():
    summary = build_registration_ocr_assessment([_result("img-1", "RJ45CA1879", 0.42)])

    assert summary["status"] == "needs_review"
    assert summary["normalized_candidate"] == "RJ45CA1879"


def test_empty_and_malformed_ocr_are_not_candidates():
    empty = build_registration_ocr_assessment([_result("img-1", "", None)])
    malformed = build_registration_ocr_assessment([_result("img-2", "The text is not visible")])

    assert empty["status"] == "unavailable"
    assert empty["normalized_candidate"] is None
    assert malformed["status"] == "needs_review"
    assert malformed["normalized_candidate"] is None
    assert normalize_registration_candidate("RJ45CA1879 and DL8CAB1234") is None


def test_leading_digit_ocr_variant_is_normalized_without_stripping_digits():
    # A space separates the stray digit from the plate, so the lookbehind passes
    # and no arbitrary digit is stripped.
    assert normalize_registration_candidate("4 RJ45CA1879") == "RJ45CA1879"
    assert normalize_registration_candidate("RJ45CA1879 4") == "RJ45CA1879"
    # A stray digit glued to the plate is ambiguous and must not be forced.
    assert normalize_registration_candidate("4RJ45CA1879") is None


def test_consistent_candidate_with_one_missing_confidence_is_usable():
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ45CA1879", 0.99), _result("img-2", "4 RJ45CA1879", None)]
    )

    assert summary["status"] == "candidate"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    assert summary["confidence"] == 0.99


def test_consistent_candidate_with_one_noisy_photo_is_usable():
    # Agreement across photos means one lower-confidence read does not veto the
    # consistent plate; the best confident read represents the candidate.
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ45CA1879", 0.99), _result("img-2", "4 RJ45CA1879", 0.50)]
    )

    assert summary["status"] == "candidate"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    assert summary["confidence"] == 0.99


def test_all_missing_confidence_still_requires_review():
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ45CA1879", None), _result("img-2", "4 RJ45CA1879", None)]
    )

    assert summary["status"] == "needs_review"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    assert summary["confidence"] is None


def test_all_low_confidence_still_requires_review():
    summary = build_registration_ocr_assessment(
        [_result("img-1", "RJ45CA1879", 0.40), _result("img-2", "4 RJ45CA1879", 0.50)]
    )

    assert summary["status"] == "needs_review"
    assert summary["normalized_candidate"] == "RJ45CA1879"
    assert summary["confidence"] == 0.50
