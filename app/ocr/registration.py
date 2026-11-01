"""Indian vehicle-registration candidates derived from OCR evidence."""

import math
import re
from typing import Any


REGISTRATION_OCR_MIN_CONFIDENCE = 0.80

_OCR_TOKEN_PATTERN = re.compile(r"[A-Z0-9]+", re.IGNORECASE)
_BH_PATTERN = re.compile(
    r"(?<![A-Z0-9])([0-9OQILBS]{2})[\s./-]*BH[\s./-]*"
    r"([0-9OQILBS]{4})[\s./-]*([A-Z]{1,2})(?![A-Z0-9])",
    re.IGNORECASE,
)
_DIGIT_OCR_CORRECTIONS = str.maketrans(
    {"O": "0", "Q": "0", "I": "1", "L": "1", "B": "8", "S": "5"}
)
_LETTER_OCR_CORRECTIONS = str.maketrans(
    {"0": "O", "1": "I", "8": "B", "5": "S"}
)
_STANDARD_REGISTRATION = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,4}[0-9]{1,4}$")
_BH_REGISTRATION = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


def _corrected(value: str, correction_map: dict[int, int]) -> tuple[str, int]:
    normalized = value.translate(correction_map)
    return normalized, sum(
        original != corrected
        for original, corrected in zip(value, normalized)
    )


def _candidate_set(raw_text: str) -> set[str]:
    candidates: set[str] = set()
    text = raw_text.upper()
    tokens = _OCR_TOKEN_PATTERN.findall(text)
    for start in range(len(tokens)):
        candidate_text = ""
        for token in tokens[start:start + 12]:
            candidate_text += token
            if not 5 <= len(candidate_text) <= 12:
                continue
            normalized = _standard_candidate(candidate_text)
            if normalized is not None:
                candidates.add(normalized)
    for match in _BH_PATTERN.finditer(text):
        state, number, series = match.groups()
        normalized_state, state_fixes = _corrected(state, _DIGIT_OCR_CORRECTIONS)
        normalized_number, number_fixes = _corrected(
            number, _DIGIT_OCR_CORRECTIONS
        )
        normalized = normalized_state + "BH" + normalized_number + series
        if _BH_REGISTRATION.fullmatch(normalized):
            candidates.add(normalized)
    return candidates


def _standard_candidate(candidate_text: str) -> str | None:
    if len(candidate_text) < 5:
        return None
    trailing_digits = re.search(r"[0-9]+$", candidate_text)
    if trailing_digits is not None and len(trailing_digits.group(0)) > 4:
        return None
    state, _state_fixes = _corrected(
        candidate_text[:2], _LETTER_OCR_CORRECTIONS
    )
    remaining = candidate_text[2:]
    rto_length = 0
    for character in remaining[:2]:
        if character not in "0123456789OQILBS":
            break
        rto_length += 1
    if not rto_length:
        return None
    rto, _rto_fixes = _corrected(
        remaining[:rto_length], _DIGIT_OCR_CORRECTIONS
    )
    remainder = remaining[rto_length:]
    number_length = 0
    for character in reversed(remainder[-4:]):
        if character not in "0123456789OQILBS":
            break
        number_length += 1
    if number_length > 4:
        return None
    if not 1 <= number_length <= 4:
        return None
    series, _series_fixes = _corrected(
        remainder[:-number_length], _LETTER_OCR_CORRECTIONS
    )
    number, _number_fixes = _corrected(
        remainder[-number_length:], _DIGIT_OCR_CORRECTIONS
    )
    normalized = state + rto + series + number
    return normalized if _STANDARD_REGISTRATION.fullmatch(normalized) else None


def normalize_registration_candidate(raw_text: str | None) -> str | None:
    """Return one structurally valid registration candidate from OCR text."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    candidates = _candidate_set(raw_text)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and 0 <= parsed <= 1 else None


def build_registration_ocr_assessment(ocr_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize OCR candidates and require confidence when reads conflict."""
    evidence: list[dict[str, Any]] = []
    latest_by_image: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(ocr_results):
        raw_text = result.get("text")
        confidence = result.get("confidence")
        candidates = (
            _candidate_set(raw_text) if isinstance(raw_text, str) else set()
        )
        candidate = next(iter(candidates)) if len(candidates) == 1 else None
        image_id = result.get("image_id")
        evidence.append(
            {
                "image_id": image_id,
                "text": raw_text,
                "confidence": confidence,
                "normalized_candidate": candidate,
                "status": result.get("status"),
            }
        )
        key = str(image_id) if image_id is not None else f"unlinked:{index}"
        latest_by_image[key] = {
            **result,
            "normalized_candidate": candidate,
            "candidate_ambiguous": len(candidates) > 1,
        }

    completed = [
        item for item in latest_by_image.values()
        if item.get("status") == "completed"
    ]
    candidates = [
        item for item in completed if item.get("normalized_candidate")
    ]
    all_distinct = {item["normalized_candidate"] for item in candidates}
    confidence_by_candidate: dict[str, float] = {}
    for item in candidates:
        confidence = _confidence(item.get("confidence"))
        candidate = item["normalized_candidate"]
        if confidence is not None:
            confidence_by_candidate[candidate] = max(
                confidence_by_candidate.get(candidate, -1.0), confidence
            )

    selected: str | None = None
    confidence: float | None = None
    if len(all_distinct) == 1:
        selected = next(iter(all_distinct))
        confidence = confidence_by_candidate.get(selected)
    elif len(confidence_by_candidate) == 1 and not all_distinct - set(confidence_by_candidate):
        selected = next(iter(confidence_by_candidate))
        confidence = confidence_by_candidate[selected]
    elif len(confidence_by_candidate) > 1:
        ranked = sorted(
            confidence_by_candidate.items(), key=lambda item: item[1], reverse=True
        )
        if (
            ranked[0][1] >= REGISTRATION_OCR_MIN_CONFIDENCE
            and ranked[0][1] > ranked[1][1]
        ):
            selected, confidence = ranked[0]

    has_readable_text = any(
        isinstance(item.get("text"), str) and item["text"].strip()
        for item in completed
    )
    ambiguous_read = any(item.get("candidate_ambiguous") for item in completed)
    if selected is not None and confidence is not None:
        status = (
            "candidate"
            if confidence >= REGISTRATION_OCR_MIN_CONFIDENCE
            else "needs_review"
        )
    elif candidates or ambiguous_read or has_readable_text:
        status = "needs_review"
        selected = None if len(all_distinct) > 1 or ambiguous_read else selected
        confidence = confidence_by_candidate.get(selected) if selected else None
    else:
        status = "unavailable"
        selected = None
        confidence = None

    return {
        "status": status,
        "normalized_candidate": selected,
        "confidence": confidence,
        "minimum_confidence": REGISTRATION_OCR_MIN_CONFIDENCE,
        "raw_evidence": evidence,
        "note": (
            "OCR provides an unverified candidate only. Confirm the registration number "
            "against the vehicle documents or plate before using it for a lookup."
        ),
    }
