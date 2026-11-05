from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

CLASSIFIER_VERSION = "2"
IMAGE_TYPES = {
    "registration_plate", "vehicle_exterior", "vehicle_interior",
    "damage_closeup", "document", "unknown",
}


@dataclass(frozen=True)
class ImageEligibilityDecision:
    image_type: str
    classification_method: str
    eligible_for_damage_inference: bool
    rejection_reason: str | None
    classified_at: datetime
    signals: dict[str, Any]
    cache_key: str
    classification_source: str = "automatic"

    def as_metadata(self) -> dict[str, Any]:
        return {
            "image_type": self.image_type,
            "classification_method": self.classification_method,
            "classification_source": self.classification_source,
            "damage_eligible": self.eligible_for_damage_inference,
            "damage_rejection_reason": self.rejection_reason,
            "classified_at": self.classified_at.isoformat(),
            "signals": self.signals,
            "cache_key": self.cache_key,
            "classifier_version": CLASSIFIER_VERSION,
        }
def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _image_dimensions(content: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(content)) as image:
        image.verify()
    with Image.open(BytesIO(content)) as image:
        return int(image.width), int(image.height)


def _ocr_texts(ocr_results: Iterable[Any] | None) -> list[str]:
    texts: list[str] = []
    for result in ocr_results or []:
        raw_text = getattr(result, "raw_text", None)
        if raw_text is None and isinstance(result, dict):
            raw_text = result.get("raw_text") or result.get("text")
        text = _clean_text(raw_text)
        if text:
            texts.append(text)
    return texts


def _ocr_result_signature(ocr_results: Iterable[Any] | None) -> str:
    parts: list[str] = []
    for result in ocr_results or []:
        result_id = getattr(result, "id", None)
        if result_id is None and isinstance(result, dict):
            result_id = result.get("id")
        raw_text = getattr(result, "raw_text", None)
        if raw_text is None and isinstance(result, dict):
            raw_text = result.get("raw_text") or result.get("text")
        confidence = getattr(result, "confidence", None) if not isinstance(result, dict) else result.get("confidence")
        created_at = getattr(result, "created_at", None) if not isinstance(result, dict) else result.get("created_at")
        parts.append(f"{result_id}|{_clean_text(raw_text)}|{confidence}|{created_at}")
    return hashlib.sha256("\n".join(sorted(parts)).encode("utf-8")).hexdigest()


def _content_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _plate_like(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text.upper())
    patterns = [
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}",
        r"[A-Z]{2}\d{2}[A-Z]{2}\d{4}",
    ]
    return any(re.search(pattern, normalized) for pattern in patterns)


def _document_like(text: str) -> bool:
    lowered = text.casefold()
    keywords = (
        "registration certificate",
        "certificate of registration",
        "insurance",
        "invoice",
        "pollution",
        "fitness",
        "puc",
        "policy",
        "challan",
    )
    return any(keyword in lowered for keyword in keywords)


def _filename_hints(filename: str | None) -> set[str]:
    lowered = (filename or "").casefold()
    hints: set[str] = set()
    if any(token in lowered for token in ("plate", "number plate", "reg", "rc")):
        hints.add("plate")
    if any(token in lowered for token in ("document", "insurance", "invoice", "certificate", "policy", "puc")):
        hints.add("document")
    if any(token in lowered for token in ("vehicle", "car", "exterior", "front", "rear", "side", "damage", "photo")):
        hints.add("vehicle")
    if any(token in lowered for token in ("closeup", "close-up", "detail", "zoom")):
        hints.add("closeup")
    if any(token in lowered for token in ("interior", "cabin", "dashboard")):
        hints.add("interior")
    return hints
def classify_image(
    *,
    content: bytes,
    ocr_results: Iterable[Any] | None = None,
    original_filename: str | None = None,
    selected_type: str | None = None,
) -> ImageEligibilityDecision:
    classified_at = datetime.now(timezone.utc)
    content_hash = _content_sha256(content)
    ocr_signature = _ocr_result_signature(ocr_results)
    cache_key = hashlib.sha256(f"{content_hash}|{ocr_signature}|{CLASSIFIER_VERSION}".encode("utf-8")).hexdigest()

    if selected_type is not None and selected_type not in IMAGE_TYPES:
        raise ValueError("Unsupported image category")

    try:
        width, height = _image_dimensions(content)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        return ImageEligibilityDecision(
            image_type="unknown",
            classification_method="image_decode_failed",
            eligible_for_damage_inference=False,
            rejection_reason=f"classification_failed: {type(exc).__name__}",
            classified_at=classified_at,
            signals={"error": str(exc)},
            cache_key=cache_key,
        )

    if selected_type is not None:
        eligible = selected_type in {"vehicle_exterior", "vehicle_interior", "damage_closeup"}
        return ImageEligibilityDecision(
            image_type=selected_type,
            classification_method="explicit_user_selection",
            eligible_for_damage_inference=eligible,
            rejection_reason=None if eligible else (
                "image_category_not_eligible_for_damage_inference"
                if selected_type != "unknown" else "classification_uncertain"
            ),
            classified_at=classified_at,
            signals={"width": width, "height": height, "category_selected_by_user": True},
            cache_key=cache_key,
            classification_source="user_input",
        )

    aspect_ratio = width / height if height else 0.0
    texts = _ocr_texts(ocr_results)
    combined_text = "\n".join(texts)
    plate_text = _plate_like(combined_text)
    document_text = _document_like(combined_text)
    hints = _filename_hints(original_filename)

    signals: dict[str, Any] = {
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 4),
        "ocr_text_present": bool(combined_text),
        "plate_like_ocr": plate_text,
        "document_like_ocr": document_text,
        "filename_hints": sorted(hints),
    }

    plate_dimensions = width <= 400 and height <= 200 and 1.8 <= aspect_ratio <= 3.2
    if "plate" in hints or (plate_text and plate_dimensions):
        if plate_dimensions or "plate" in hints:
            return ImageEligibilityDecision(
                image_type="registration_plate",
                classification_method="dimensions+ocr_plate_text+filename" if plate_text else "dimensions+filename",
                eligible_for_damage_inference=False,
                rejection_reason="registration_plate_not_eligible_for_damage_inference",
                classified_at=classified_at,
                signals=signals,
                cache_key=cache_key,
            )

    if "document" in hints or document_text:
        return ImageEligibilityDecision(
            image_type="document",
            classification_method="ocr_document_text+filename" if document_text else "filename",
            eligible_for_damage_inference=False,
            rejection_reason="document_not_eligible_for_damage_inference",
            classified_at=classified_at,
            signals=signals,
            cache_key=cache_key,
        )

    if "interior" in hints:
        return ImageEligibilityDecision(
            image_type="vehicle_interior",
            classification_method="filename",
            eligible_for_damage_inference=False,
            rejection_reason="interior_not_supported_for_exterior_damage",
            classified_at=classified_at,
            signals=signals,
            cache_key=cache_key,
        )

    closeup_hint = "closeup" in hints
    vehicle_hint = "vehicle" in hints
    sufficiently_large = width >= 320 and height >= 240
    reasonable_aspect = 0.5 <= aspect_ratio <= 2.5

    if vehicle_hint or closeup_hint:
        if closeup_hint and not vehicle_hint:
            image_type = "vehicle_closeup"
        elif width < 800 and height < 600 and closeup_hint:
            image_type = "vehicle_closeup"
        else:
            image_type = "vehicle_exterior"
        if image_type == "vehicle_exterior":
            eligible = True
        else:
            eligible = sufficiently_large and reasonable_aspect and width >= 320 and height >= 240
        return ImageEligibilityDecision(
            image_type=image_type,
            classification_method="filename+dimensions",
            eligible_for_damage_inference=eligible,
            rejection_reason=None if eligible else "low_resolution_or_unsupported_image",
            classified_at=classified_at,
            signals=signals,
            cache_key=cache_key,
        )

    return ImageEligibilityDecision(
        image_type="unknown",
        classification_method="insufficient_evidence",
        eligible_for_damage_inference=False,
        rejection_reason="insufficient_classification_evidence",
        classified_at=classified_at,
        signals=signals,
        cache_key=cache_key,
        classification_source="automatic",
    )
def classification_cache_key(content: bytes, ocr_results: Iterable[Any] | None = None) -> str:
    content_hash = _content_sha256(content)
    ocr_signature = _ocr_result_signature(ocr_results)
    return hashlib.sha256(f"{content_hash}|{ocr_signature}|{CLASSIFIER_VERSION}".encode("utf-8")).hexdigest()


def cached_decision(image: Any, content: bytes, ocr_results: Iterable[Any] | None = None) -> ImageEligibilityDecision | None:
    metadata = getattr(image, "image_metadata", None) or {}
    cached = metadata.get("damage_eligibility")
    if not isinstance(cached, dict):
        return None
    expected_key = classification_cache_key(content, ocr_results)
    if cached.get("cache_key") != expected_key or cached.get("classifier_version") != CLASSIFIER_VERSION:
        return None
    try:
        classified_at = datetime.fromisoformat(str(cached.get("classified_at")).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if classified_at.tzinfo is None:
        classified_at = classified_at.replace(tzinfo=timezone.utc)
    return ImageEligibilityDecision(
        image_type=str(cached.get("image_type") or "unknown"),
        classification_method=str(cached.get("classification_method") or "cached"),
        eligible_for_damage_inference=bool(cached.get("damage_eligible")),
        rejection_reason=cached.get("damage_rejection_reason"),
        classified_at=classified_at,
        signals=dict(cached.get("signals") or {}),
        cache_key=expected_key,
        classification_source=str(cached.get("classification_source") or "automatic"),
    )


def store_classification(image: Any, decision: ImageEligibilityDecision) -> None:
    metadata = dict(getattr(image, "image_metadata", None) or {})
    metadata["damage_eligibility"] = decision.as_metadata()
    image.image_metadata = metadata
