from io import BytesIO

from PIL import Image
import pytest

from app.images.classification import (
    cached_decision,
    classify_image,
    classification_cache_key,
)


def png_bytes(size=(100, 100), color=(200, 200, 200)):
    output = BytesIO()
    Image.new("RGB", size, color=color).save(output, format="PNG")
    return output.getvalue()


def webp_bytes(size=(1280, 960), color=(200, 200, 200)):
    output = BytesIO()
    Image.new("RGB", size, color=color).save(output, format="WEBP")
    return output.getvalue()


def test_plate_crop_classified_as_registration_plate():
    content = png_bytes(size=(141, 59))
    decision = classify_image(content=content, original_filename="plate.png")
    assert decision.image_type == "registration_plate"
    assert decision.eligible_for_damage_inference is False


def test_ocr_plate_text_alone_does_not_classify_large_image():
    class OCR:
        raw_text = "RJ14AB1234"
        id = "ocr-1"

    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content, ocr_results=[OCR()], original_filename="vehicle.jpg")
    # filename vehicle hint makes it exterior; OCR plate text alone does not make plate
    assert decision.image_type == "vehicle_exterior"
    assert decision.eligible_for_damage_inference is True


def test_document_ocr_classified_as_document():
    class OCR:
        raw_text = "Registration Certificate"
        id = "ocr-2"

    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content, ocr_results=[OCR()])
    assert decision.image_type == "document"
    assert decision.eligible_for_damage_inference is False


def test_large_generic_photo_is_admitted_as_vehicle_exterior_candidate():
    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content)
    assert decision.image_type == "vehicle_exterior"
    assert decision.eligible_for_damage_inference is True
    assert decision.rejection_reason is None
    assert decision.signals.get("heuristic_vehicle_candidate") is True


def test_small_generic_photo_remains_unknown_and_ineligible():
    content = png_bytes(size=(200, 150))
    decision = classify_image(content=content)
    assert decision.image_type == "unknown"
    assert decision.eligible_for_damage_inference is False
    assert decision.rejection_reason == "insufficient_classification_evidence"


def test_extreme_aspect_large_image_is_not_admitted():
    # Wide panorama-like aspect: not a plausible ordinary vehicle photo.
    content = png_bytes(size=(800, 100))
    decision = classify_image(content=content)
    assert decision.image_type == "unknown"
    assert decision.eligible_for_damage_inference is False


def test_generic_filename_large_no_ocr_is_eligible():
    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content, original_filename="IMG_1234.jpg")
    assert decision.image_type == "vehicle_exterior"
    assert decision.eligible_for_damage_inference is True


def test_production_generic_webp_exterior_is_eligible():
    # Production report scenario: clear front-facing exterior photo named
    # IMG_1234.webp, no OCR signals, sufficiently large.
    content = webp_bytes(size=(1280, 960))
    decision = classify_image(content=content, original_filename="IMG_1234.webp")
    assert decision.image_type == "vehicle_exterior"
    assert decision.classification_method == "dimensions_only_conservative"
    assert decision.classification_source == "automatic"
    assert decision.eligible_for_damage_inference is True
    assert decision.rejection_reason is None
    assert decision.signals.get("heuristic_vehicle_candidate") is True


def test_production_rejection_message_only_from_explicit_ineligible_selection():
    # The exact production rejection reason only arises from the explicit
    # selected_type branch, never from the generic auto-classification path.
    content = webp_bytes(size=(1280, 960))
    generic = classify_image(content=content, original_filename="IMG_1234.webp")
    assert generic.rejection_reason is None
    explicit = classify_image(content=content, selected_type="vehicle_interior")
    assert explicit.rejection_reason == "image_category_not_eligible_for_damage_inference"


def test_descriptive_vehicle_filename_preserves_existing_behavior():
    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content, original_filename="vehicle_front.jpg")
    assert decision.image_type == "vehicle_exterior"
    assert decision.eligible_for_damage_inference is True
    assert decision.classification_method == "filename+dimensions"


def test_closeup_low_resolution_ineligible():
    content = png_bytes(size=(200, 150))
    decision = classify_image(content=content, original_filename="closeup.png")
    assert decision.image_type == "vehicle_closeup"
    assert decision.eligible_for_damage_inference is False


def test_stale_cache_invalidated():
    content = png_bytes(size=(100, 100))
    class FakeImage:
        image_metadata = None

    image = FakeImage()
    first = classify_image(content=content, original_filename="vehicle.png")
    from app.images.classification import store_classification

    store_classification(image, first)
    assert cached_decision(image, content, None) is not None

    changed = png_bytes(size=(101, 101))
    assert cached_decision(image, changed, None) is None


@pytest.mark.parametrize("category", ["vehicle_exterior", "damage_closeup"])
def test_explicit_eligible_category_does_not_depend_on_filename_or_ocr(category):
    decision = classify_image(content=png_bytes(size=(415, 378)), selected_type=category)
    assert decision.image_type == category
    assert decision.classification_source == "user_input"
    assert decision.eligible_for_damage_inference is True
    assert decision.rejection_reason is None


@pytest.mark.parametrize("category", ["registration_plate", "document", "vehicle_interior"])
def test_explicit_ineligible_categories_are_excluded(category):
    decision = classify_image(content=png_bytes(), selected_type=category)
    assert decision.image_type == category
    assert decision.eligible_for_damage_inference is False
    assert decision.classification_source == "user_input"


def test_explicit_unknown_remains_classification_uncertain():
    decision = classify_image(content=png_bytes(), selected_type="unknown")
    assert decision.image_type == "unknown"
    assert decision.eligible_for_damage_inference is False
    assert decision.rejection_reason == "classification_uncertain"
