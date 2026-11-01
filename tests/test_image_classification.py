from io import BytesIO

from PIL import Image

from app.images.classification import (
    cached_decision,
    classify_image,
    classification_cache_key,
)


def png_bytes(size=(100, 100), color=(200, 200, 200)):
    output = BytesIO()
    Image.new("RGB", size, color=color).save(output, format="PNG")
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


def test_large_dimensions_alone_do_not_establish_vehicle_exterior():
    content = png_bytes(size=(800, 600))
    decision = classify_image(content=content)
    assert decision.image_type == "unknown"
    assert decision.eligible_for_damage_inference is False


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
