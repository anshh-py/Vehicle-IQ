import time

import pytest

from app.ocr.client import (
    AnprGradioOCRClient,
    OCRInvalidResponse,
    OCRNoPlateDetected,
    OCRRecognition,
    OCRServiceUnavailable,
    OCRUnreadableText,
)
from app.ocr.registration import build_registration_ocr_assessment


class FakeGradioClient:
    def __init__(self, result, error=None, delay=0.0):
        self.result = result
        self.error = error
        self.delay = delay
        self.predict_calls = []

    def predict(self, file_data, api_name=None):
        self.predict_calls.append((file_data, api_name))
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result

    def close(self):
        pass


class FakeFactory:
    def __init__(self, client):
        self.client = client

    def __call__(self, space_url):
        return self.client


def make_client(result=None, error=None, delay=0.0, timeout=30.0, **kwargs):
    fake = FakeGradioClient(result if result is not None else ({}, "Biển số: R18UC3721"), error=error, delay=delay)
    return AnprGradioOCRClient(
        "https://hung102004-demo-licenseplate.hf.space",
        timeout_seconds=timeout,
        client_factory=FakeFactory(fake),
        **kwargs,
    ), fake


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_successful_extraction_with_vietnamese_prefix():
    client, fake = make_client(({"path": "/tmp/crop.jpg"}, "Biển số: R18UC3721"))
    result = client.recognize(PNG, "vehicle.png", "image/png")
    assert result == OCRRecognition(text="R18UC3721", provider="anpr_gradio")
    assert fake.predict_calls[0][1] == "/recognize_license_plate"


def test_successful_extraction_plain_plate():
    client, _fake = make_client(({"path": "/tmp/crop.jpg"}, "MH12UC4321"))
    result = client.recognize(PNG, "vehicle.png", "image/png")
    assert result.text == "MH12UC4321"


def test_whitespace_and_lowercase_normalization():
    client, _fake = make_client(({"path": "/tmp/crop.jpg"}, "  biển số:  mh 12uc 4321  "))
    result = client.recognize(PNG, "vehicle.png", "image/png")
    assert result.text == "MH 12UC 4321"


def test_no_plate_detected_raises_no_plate():
    client, _fake = make_client((None, "Không phát hiện được biển số"))
    with pytest.raises(OCRNoPlateDetected):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_no_plate_english_variant():
    client, _fake = make_client((None, "No license plate detected"))
    with pytest.raises(OCRNoPlateDetected):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_unreadable_plate_raises_unreadable():
    client, _fake = make_client(({"path": "/tmp/crop.jpg"}, "Biển số: ???"))
    with pytest.raises(OCRUnreadableText):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_malformed_non_string_text_raises_invalid_response():
    client, _fake = make_client(({"path": "/tmp/crop.jpg"}, 12345))
    with pytest.raises(OCRInvalidResponse):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_malformed_wrong_shape_raises_invalid_response():
    client, _fake = make_client("just-a-string")
    with pytest.raises(OCRInvalidResponse):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_space_unavailable_maps_to_service_unavailable():
    client, _fake = make_client(error=ConnectionError("boom"), timeout=10.0)
    with pytest.raises(OCRServiceUnavailable):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_timeout_maps_to_service_unavailable():
    client, fake = make_client(delay=5.0, timeout=0.5)
    with pytest.raises(OCRServiceUnavailable):
        client.recognize(PNG, "vehicle.png", "image/png")


def test_retry_transient_failure_then_success():
    calls = {"n": 0}

    class RetryFactory:
        def __call__(self, space_url):
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeGradioClient(error=ConnectionError("cold start"))
            return FakeGradioClient(({"path": "/tmp/crop.jpg"}, "Biển số: R18UC3721"))

    client = AnprGradioOCRClient(
        "https://hung102004-demo-licenseplate.hf.space",
        timeout_seconds=10.0,
        max_retries=2,
        client_factory=RetryFactory(),
    )
    result = client.recognize(PNG, "vehicle.png", "image/png")
    assert result.text == "R18UC3721"
    assert calls["n"] == 2


def test_unsupported_content_type_rejected():
    client, fake = make_client()
    with pytest.raises(OCRInvalidResponse):
        client.recognize(PNG, "vehicle.txt", "text/plain")
    assert fake.predict_calls == []


def test_registration_handoff_from_anpr_result():
    assessment = build_registration_ocr_assessment(
        [
            {
                "image_id": "img-1",
                "text": "R18UC3721",
                "confidence": None,
                "status": "completed",
            }
        ]
    )
    # Without a provider confidence score the candidate is surfaced for human
    # review rather than auto-selected - safe confirmation flow.
    assert assessment["status"] == "needs_review"
    assert assessment["normalized_candidate"] is not None


def test_manual_registration_fallback_when_no_plate():
    assessment = build_registration_ocr_assessment(
        [
            {
                "image_id": "img-1",
                "text": None,
                "confidence": None,
                "status": "no_plate",
            }
        ]
    )
    assert assessment["status"] == "unavailable"
    assert assessment["normalized_candidate"] is None
