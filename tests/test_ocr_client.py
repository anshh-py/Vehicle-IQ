import httpx
import pytest

from app.ocr.client import (
    BaiduUnlimitedOCRClient,
    OCRInvalidResponse,
    OCRServiceUnavailable,
    OCRUnreadableText,
)


def test_baidu_client_uses_extracted_multipart_contract_and_parses_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["content_type"] = request.headers["content-type"]
        seen["body"] = request.read()
        return httpx.Response(
            200,
            json={"success": True, "filename": "image.png", "result": "  KA 01 AB 1234  "},
        )

    client = BaiduUnlimitedOCRClient("http://ocr.test:7860/", transport=httpx.MockTransport(handler))
    result = client.recognize(b"image bytes", "image.png", "image/png")

    assert seen["path"] == "/ocr"
    assert "multipart/form-data" in seen["content_type"]
    assert b'name="file"' in seen["body"]
    assert b"image.png" in seen["body"]
    assert result.text == "KA 01 AB 1234"
    assert result.confidence is None


def test_baidu_client_rejects_invalid_response_shape():
    client = BaiduUnlimitedOCRClient(
        "http://ocr.test", transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"text": "x"}))
    )

    with pytest.raises(OCRInvalidResponse):
        client.recognize(b"image", "image.png", "image/png")


def test_baidu_client_maps_unavailable_service():
    client = BaiduUnlimitedOCRClient(
        "http://ocr.test", transport=httpx.MockTransport(lambda _request: httpx.Response(503, json={"detail": "secret"}))
    )

    with pytest.raises(OCRServiceUnavailable):
        client.recognize(b"image", "image.png", "image/png")


def test_baidu_client_maps_timeout_without_exposing_transport_details():
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private transport detail", request=request)

    client = BaiduUnlimitedOCRClient("http://ocr.test", transport=httpx.MockTransport(timeout))

    with pytest.raises(OCRServiceUnavailable) as error:
        client.recognize(b"image", "image.png", "image/png")

    assert "private transport detail" not in str(error.value)


def test_baidu_client_treats_service_sentinel_as_unreadable():
    client = BaiduUnlimitedOCRClient(
        "http://ocr.test",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"success": True, "filename": "image.png", "result": "No output found"}
            )
        ),
    )

    with pytest.raises(OCRUnreadableText):
        client.recognize(b"image", "image.png", "image/png")


def test_baidu_client_rejects_non_json_response():
    client = BaiduUnlimitedOCRClient(
        "http://ocr.test",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})
        ),
    )

    with pytest.raises(OCRInvalidResponse):
        client.recognize(b"image", "image.png", "image/png")
