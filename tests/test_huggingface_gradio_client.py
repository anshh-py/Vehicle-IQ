import json
from types import SimpleNamespace

import httpx
import pytest

from app.ocr.client import (
    BaiduUnlimitedOCRClient,
    HuggingFaceGradioOCRClient,
    OCRInferenceFailed,
    OCRInvalidResponse,
    OCRRecognition,
    OCRServiceUnavailable,
    OCRUnreadableText,
    PaddleOCRLocalClient,
)


BASE_URL = "https://baidu-unlimited-ocr.hf.space"
SSE_RESULT = (
    'event: generating\ndata: [{"text":"partial","done":false}]\n\n'
    'event: complete\ndata: [{"text":"RJ45CA1879","done":true}]\n\n'
)


def test_public_space_upload_queue_and_sse_result_flow():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/gradio_api/upload":
            body = request.read()
            assert request.method == "POST"
            assert b'name="files"' in body
            assert b"vehicle.png" in body
            assert b"image/png" in body
            return httpx.Response(200, json=["/tmp/gradio/abc/vehicle.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            assert request.method == "POST"
            assert json.loads(request.read()) == {
                "image_path": {
                    "path": "/tmp/gradio/abc/vehicle.png",
                    "meta": {"_type": "gradio.FileData"},
                },
                "mode": "gundam",
                "prompt": HuggingFaceGradioOCRClient.PROMPT,
            }
            return httpx.Response(200, json={"event_id": "event-123"})
        if request.url.path == "/gradio_api/call/run_ocr/event-123":
            assert request.method == "GET"
            return httpx.Response(200, text=SSE_RESULT, headers={"content-type": "text/event-stream"})
        raise AssertionError(f"Unexpected mocked request: {request.method} {request.url.path}")

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    result = client.recognize(b"png bytes", "vehicle.png", "image/png")

    assert result == OCRRecognition(text="RJ45CA1879")
    assert client.provider_name == "huggingface_gradio"
    assert len(seen) == 3
    assert all(request.url.host == "baidu-unlimited-ocr.hf.space" for request in seen)


@pytest.mark.parametrize("upload_payload", [[], [""], [None], {"path": "/tmp/image.png"}, ["a", "b"]])
def test_public_space_rejects_malformed_or_empty_upload_response(upload_payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=upload_payload)

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(OCRInvalidResponse):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_rejects_non_json_upload_response():
    client = HuggingFaceGradioOCRClient(
        BASE_URL,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"not json", headers={"content-type": "application/json"})
        ),
    )
    with pytest.raises(OCRInvalidResponse):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_rejects_missing_event_id():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        return httpx.Response(200, json={})

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(OCRInvalidResponse, match="no event ID"):
        client.recognize(b"image", "image.png", "image/png")


@pytest.mark.parametrize("status_code", [429, 503])
def test_public_space_maps_http_service_failures(status_code):
    client = HuggingFaceGradioOCRClient(
        BASE_URL,
        transport=httpx.MockTransport(lambda _request: httpx.Response(status_code)),
    )
    with pytest.raises(OCRServiceUnavailable):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_maps_network_failure_without_transport_details():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private network detail", request=request)

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(fail))
    with pytest.raises(OCRServiceUnavailable) as error:
        client.recognize(b"image", "image.png", "image/png")
    assert "private network detail" not in str(error.value)


@pytest.mark.parametrize(
    ("sse_body", "error_type"),
    [
        ("event: complete\ndata: invalid json\n\n", OCRInvalidResponse),
        ("event: generating\ndata: []\n\n", OCRInvalidResponse),
        ("event: error\ndata: inference failed\n\n", OCRInferenceFailed),
        ("event: complete\ndata: [{\"text\":\"x\",\"done\":false}]\n\n", OCRInvalidResponse),
    ],
)
def test_public_space_maps_malformed_sse_and_inference_errors(sse_body, error_type):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        return httpx.Response(200, text=sse_body, headers={"content-type": "text/event-stream"})

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(error_type):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_preserves_safe_upstream_inference_error_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        return httpx.Response(
            200,
            text='event: error\ndata: "CUDA worker is unavailable; token=secret https://private.example/x"\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(OCRInferenceFailed) as error:
        client.recognize(b"image", "image.png", "image/png")
    assert "CUDA worker is unavailable" in str(error.value)
    assert "token=secret" not in str(error.value)
    assert "private.example" not in str(error.value)


def test_public_space_empty_final_text_is_unreadable():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        return httpx.Response(
            200,
            text='event: complete\ndata: [{"text":"  ","done":true}]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(OCRUnreadableText):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_no_text_marker_is_unreadable():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        text = (
            "The English text in the image is not clearly visible.\n\n"
            "[No text detected]\n![](images/0.jpg)"
        )
        output = json.dumps([{"text": text, "done": True}])
        return httpx.Response(
            200,
            text=f"event: complete\ndata: {output}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    with pytest.raises(OCRUnreadableText):
        client.recognize(b"image", "image.png", "image/png")


def test_public_space_markdown_image_reference_is_not_returned_as_ocr_text():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        output = json.dumps([{"text": "RJ45CA1879\n![](images/0.jpg)", "done": True}])
        return httpx.Response(
            200,
            text=f"event: complete\ndata: {output}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    client = HuggingFaceGradioOCRClient(BASE_URL, transport=httpx.MockTransport(handler))
    assert client.recognize(b"image", "image.png", "image/png").text == "RJ45CA1879"


def test_public_space_base_mode_is_configurable():
    seen_body = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            seen_body.update(json.loads(request.read()))
            return httpx.Response(200, json={"event_id": "job"})
        return httpx.Response(
            200,
            text='event: complete\ndata: [{"text":"text","done":true}]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    client = HuggingFaceGradioOCRClient(BASE_URL, mode="base", transport=httpx.MockTransport(handler))
    assert client.recognize(b"image", "image.png", "image/png").text == "text"
    assert seen_body["mode"] == "base"


def test_public_space_oversized_response_is_rejected_safely(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "max_external_response_bytes", 16)
    client = HuggingFaceGradioOCRClient(
        BASE_URL,
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=b"sensitive-upstream-body")
        ),
    )

    with pytest.raises(OCRInvalidResponse) as error:
        client.recognize(b"image", "vehicle.png", "image/png")

    assert "exceeded the allowed size" in str(error.value)
    assert "sensitive-upstream-body" not in str(error.value)


def test_provider_selection_defaults_and_keeps_local_baidu_provider(monkeypatch):
    import app.api.v1.ocr as ocr_api

    monkeypatch.setattr(
        ocr_api,
        "settings",
        SimpleNamespace(
            ocr_service_provider="huggingface_gradio",
            ocr_service_url=None,
            ocr_gradio_space_url=BASE_URL,
            ocr_gradio_mode="base",
        ),
    )
    public_client = ocr_api.get_ocr_client()
    assert isinstance(public_client, HuggingFaceGradioOCRClient)
    assert public_client.base_url == f"{BASE_URL}/"
    assert public_client.mode == "base"

    monkeypatch.setattr(
        ocr_api,
        "settings",
        SimpleNamespace(
            ocr_service_provider="baidu_unlimited_ocr",
            ocr_service_url="http://local-ocr:7860",
            ocr_gradio_space_url=BASE_URL,
            ocr_gradio_mode="gundam",
        ),
    )
    local_client = ocr_api.get_ocr_client()
    assert isinstance(local_client, BaiduUnlimitedOCRClient)
    assert local_client.endpoint == "http://local-ocr:7860/ocr"

    monkeypatch.setattr(
        ocr_api,
        "settings",
        SimpleNamespace(
            ocr_service_provider="paddleocr_local",
            ocr_paddle_detection_model_dir=".local-storage/detector",
            ocr_paddle_recognition_model_dir=".local-storage/recognizer",
        ),
    )
    paddle_client = ocr_api.get_ocr_client()
    assert isinstance(paddle_client, PaddleOCRLocalClient)
    assert paddle_client.detection_model_dir.name == "detector"
    assert paddle_client.recognition_model_dir.name == "recognizer"
