from collections.abc import Generator
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.v1.ocr import get_ocr_client
from app.db.base import Base
from app.db.models import OCRResult, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.ocr.client import (
    HuggingFaceGradioOCRClient,
    OCRInferenceFailed,
    OCRInvalidResponse,
    OCRRecognition,
    OCRServiceUnavailable,
    OCRUnreadableText,
    PaddleOCRLocalClient,
)
from app.storage.images import LocalImageStorage, get_image_storage


def valid_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 3), color=(80, 120, 160)).save(output, format="PNG")
    return output.getvalue()


class FakeOCRClient:
    provider_name = "baidu_unlimited_ocr"

    def __init__(self, result=None, error: Exception | None = None):
        self.result = result or OCRRecognition(text="some visible text")
        self.error = error
        self.calls = []

    def recognize(self, content: bytes, filename: str, content_type: str) -> OCRRecognition:
        self.calls.append((content, filename, content_type))
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def ocr_api(tmp_path) -> Generator[tuple[TestClient, str, str, LocalImageStorage, object, FakeOCRClient], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    storage = LocalImageStorage(tmp_path / "ocr-images")
    content = valid_png()
    image_key = storage.save(content, "png")
    with Session(engine) as session:
        assessment = VehicleAssessment()
        session.add(assessment)
        session.flush()
        image = UploadedImage(
            assessment_id=assessment.id,
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key=image_key,
            original_filename="vehicle.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="test-sha256",
        )
        session.add(image)
        session.commit()
        assessment_id, image_id = str(assessment.id), str(image.id)

    fake = FakeOCRClient()

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_image_storage] = lambda: storage
    app.dependency_overrides[get_ocr_client] = lambda: fake
    try:
        with TestClient(app) as client:
            yield client, assessment_id, image_id, storage, engine, fake
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def count_ocr_results(engine) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(OCRResult)) or 0


def test_ocr_success_persists_output_and_omits_missing_confidence(ocr_api):
    client, assessment_id, image_id, _storage, engine, fake = ocr_api
    fake.result = OCRRecognition(text="Text read from the image")

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 200
    assert response.json()["text"] == "Text read from the image"
    assert response.json()["provider"] == "baidu_unlimited_ocr"
    assert "confidence" not in response.json()
    assert len(fake.calls) == 1
    assert fake.calls[0][1:] == ("image.png", "image/png")
    assert count_ocr_results(engine) == 1
    with Session(engine) as session:
        record = session.scalar(select(OCRResult))
        assert record is not None
        assert record.raw_text == "Text read from the image"
        assert record.result_type == "image_ocr"
        assert record.status == "completed"


def _make_local_paddle_model_dirs(root):
    detection = root / "detector"
    recognition = root / "recognizer"
    for model_dir in (detection, recognition):
        model_dir.mkdir(parents=True)
        (model_dir / "inference.yml").write_text("model: test", encoding="utf-8")
        (model_dir / "inference.json").write_text("{}", encoding="utf-8")
        (model_dir / "inference.pdiparams").write_bytes(b"test-weights")
    return detection, recognition


def test_paddle_local_client_reuses_model_and_returns_model_confidence(tmp_path, monkeypatch):
    detection, recognition = _make_local_paddle_model_dirs(tmp_path)
    calls = {"load": 0, "predict": 0}

    class FakeModel:
        def predict(self, image_path):
            calls["predict"] += 1
            assert Path(image_path).is_file()
            return [SimpleNamespace(json={"res": {"rec_texts": ["RJ45CA1879"], "rec_scores": [0.9977]}})]

    monkeypatch.setattr(PaddleOCRLocalClient, "_model_cache", {})

    def build_model(self):
        calls["load"] += 1
        return FakeModel()

    monkeypatch.setattr(PaddleOCRLocalClient, "_create_model", build_model)
    first = PaddleOCRLocalClient(detection, recognition)
    second = PaddleOCRLocalClient(detection, recognition)

    result1 = first.recognize(b"image bytes", "vehicle.jpg", "image/jpeg")
    result2 = second.recognize(b"image bytes", "vehicle.jpg", "image/jpeg")

    assert result1 == OCRRecognition(text="RJ45CA1879", confidence=0.9977)
    assert result2 == result1
    assert calls == {"load": 1, "predict": 2}


def test_paddle_local_client_missing_models_fail_without_model_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(PaddleOCRLocalClient, "_model_cache", {})
    monkeypatch.setattr(
        PaddleOCRLocalClient,
        "_create_model",
        lambda _self: pytest.fail("model loading must not run without local weights"),
    )
    client = PaddleOCRLocalClient(tmp_path / "missing-det", tmp_path / "missing-rec")

    with pytest.raises(OCRServiceUnavailable, match="OCR_PADDLE_DETECTION_MODEL_DIR"):
        client.recognize(b"image", "image.png", "image/png")


def test_paddle_local_provider_returns_and_persists_real_response_fields(ocr_api, tmp_path, monkeypatch):
    client, assessment_id, image_id, _storage, engine, _fake = ocr_api
    detection, recognition = _make_local_paddle_model_dirs(tmp_path)
    monkeypatch.setattr(PaddleOCRLocalClient, "_model_cache", {})

    class FakeModel:
        def predict(self, image_path):
            assert Path(image_path).is_file()
            return [SimpleNamespace(json={"res": {"rec_texts": ["RJ45CA1879"], "rec_scores": [0.9977]}})]

    monkeypatch.setattr(PaddleOCRLocalClient, "_create_model", lambda _self: FakeModel())
    local_client = PaddleOCRLocalClient(detection, recognition)
    app.dependency_overrides[get_ocr_client] = lambda: local_client

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 200
    assert response.json()["provider"] == "paddleocr_local"
    assert response.json()["text"] == "RJ45CA1879"
    assert response.json()["confidence"] == pytest.approx(0.9977)
    with Session(engine) as session:
        record = session.scalar(select(OCRResult))
        assert record is not None
        assert record.raw_text == "RJ45CA1879"
        assert record.result_data is not None
        assert record.result_data["confidence"] == pytest.approx(0.9977)


def test_public_provider_name_is_returned_and_persisted(ocr_api):
    client, assessment_id, image_id, _storage, engine, fake = ocr_api
    fake.provider_name = "huggingface_gradio"
    fake.result = OCRRecognition(text="RJ45CA1879")

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 200
    assert response.json()["provider"] == "huggingface_gradio"
    with Session(engine) as session:
        record = session.scalar(select(OCRResult))
        assert record is not None
        assert record.provider == "huggingface_gradio"


def test_public_no_text_output_does_not_complete_or_persist_ocr(ocr_api):
    client, assessment_id, image_id, _storage, engine, _fake = ocr_api
    output = json.dumps(
        [
            {
                "text": (
                    "The English text in the image is not clearly visible.\n\n"
                    "[No text detected]\n![](images/0.jpg)"
                ),
                "done": True,
            }
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gradio_api/upload":
            return httpx.Response(200, json=["/tmp/image.png"])
        if request.url.path == "/gradio_api/call/v2/run_ocr":
            return httpx.Response(200, json={"event_id": "job"})
        return httpx.Response(
            200,
            text=f"event: complete\ndata: {output}\n\n",
            headers={"content-type": "text/event-stream"},
        )

    app.dependency_overrides[get_ocr_client] = lambda: HuggingFaceGradioOCRClient(
        "https://baidu-unlimited-ocr.hf.space",
        transport=httpx.MockTransport(handler),
    )

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 422
    assert response.json()["detail"] == "No readable text was found in the image."
    assert count_ocr_results(engine) == 0


@pytest.mark.parametrize(
    ("error", "status_code", "detail"),
    [
        (OCRServiceUnavailable(), 503, "OCR service is unavailable."),
        (OCRInvalidResponse(), 502, "OCR service returned an invalid response."),
        (OCRInferenceFailed(), 502, "OCR inference failed. The OCR service returned no error details."),
        (OCRUnreadableText(), 422, "No readable text was found in the image."),
    ],
)
def test_ocr_service_failures_are_safe_and_not_persisted(ocr_api, error, status_code, detail):
    client, assessment_id, image_id, _storage, engine, fake = ocr_api
    fake.error = error

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == status_code
    assert response.json()["detail"] == detail
    assert count_ocr_results(engine) == 0


def test_ocr_inference_failure_detail_is_logged_and_returned_safely(ocr_api, caplog):
    client, assessment_id, image_id, _storage, engine, fake = ocr_api
    fake.error = OCRInferenceFailed("Hugging Face OCR Space inference failed: CUDA worker is unavailable")

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 502
    assert response.json()["detail"] == str(fake.error)
    assert "CUDA worker is unavailable" in caplog.text
    assert count_ocr_results(engine) == 0


def test_ocr_returns_404_for_missing_assessment_or_unrelated_image(ocr_api):
    client, assessment_id, image_id, _storage, _engine, fake = ocr_api
    missing_assessment = str(uuid4())

    response = client.post(f"/api/v1/assessments/{missing_assessment}/images/{image_id}/ocr")
    assert response.status_code == 404

    with Session(_engine) as session:
        other_assessment = VehicleAssessment()
        session.add(other_assessment)
        session.commit()
        other_id = str(other_assessment.id)

    response = client.post(f"/api/v1/assessments/{other_id}/images/{image_id}/ocr")
    assert response.status_code == 404
    assert fake.calls == []


def test_ocr_returns_404_for_missing_image(ocr_api):
    client, assessment_id, _image_id, _storage, _engine, fake = ocr_api

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{uuid4()}/ocr")

    assert response.status_code == 404
    assert fake.calls == []


def test_ocr_returns_safe_unavailable_error_when_service_url_is_not_configured(ocr_api):
    client, assessment_id, image_id, _storage, _engine, fake = ocr_api
    app.dependency_overrides[get_ocr_client] = lambda: None

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 503
    assert response.json()["detail"] == "OCR service is not configured."
    assert fake.calls == []


def test_ocr_rejects_corrupt_stored_image_before_calling_service(ocr_api):
    client, assessment_id, image_id, storage, engine, fake = ocr_api
    with Session(engine) as session:
        image = session.get(UploadedImage, UUID(image_id))
        assert image is not None
        storage_path = storage.root / image.object_key
        storage_path.write_bytes(b"corrupt image bytes")

    response = client.post(f"/api/v1/assessments/{assessment_id}/images/{image_id}/ocr")

    assert response.status_code == 422
    assert response.json()["detail"] == "Stored image is corrupt or unsupported."
    assert fake.calls == []
    assert count_ocr_results(engine) == 0
