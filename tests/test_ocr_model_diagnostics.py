import logging

import pytest

from app import main
from app.ocr.client import OCRServiceUnavailable, PaddleOCRLocalClient


def _write_valid_model(model_dir):
    model_dir.mkdir(parents=True)
    (model_dir / "inference.yml").write_text("model: test", encoding="utf-8")
    (model_dir / "inference.pdiparams").write_bytes(b"weights")
    (model_dir / "inference.json").write_text("{}", encoding="utf-8")


def test_startup_logs_model_asset_presence_without_loading_paddle(tmp_path, monkeypatch, caplog):
    detection = tmp_path / "detector"
    recognition = tmp_path / "recognizer"
    _write_valid_model(detection)
    recognition.mkdir()

    monkeypatch.setattr(main.settings, "ocr_service_provider", "hybrid")
    monkeypatch.setattr(main.settings, "ocr_paddle_detection_model_dir", detection)
    monkeypatch.setattr(main.settings, "ocr_paddle_recognition_model_dir", recognition)
    monkeypatch.setattr(
        PaddleOCRLocalClient,
        "_create_model",
        lambda _self: (_ for _ in ()).throw(AssertionError("startup must not load Paddle")),
    )

    with caplog.at_level(logging.INFO, logger="app.ocr.client"):
        main.log_local_ocr_model_assets()

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert "model=PP-OCRv5_mobile_det" in messages[0]
    assert "inference_json_or_pdmodel=True, ready=True" in messages[0]
    assert "model=en_PP-OCRv5_mobile_rec" in messages[1]
    assert "directory_exists=True" in messages[1]
    assert "inference_yml=False" in messages[1]
    assert "ready=False" in messages[1]


def test_unconfigured_ocr_provider_does_not_emit_paddle_asset_logs(monkeypatch, caplog):
    monkeypatch.setattr(main.settings, "ocr_service_provider", "huggingface_gradio")

    with caplog.at_level(logging.INFO, logger="app.ocr.client"):
        main.log_local_ocr_model_assets()

    assert not caplog.records


def test_model_initialization_logs_exception_class_without_exception_message(
    tmp_path, monkeypatch, caplog
):
    detection = tmp_path / "detector"
    recognition = tmp_path / "recognizer"
    _write_valid_model(detection)
    _write_valid_model(recognition)
    monkeypatch.setattr(PaddleOCRLocalClient, "_model_cache", {})
    monkeypatch.setattr(
        PaddleOCRLocalClient,
        "_create_model",
        lambda _self: (_ for _ in ()).throw(RuntimeError("sensitive upstream detail")),
    )

    client = PaddleOCRLocalClient(detection, recognition)
    with caplog.at_level(logging.ERROR, logger="app.ocr.client"):
        with pytest.raises(OCRServiceUnavailable, match="RuntimeError"):
            client._get_model_entry()

    assert "exception_class=RuntimeError" in caplog.text
    assert "sensitive upstream detail" not in caplog.text
