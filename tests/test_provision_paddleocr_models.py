from pathlib import Path

import pytest

from scripts.provision_paddleocr_models import (
    ALTERNATIVE_CONFIG_FILES,
    MODEL_NAMES,
    REQUIRED_FILES,
    OFFICIAL_MODELS_DIR,
    ModelProvisioningError,
    model_dir_is_valid,
    provision_models,
)
from app.core.config import settings


def _create_valid_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in (*REQUIRED_FILES, ALTERNATIVE_CONFIG_FILES[0]):
        (model_dir / filename).write_bytes(b"mock model metadata")


def test_provisioner_targets_application_model_directories():
    assert OFFICIAL_MODELS_DIR / MODEL_NAMES[0] == settings.ocr_paddle_detection_model_dir
    assert OFFICIAL_MODELS_DIR / MODEL_NAMES[1] == settings.ocr_paddle_recognition_model_dir


def test_valid_existing_models_are_reused_without_download(tmp_path):
    models_dir = tmp_path / ".local-storage" / "paddleocr-smoke-bos" / "paddlex" / "official_models"
    for model_name in MODEL_NAMES:
        _create_valid_model(models_dir / model_name)

    def unexpected_download(*_args):
        pytest.fail("valid local model should not be downloaded again")

    result = provision_models(tmp_path, downloader=unexpected_download)

    assert result == [models_dir / model_name for model_name in MODEL_NAMES]
    assert all(model_dir_is_valid(path) for path in result)


def test_missing_model_file_causes_only_invalid_model_to_be_downloaded(tmp_path):
    models_dir = tmp_path / ".local-storage" / "paddleocr-smoke-bos" / "paddlex" / "official_models"
    for model_name in MODEL_NAMES:
        _create_valid_model(models_dir / model_name)
    invalid_model = models_dir / MODEL_NAMES[0]
    (invalid_model / REQUIRED_FILES[1]).unlink()
    downloaded = []

    def fake_download(model_name, cache_root):
        downloaded.append(model_name)
        target = cache_root / "official_models" / model_name
        _create_valid_model(target)
        return target

    provision_models(tmp_path, downloader=fake_download)

    assert downloaded == [MODEL_NAMES[0]]
    assert all(model_dir_is_valid(models_dir / name) for name in MODEL_NAMES)


def test_download_failure_fails_clearly_and_restores_invalid_previous_directory(tmp_path):
    models_dir = tmp_path / ".local-storage" / "paddleocr-smoke-bos" / "paddlex" / "official_models"
    previous_target = models_dir / MODEL_NAMES[0]
    previous_target.mkdir(parents=True)
    (previous_target / "partial-file").write_text("preserve until replacement succeeds", encoding="utf-8")

    def failed_download(_model_name, _cache_root):
        raise OSError("upstream download failure")

    with pytest.raises(ModelProvisioningError, match="Failed to provision official PaddleX model"):
        provision_models(tmp_path, downloader=failed_download)

    assert (previous_target / "partial-file").read_text(encoding="utf-8") == "preserve until replacement succeeds"


def test_downloader_returning_incomplete_model_fails_validation(tmp_path):
    def incomplete_download(model_name, cache_root):
        target = cache_root / "official_models" / model_name
        target.mkdir(parents=True)
        (target / "inference.yml").write_text("config", encoding="utf-8")
        return target

    with pytest.raises(ModelProvisioningError, match="missing required inference files"):
        provision_models(tmp_path, downloader=incomplete_download)


def test_repeated_provisioning_downloads_each_model_only_once(tmp_path):
    downloaded = []

    def fake_download(model_name, cache_root):
        downloaded.append(model_name)
        target = cache_root / "official_models" / model_name
        _create_valid_model(target)
        return target

    first = provision_models(tmp_path, downloader=fake_download)
    second = provision_models(tmp_path, downloader=fake_download)

    assert first == second
    assert downloaded == list(MODEL_NAMES)
