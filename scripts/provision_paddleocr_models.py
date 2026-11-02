"""Provision the two official PaddleOCR inference models used by Vehicle IQ.

Run after installing requirements.txt. Model binaries stay in the ignored
.local-storage tree; this script never downloads models during API startup.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Callable
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PADDLEX_CACHE_ROOT = PROJECT_ROOT / ".local-storage" / "paddleocr-smoke-bos" / "paddlex"
OFFICIAL_MODELS_DIR = PADDLEX_CACHE_ROOT / "official_models"
MODEL_NAMES = ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec")
REQUIRED_FILES = ("inference.yml", "inference.pdiparams")
ALTERNATIVE_CONFIG_FILES = ("inference.json", "inference.pdmodel")


class ModelProvisioningError(RuntimeError):
    """An official OCR model could not be downloaded or validated."""


def model_dir_is_valid(model_dir: Path) -> bool:
    return (
        model_dir.is_dir()
        and all((model_dir / filename).is_file() for filename in REQUIRED_FILES)
        and any((model_dir / filename).is_file() for filename in ALTERNATIVE_CONFIG_FILES)
    )


def _download_official_model(model_name: str, cache_root: Path) -> Path:
    """Use PaddleX's own official model manager with BOS preferred."""
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_root)
    os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

    # Import only after setting cache/source variables: PaddleX captures these
    # values when its official model manager is initialized.
    from paddlex.inference.utils.official_models import official_models

    return Path(official_models[model_name])


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def provision_models(
    project_root: Path = PROJECT_ROOT,
    downloader: Callable[[str, Path], Path] | None = None,
) -> list[Path]:
    """Validate existing models and download/install only missing or invalid ones."""
    cache_root = (
        project_root / ".local-storage" / "paddleocr-smoke-bos" / "paddlex"
    )
    models_dir = cache_root / "official_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    download_model = downloader or _download_official_model
    installed: list[Path] = []

    for model_name in MODEL_NAMES:
        target = models_dir / model_name
        if model_dir_is_valid(target):
            print(f"PaddleOCR model already valid: {model_name}")
            installed.append(target)
            continue

        backup = target.with_name(f".{model_name}.backup-{uuid4().hex}")
        had_previous_target = target.exists() or target.is_symlink()
        if had_previous_target:
            target.rename(backup)

        try:
            downloaded = Path(download_model(model_name, cache_root)).resolve()
            if downloaded != target.resolve():
                raise ModelProvisioningError(
                    f"Official model manager returned an unexpected directory for {model_name}."
                )
            if not model_dir_is_valid(target):
                raise ModelProvisioningError(
                    f"Downloaded model {model_name} is missing required inference files."
                )
        except Exception as exc:
            if target.exists() or target.is_symlink():
                _remove_path(target)
            if had_previous_target and backup.exists():
                backup.rename(target)
            if isinstance(exc, ModelProvisioningError):
                raise
            raise ModelProvisioningError(
                f"Failed to provision official PaddleX model {model_name} "
                f"({type(exc).__name__}). Check build network access and PaddleX model sources."
            ) from None

        if had_previous_target and backup.exists():
            _remove_path(backup)
        print(f"PaddleOCR model provisioned and validated: {model_name}")
        installed.append(target)

    for model_name, model_dir in zip(MODEL_NAMES, installed, strict=True):
        if not model_dir_is_valid(model_dir):
            raise ModelProvisioningError(
                f"PaddleOCR model validation failed after provisioning: {model_name}."
            )
    return installed


def main() -> int:
    try:
        provision_models()
    except ModelProvisioningError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Both required PaddleOCR models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
