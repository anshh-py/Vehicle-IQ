"""Run one local CPU PaddleOCR check against a stored Vehicle IQ image."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from uuid import UUID

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This machine's normal profile cache is not writable in the current runtime.
# Keep all Paddle/PaddleX cache files in the already git-ignored local storage.
_cache_root = PROJECT_ROOT / ".local-storage" / "paddleocr-smoke-bos"
os.environ["USERPROFILE"] = str(_cache_root / "home")
os.environ["HOME"] = str(_cache_root / "home")
os.environ["PADDLE_PDX_CACHE_HOME"] = str(_cache_root / "paddlex")
os.environ["PADDLE_PDX_MODEL_SOURCE"] = "bos"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import psutil
from sqlalchemy import select

from app.core.config import settings
from app.db.models import UploadedImage
from app.db.session import SessionLocal
from app.storage.images import LocalImageStorage


DEFAULT_ASSESSMENT_ID = "ddf2430f-e473-4eed-bbd4-681f28f7bf09"
DEFAULT_IMAGE_ID = "524f560c-10a9-4a6b-9d05-107cdb7c8e15"
MODEL_BASE_URL = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"


def ensure_bos_model(model_name: str) -> Path:
    """Download one official Paddle model directly from BOS and cache it locally."""
    model_dir = _cache_root / "paddlex" / "official_models" / model_name
    if model_dir.exists():
        # Paddle has already loaded this directory successfully; don't trigger
        # a network fallback or crawl the cache just to validate its contents.
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MODEL_BASE_URL}/{model_name}_infer.tar"
    with tempfile.TemporaryDirectory(prefix="vehicle-iq-paddle-") as temp_dir:
        archive_path = Path(temp_dir) / f"{model_name}_infer.tar"
        with requests.get(url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)

        extract_root = Path(temp_dir) / "extracted"
        extract_root.mkdir()
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive.getmembers():
                target = (extract_root / member.name).resolve()
                if extract_root.resolve() not in target.parents and target != extract_root.resolve():
                    raise RuntimeError(f"Unsafe path in official model archive: {model_name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Links are not allowed in official model archive: {model_name}")
            archive.extractall(extract_root, filter="data")

        model_file = next(
            iter((*extract_root.rglob("inference.json"), *extract_root.rglob("inference.pdmodel"))),
            None,
        )
        if model_file is None:
            raise RuntimeError(f"Official model archive has no inference files: {model_name}")
        source_dir = model_file.parent
        if model_dir.exists():
            shutil.rmtree(model_dir)
        shutil.copytree(source_dir, model_dir)
    return model_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment-id", default=DEFAULT_ASSESSMENT_ID)
    parser.add_argument("--image-id", default=DEFAULT_IMAGE_ID)
    args = parser.parse_args()
    assessment_id = UUID(args.assessment_id)
    image_id = UUID(args.image_id)

    with SessionLocal() as session:
        image = session.scalar(
            select(UploadedImage).where(
                UploadedImage.id == image_id,
                UploadedImage.assessment_id == assessment_id,
            )
        )
        if image is None:
            parser.error("No uploaded image matches the supplied assessment and image IDs.")
        content = LocalImageStorage(settings.image_storage_dir).read(image.object_key)
        suffix = Path(image.object_key).suffix or ".img"

    process = psutil.Process()
    rss_before_model = process.memory_info().rss

    from paddle import is_compiled_with_cuda, set_device
    from paddleocr import PaddleOCR

    set_device("cpu")
    detection_model_dir = ensure_bos_model("PP-OCRv5_mobile_det")
    recognition_model_dir = ensure_bos_model("en_PP-OCRv5_mobile_rec")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_detection_model_dir=str(detection_model_dir),
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(recognition_model_dir),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
        device="cpu",
    )
    rss_after_model = process.memory_info().rss

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
            image_file.write(content)
            temporary_path = image_file.name

        results = ocr.predict(temporary_path)
        recognized: list[tuple[str, float | None]] = []
        for result in results:
            payload = result.json if hasattr(result, "json") else result
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload["res"]
            if not isinstance(payload, dict):
                continue
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or []
            for index, text in enumerate(texts):
                if not isinstance(text, str) or not text.strip():
                    continue
                score = scores[index] if index < len(scores) else None
                recognized.append((text.strip(), float(score) if score is not None else None))

        print("device=cpu")
        print(f"cuda_compiled={is_compiled_with_cuda()}")
        print(f"recognized_text={' '.join(text for text, _ in recognized)}")
        confidence_values = [score for _, score in recognized if score is not None]
        if confidence_values:
            print("confidence=" + ",".join(f"{score:.4f}" for score in confidence_values))
        else:
            print("confidence=not provided")
        rss_after_inference = process.memory_info().rss
        print(f"rss_before_model_mib={rss_before_model / (1024 * 1024):.1f}")
        print(f"rss_after_model_mib={rss_after_model / (1024 * 1024):.1f}")
        print(f"rss_after_inference_mib={rss_after_inference / (1024 * 1024):.1f}")
        return 0 if recognized else 2
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
