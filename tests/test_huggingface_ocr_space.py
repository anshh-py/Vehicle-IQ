from pathlib import Path
import sys

import pytest
from PIL import Image


SPACE_DIR = Path(__file__).resolve().parents[1] / "unlimited-ocr-api" / "unlimited-ocr-api" / "huggingface-space"
sys.path.insert(0, str(SPACE_DIR))

from ocr_core import InvalidUploadedImage, NoReadableText, OCR_PROMPT, run_ocr  # noqa: E402


class FakeModel:
    def __init__(self, result="recognized words"):
        self.result = result
        self.kwargs = None

    def infer(self, tokenizer, **kwargs):
        self.kwargs = {"tokenizer": tokenizer, **kwargs}
        Path(kwargs["output_path"], "result.txt").write_text(self.result, encoding="utf-8")


def test_model_core_uses_extracted_inference_contract_and_reads_text(tmp_path):
    image_path = tmp_path / "original-name.png"
    Image.new("RGB", (3, 2), color="white").save(image_path)
    model = FakeModel("  some recognized text  ")

    assert run_ocr(str(image_path), model, "fake-tokenizer") == "some recognized text"
    assert model.kwargs == {
        "tokenizer": "fake-tokenizer",
        "prompt": OCR_PROMPT,
        "image_file": str(image_path),
        "output_path": model.kwargs["output_path"],
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "max_length": 32768,
        "no_repeat_ngram_size": 35,
        "ngram_window": 128,
        "save_results": True,
    }


def test_model_core_rejects_corrupt_image(tmp_path):
    image_path = tmp_path / "bad.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(InvalidUploadedImage):
        run_ocr(str(image_path), FakeModel(), "fake-tokenizer")


def test_model_core_reports_no_readable_output(tmp_path):
    image_path = tmp_path / "empty-output.png"
    Image.new("RGB", (3, 2), color="white").save(image_path)
    model = FakeModel("")

    with pytest.raises(NoReadableText):
        run_ocr(str(image_path), model, "fake-tokenizer")


