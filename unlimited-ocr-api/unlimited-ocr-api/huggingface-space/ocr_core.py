"""Small, import-safe wrapper around the model's documented ``infer`` API."""

from pathlib import Path
import tempfile

from PIL import Image, UnidentifiedImageError


MODEL_NAME = "baidu/Unlimited-OCR"
OCR_PROMPT = (
    "<image>Read the vehicle registration number visible on this Indian vehicle number plate. "
    "Return only characters that are clearly visible; do not infer missing or uncertain characters."
)


class InvalidUploadedImage(ValueError):
    """The uploaded file is not a readable image."""


class NoReadableText(ValueError):
    """Inference completed without producing readable text."""


def run_ocr(image_path: str, model, tokenizer) -> str:
    """Run the same inference arguments and output-file convention as the FastAPI service."""
    source = Path(image_path)
    try:
        with Image.open(source) as image:
            image.verify()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise InvalidUploadedImage from exc

    with tempfile.TemporaryDirectory(prefix="unlimited-ocr-output-") as output_dir:
        model.infer(
            tokenizer,
            prompt=OCR_PROMPT,
            image_file=str(source),
            output_path=output_dir,
            base_size=1024,
            image_size=640,
            crop_mode=True,
            max_length=32768,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
        )
        for result_file in Path(output_dir).rglob("*"):
            if result_file.is_file() and result_file.suffix.lower() in {".txt", ".json", ".md"}:
                text = result_file.read_text(encoding="utf-8").strip()
                if text and text.casefold() != "no output found":
                    return text
    raise NoReadableText
