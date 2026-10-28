"""Hugging Face Spaces Gradio + ZeroGPU entry point.

This module is intentionally separate from the extracted FastAPI ``app.py``.
Spaces loads model weights at module scope as required by ZeroGPU; inference is
only run from the GPU-decorated function.
"""

import spaces  # Must be imported before torch/transformers for ZeroGPU CUDA support.
import torch
import gradio as gr
from transformers import AutoModel, AutoTokenizer

from ocr_core import InvalidUploadedImage, MODEL_NAME, NoReadableText, run_ocr


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_safetensors=True,
    torch_dtype=torch.bfloat16,
).eval().to("cuda")


@spaces.GPU(duration=120)
def recognize_image(image_path: str | None):
    """Recognize text from one uploaded image and return a stable JSON result."""
    filename = None
    try:
        if not image_path:
            return {"success": False, "filename": None, "text": "", "error": "Choose an image to process."}
        from pathlib import Path

        filename = Path(image_path).name
        with torch.no_grad():
            text = run_ocr(image_path, model, tokenizer)
        return {"success": True, "filename": filename, "text": text, "error": None}
    except InvalidUploadedImage:
        return {"success": False, "filename": filename, "text": "", "error": "The uploaded file is not a readable image."}
    except NoReadableText:
        return {"success": False, "filename": filename, "text": "", "error": "No readable text was found in the image."}
    except Exception:
        # Deliberately avoid returning exception details, paths, or credentials.
        return {"success": False, "filename": filename, "text": "", "error": "OCR inference failed. Please retry with a valid image."}


demo = gr.Interface(
    fn=recognize_image,
    inputs=gr.File(type="filepath", file_types=["image"], label="Image"),
    outputs=gr.JSON(label="OCR result"),
    api_name="predict",
    title="Vehicle IQ — Baidu Unlimited-OCR (ZeroGPU)",
    description="Upload an image to extract visible text. Results are OCR output only and are not verified vehicle-registration records.",
)


if __name__ == "__main__":
    demo.launch()
