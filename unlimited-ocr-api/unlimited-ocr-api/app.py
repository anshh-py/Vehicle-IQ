import os
import tempfile
import shutil

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoTokenizer

app = FastAPI(
    title="Unlimited-OCR API",
    version="1.0.0"
)

MODEL_NAME = "baidu/Unlimited-OCR"
OCR_PROMPT = (
    "<image>Read the vehicle registration number visible on this Indian vehicle number plate. "
    "Return only characters that are clearly visible; do not infer missing or uncertain characters."
)

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def load_model():
    global tokenizer, model

    print(f"Loading model on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=dtype,
    )

    model = model.eval().to(device)

    print("Model loaded successfully!")


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": device
    }


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    contents = await file.read()

    tmp_path = None
    output_dir = None

    try:
        # Get proper extension
        extension = os.path.splitext(file.filename)[1] or ".png"

        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as tmp:

            tmp.write(contents)
            tmp_path = tmp.name

        output_dir = tempfile.mkdtemp(prefix="ocr_out_")

        with torch.no_grad():

            model.infer(
                tokenizer,
                prompt=OCR_PROMPT,
                image_file=tmp_path,
                output_path=output_dir,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=128,
                save_results=True,
            )

        result_text = None

        # Search for output files
        for root, dirs, files in os.walk(output_dir):

            for filename in files:

                if filename.endswith((".txt", ".json", ".md")):

                    result_path = os.path.join(root, filename)

                    with open(
                        result_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        result_text = f.read()

                    break

            if result_text:
                break

        if not result_text:
            result_text = "No output found"

        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "result": result_text
            }
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        if output_dir:
            shutil.rmtree(
                output_dir,
                ignore_errors=True
            )


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860
    )
