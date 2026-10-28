---
title: Vehicle IQ Baidu Unlimited-OCR ZeroGPU
emoji: 🚘
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_space.py
python_version: 3.12.12
models:
  - baidu/Unlimited-OCR
---

# Vehicle IQ Baidu Unlimited-OCR ZeroGPU Space

This is a separate Gradio entry point for Hugging Face Spaces. It leaves the extracted FastAPI `app.py` and its local `POST /ocr` contract unchanged. The Gradio app returns JSON with `success`, `filename`, `text`, and `error`; its automatically exposed Gradio API endpoint is `/predict`.

The model call and inference arguments follow `baidu/Unlimited-OCR` and the extracted service. This code has not been run against model weights; OCR recognition quality is unverified.

## Deploy manually

1. Create a **new** Space on Hugging Face and choose the **Gradio** SDK. Do not convert or overwrite the existing FastAPI Space.
2. In the new Space's **Settings → Hardware**, select **ZeroGPU**.
3. Copy the files in this directory (`README.md`, `gradio_space.py`, `ocr_core.py`, and `requirements.txt`) to the root of the new Space repository.
4. Commit and push those files to the new Space repository using your normal Hugging Face Git workflow. No token is needed in these files. The public model can be fetched by the Space runtime; if you make the Space private, configure access through Hugging Face's normal private-repository permissions.
5. Wait for the Space build and model initialization, then test one image in the Space UI. Startup downloads the model weights; this project did not download them locally.
6. Copy the Space's public `https://<namespace>-<space-name>.hf.space` URL. In Vehicle IQ's project-root `.env`, set:

   ```dotenv
   OCR_SERVICE_PROVIDER=huggingface_gradio
   OCR_SERVICE_URL=https://<namespace>-<space-name>.hf.space
   ```

   Restart Vehicle IQ. The backend calls the Space through Gradio's `/predict` API. For a private Space, set `HF_TOKEN` in the backend process environment with a read-scoped token; never put it in source control or logs. A public Space needs no token.

## Runtime and quota

The Space metadata pins Python 3.12.12, which is currently listed as ZeroGPU-supported. The model-card-tested PyTorch 2.10.0 and related model dependencies are pinned in `requirements.txt`. Gradio, `spaces`, and `huggingface_hub` are provided by the managed Spaces runtime and are intentionally not pinned here. The app imports `spaces` before PyTorch, loads the model at module scope onto CUDA as ZeroGPU requires, and wraps inference with `@spaces.GPU`.

ZeroGPU hosting on a free personal account currently requires the account to be in good standing, have a verified email, and be at least 30 days old. The documented free-account quota is currently 5 GPU minutes per day; queue priority and quota availability vary. Free access is limited and may change, and the service is not guaranteed to be continuously available. Do not enable paid overage for this project with a zero budget. Confirm current eligibility and quota on Hugging Face before relying on the Space.
