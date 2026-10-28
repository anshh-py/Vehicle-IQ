---
title: Unlimited-OCR API
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: fastapi
sdk_version: 0.115.6
app_port: 7860
pinned: false
---

# Unlimited-OCR API

FastAPI-based REST API for Baidu's Unlimited-OCR model.

## Endpoints

- `GET /` - Health check
- `POST /ocr` - Upload image for OCR parsing

## Usage

```bash
curl -X POST "https://your-space-url/ocr" \
  -F "file=@image.png"
```

## Model

Uses `baidu/Unlimited-OCR` from  for document parsing.
