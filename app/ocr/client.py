from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import logging
import tempfile
import threading
from typing import ClassVar, Protocol
from urllib.parse import quote, urljoin

import httpx


class OCRServiceUnavailable(Exception):
    """The configured OCR service could not be reached or is unavailable."""


class OCRInvalidResponse(Exception):
    """The OCR service returned a response outside its documented contract."""


class OCRUnreadableText(Exception):
    """The OCR service completed but did not find readable text."""


class OCRInferenceFailed(Exception):
    """The remote OCR application reported an inference failure."""


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OCRRecognition:
    text: str
    confidence: float | None = None


class OCRClient(Protocol):
    provider_name: str

    def recognize(self, content: bytes, filename: str, content_type: str) -> OCRRecognition: ...


@dataclass
class _PaddleModelEntry:
    model: object
    inference_lock: threading.Lock


class PaddleOCRLocalClient:
    """Local CPU PaddleOCR adapter using only explicitly configured model files."""

    provider_name = "paddleocr_local"
    _model_cache: ClassVar[dict[tuple[str, str], _PaddleModelEntry]] = {}
    _model_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    _model_files = ("inference.yml", "inference.pdiparams")

    def __init__(self, detection_model_dir: Path | str, recognition_model_dir: Path | str):
        self.detection_model_dir = Path(detection_model_dir).expanduser().resolve()
        self.recognition_model_dir = Path(recognition_model_dir).expanduser().resolve()

    def recognize(self, content: bytes, filename: str, content_type: str) -> OCRRecognition:
        if content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise OCRInvalidResponse("Local PaddleOCR received an unsupported image type")
        entry = self._get_model_entry()
        suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[content_type]
        temp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
                image_file.write(content)
                temp_path = image_file.name
            with entry.inference_lock:
                results = entry.model.predict(temp_path)
        except Exception as exc:
            raise OCRInferenceFailed(
                f"Local PaddleOCR inference failed ({type(exc).__name__})."
            ) from None
        finally:
            if temp_path is not None:
                Path(temp_path).unlink(missing_ok=True)

        recognized: list[tuple[str, float | None]] = []
        try:
            for result in results:
                payload = result.json if hasattr(result, "json") else result
                if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                    payload = payload["res"]
                if not isinstance(payload, dict):
                    raise OCRInvalidResponse("Local PaddleOCR returned an invalid result")
                texts = payload.get("rec_texts") or []
                scores = payload.get("rec_scores") or []
                if not isinstance(texts, (list, tuple)):
                    raise OCRInvalidResponse("Local PaddleOCR returned invalid recognized text")
                for index, text in enumerate(texts):
                    if not isinstance(text, str) or not text.strip():
                        continue
                    score = scores[index] if isinstance(scores, (list, tuple)) and index < len(scores) else None
                    try:
                        confidence = float(score) if score is not None else None
                    except (TypeError, ValueError):
                        confidence = None
                    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
                        confidence = None
                    recognized.append((text.strip(), confidence))
        except OCRInvalidResponse:
            raise
        except Exception:
            raise OCRInvalidResponse("Local PaddleOCR returned an invalid result") from None

        text = " ".join(value for value, _ in recognized).strip()
        if not text:
            raise OCRUnreadableText("Local PaddleOCR found no readable text")
        # The existing API has one scalar confidence field. Preserve a score
        # only when the model returned exactly one recognized text segment.
        confidence = recognized[0][1] if len(recognized) == 1 else None
        return OCRRecognition(text=text, confidence=confidence)

    def _get_model_entry(self) -> _PaddleModelEntry:
        key = (str(self.detection_model_dir), str(self.recognition_model_dir))
        with self._model_cache_lock:
            cached = self._model_cache.get(key)
            if cached is not None:
                return cached
            self._validate_model_dir(self.detection_model_dir)
            self._validate_model_dir(self.recognition_model_dir)
            try:
                model = self._create_model()
            except Exception as exc:
                raise OCRServiceUnavailable(
                    f"Local PaddleOCR model initialization failed ({type(exc).__name__})."
                ) from None
            entry = _PaddleModelEntry(model=model, inference_lock=threading.Lock())
            self._model_cache[key] = entry
            return entry

    @classmethod
    def _validate_model_dir(cls, model_dir: Path) -> None:
        if not model_dir.is_dir() or any(not (model_dir / name).is_file() for name in cls._model_files):
            raise OCRServiceUnavailable(
                "Local PaddleOCR model files are missing. Configure "
                "OCR_PADDLE_DETECTION_MODEL_DIR and OCR_PADDLE_RECOGNITION_MODEL_DIR."
            )
        if not any((model_dir / name).is_file() for name in ("inference.json", "inference.pdmodel")):
            raise OCRServiceUnavailable(
                "Local PaddleOCR model files are incomplete. Configure valid local inference model directories."
            )

    def _create_model(self):
        # Explicit model directories below prevent PaddleX from resolving or
        # downloading models. Disable host checks and Hugging Face networking as
        # a second guard. Redirect only Paddle's caches to git-ignored storage.
        cache_root = Path(__file__).resolve().parents[2] / ".local-storage" / "paddleocr-smoke-bos"
        process_env = {
            "USERPROFILE": str(cache_root / "home"),
            "HOME": str(cache_root / "home"),
            "PADDLE_PDX_CACHE_HOME": str(cache_root / "paddlex"),
            "PADDLE_PDX_MODEL_SOURCE": "bos",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "HF_HUB_OFFLINE": "1",
        }
        previous_env = {name: os.environ.get(name) for name in process_env}
        os.environ.update(process_env)
        try:
            from paddle import set_device
            from paddleocr import PaddleOCR

            set_device("cpu")
            return PaddleOCR(
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_detection_model_dir=str(self.detection_model_dir),
                text_recognition_model_name="en_PP-OCRv5_mobile_rec",
                text_recognition_model_dir=str(self.recognition_model_dir),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
                device="cpu",
            )
        finally:
            for name, old_value in previous_env.items():
                if old_value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = old_value


class BaiduUnlimitedOCRClient:
    """HTTP adapter for the extracted Unlimited-OCR FastAPI service."""

    provider_name = "baidu_unlimited_ocr"

    def __init__(self, service_url: str, timeout_seconds: float = 120.0, transport: httpx.BaseTransport | None = None):
        self.endpoint = urljoin(service_url.rstrip("/") + "/", "ocr")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def recognize(self, content: bytes, filename: str, content_type: str) -> OCRRecognition:
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                response = client.post(
                    self.endpoint,
                    files={"file": (filename, content, content_type)},
                )
        except httpx.TimeoutException as exc:
            raise OCRServiceUnavailable("OCR service timed out") from exc
        except httpx.RequestError as exc:
            raise OCRServiceUnavailable("OCR service request failed") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise OCRServiceUnavailable("OCR service is unavailable")
        if not response.is_success:
            raise OCRInvalidResponse("OCR service rejected the request")

        try:
            payload = response.json()
        except ValueError as exc:
            raise OCRInvalidResponse("OCR service returned invalid JSON") from exc

        if not isinstance(payload, dict) or payload.get("success") is not True:
            raise OCRInvalidResponse("OCR service returned an invalid response")

        text = payload.get("result")
        if not isinstance(text, str):
            raise OCRInvalidResponse("OCR service response has no text result")
        text = text.strip()
        if not text or text.casefold() == "no output found":
            raise OCRUnreadableText("OCR service found no readable text")

        # The extracted service's response contains success, filename, and
        # result only. It does not provide a confidence score.
        return OCRRecognition(text=text)


class HuggingFaceGradioOCRClient:
    """HTTP adapter for the public Baidu Gradio Space's queued OCR API."""

    provider_name = "huggingface_gradio"
    PROMPT = "document parsing."

    def __init__(
        self,
        space_url: str,
        mode: str = "gundam",
        transport: httpx.BaseTransport | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 180.0,
    ):
        if mode not in {"gundam", "base"}:
            raise ValueError("OCR mode must be 'gundam' or 'base'")
        self.base_url = space_url.rstrip("/") + "/"
        self.mode = mode
        self.transport = transport
        self.timeout = httpx.Timeout(
            connect=connect_timeout_seconds,
            read=read_timeout_seconds,
            write=30.0,
            pool=10.0,
        )

    def recognize(self, content: bytes, filename: str, content_type: str) -> OCRRecognition:
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                upload_response = client.post(
                    urljoin(self.base_url, "gradio_api/upload"),
                    files={"files": (filename, content, content_type)},
                )
                self._check_http_response(upload_response, "upload")
                uploaded = self._parse_json(upload_response, "upload")
                if (
                    not isinstance(uploaded, list)
                    or len(uploaded) != 1
                    or not isinstance(uploaded[0], str)
                    or not uploaded[0].strip()
                ):
                    raise OCRInvalidResponse("Hugging Face Space returned an invalid upload path")

                start_response = client.post(
                    urljoin(self.base_url, "gradio_api/call/v2/run_ocr"),
                    json={
                        "image_path": {
                            "path": uploaded[0],
                            "meta": {"_type": "gradio.FileData"},
                        },
                        "mode": self.mode,
                        "prompt": self.PROMPT,
                    },
                )
                self._check_http_response(start_response, "OCR request")
                start_payload = self._parse_json(start_response, "OCR request")
                event_id = start_payload.get("event_id") if isinstance(start_payload, dict) else None
                if not isinstance(event_id, str) or not event_id.strip():
                    raise OCRInvalidResponse("Hugging Face Space response has no event ID")

                result_response = client.get(
                    urljoin(self.base_url, f"gradio_api/call/run_ocr/{quote(event_id, safe='')}"),
                )
                self._check_http_response(result_response, "OCR result")
                text = self._parse_result_sse(result_response.text)
        except httpx.TimeoutException as exc:
            raise OCRServiceUnavailable("Hugging Face OCR Space request timed out") from exc
        except httpx.RequestError as exc:
            raise OCRServiceUnavailable("Hugging Face OCR Space request failed") from exc

        if self._is_unreadable_text(text):
            raise OCRUnreadableText("Hugging Face OCR Space found no readable text")
        return OCRRecognition(text=text)

    @staticmethod
    def _is_unreadable_text(text: str) -> bool:
        normalized = text.strip().casefold()
        return not normalized or any(
            marker in normalized
            for marker in (
                "[no text detected]",
                "no readable text was found in the image",
            )
        )

    @staticmethod
    def _check_http_response(response: httpx.Response, operation: str) -> None:
        if response.is_success:
            return
        if response.status_code == 429 or response.status_code >= 500:
            raise OCRServiceUnavailable(f"Hugging Face Space {operation} is unavailable")
        raise OCRInvalidResponse(f"Hugging Face Space rejected the {operation}")

    @staticmethod
    def _parse_json(response: httpx.Response, description: str):
        try:
            return response.json()
        except ValueError as exc:
            raise OCRInvalidResponse(f"Hugging Face Space returned invalid {description} JSON") from exc

    @classmethod
    def _parse_result_sse(cls, body: str) -> str:
        events: list[tuple[str, str]] = []
        event_name = "message"
        data_lines: list[str] = []

        def finish_event() -> None:
            nonlocal event_name, data_lines
            if data_lines or event_name != "message":
                events.append((event_name, "\n".join(data_lines)))
            event_name = "message"
            data_lines = []

        normalized = body.replace("\r\n", "\n").replace("\r", "\n")
        for line in normalized.split("\n"):
            if line == "":
                finish_event()
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)
        finish_event()

        complete_events = [data for name, data in events if name == "complete"]
        if not complete_events:
            error_events = [data for name, data in events if name == "error"]
            if error_events:
                detail = cls._safe_error_detail(error_events[-1])
                raise OCRInferenceFailed(f"Hugging Face OCR Space inference failed: {detail}")
            raise OCRInvalidResponse("Hugging Face OCR Space returned no complete SSE event")
        try:
            payload = httpx.Response(200, content=complete_events[-1]).json()
        except ValueError as exc:
            raise OCRInvalidResponse("Hugging Face OCR Space returned invalid SSE JSON") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise OCRInvalidResponse("Hugging Face OCR Space returned an invalid OCR output")
        output = payload[0]
        text = output.get("text")
        if not isinstance(text, str) or output.get("done") is not True:
            raise OCRInvalidResponse("Hugging Face OCR Space returned an invalid final OCR output")
        # The Space may include generated Markdown image references alongside
        # its text. They are not OCR content and should not be stored as such.
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text).strip()
        return text

    @staticmethod
    def _safe_error_detail(data: str) -> str:
        """Keep useful upstream error text while excluding URLs, credentials and paths."""
        value = data.strip()
        if not value:
            return "the Space returned no error details"
        try:
            parsed = httpx.Response(200, content=value).json()
            if isinstance(parsed, str):
                value = parsed.strip()
            elif isinstance(parsed, dict):
                value = str(parsed.get("message") or parsed.get("error") or "")
            else:
                value = ""
        except ValueError:
            pass
        value = " ".join(value.split())
        value = re.sub(r"https?://\S+", "[URL]", value, flags=re.IGNORECASE)
        value = re.sub(r"(?i)(bearer\s+|token[=: ]+|password[=: ]+)\S+", r"\1[REDACTED]", value)
        value = re.sub(r"(?i)([A-Z]:\\|/)(?:[^\s:'\"]+[\\/])*[^\s:'\"]*", "[path]", value)
        value = re.sub(r"[\r\n\x00-\x1f\x7f]", " ", value)
        value = " ".join(value.split())[:300]
        return value or "the Space returned no safe error details"
