"""External Gradio damage-detection provider for Vehicle IQ.

This provider calls a public Hugging Face Gradio Space that wraps a YOLO
car-damage detector. It implements the same ``DamageProvider`` protocol as the
isolated local YOLO worker and returns Vehicle IQ's normalized damage response
(``DamageInferenceResult`` with pixel xyxy boxes), so the existing API wiring
and pricing safety policy are reused unchanged.

Contract with the Space (verified against a live call, not assumed):

* ``api_name="/run_detection"`` with inputs ``img`` (image file), ``resize_flag``
  (bool) and eight per-class confidence thresholds ``t0..t7``.
* Returns four values: (1) annotated image filepath, (2) a dataframe ``dict``
  with ``headers=["Class", "Confidence", "YOLO [x_center, y_center, w, h]"]``
  and ``data`` rows of ``[class_name, confidence, yolo_norm_str]``, (3) a CSV
  filepath and (4) a YOLO label filepath.
* The YOLO column is **normalized** ``[x_center, y_center, w, h]`` (0..1).
  This provider converts it to pixel xyxy using the original image dimensions.

Only the dataframe value is consumed; the image/CSV/label file artifacts are
ignored. Original provider class names and confidence values are preserved for
all genuine damage classes.

The Space explicitly emits a ``"no damage"`` class to describe a clean vehicle.
That label is a non-damage sentinel, not a damage detection, so it is filtered
out during parsing and is never surfaced as a ``DamageDetection``/
``DamageFinding``. This preserves the distinction between a successful
no-damage inference, a successful inference with actual damage, and a failed/
unavailable inference. Failure/empty/malformed responses are never treated as
clean vehicles.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as ExecutorTimeoutError
from io import BytesIO
import logging
from pathlib import Path
import re
import tempfile
import threading
from time import perf_counter
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError

from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import (
    DamageWorkerInferenceFailed,
    DamageWorkerInvalidResponse,
    DamageWorkerTimeout,
    DamageWorkerUnavailable,
)


logger = logging.getLogger(__name__)

DEFAULT_SPACE = "svuppu/car-damage-detection"
DEFAULT_API_NAME = "/run_detection"

# Per-class confidence thresholds as exposed by the Space (verified defaults).
DEFAULT_THRESHOLDS = {
    "t0": 0.9,    # no damage
    "t1": 0.26,   # lost parts
    "t2": 0.05,   # torn
    "t3": 0.05,   # dent
    "t4": 0.05,   # paint scratch
    "t5": 0.16,   # hole
    "t6": 0.33,   # broken glass
    "t7": 0.05,   # broken lamp
}

_ALLOWED_EXTENSIONS = {"jpg", "png", "webp"}

# Class labels returned by the external Space that indicate the ABSENCE of
# damage. They are treated as non-damage classifications and are therefore
# never emitted as a DamageDetection/DamageFinding.
NO_DAMAGE_CLASSES = frozenset({"no damage"})

_NUMBER_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")
_REDACTED_MESSAGE = "External damage inference service failed."

# A single shared, bounded executor runs every external Gradio damage call.
# Creating one executor per provider instance (which happens per request when
# GradioDamageProvider is used as a FastAPI dependency) would leak a thread and
# its pool per request. A bounded pool also ensures one hung request cannot
# occupy the only worker forever: a timed-out call holds one slot, but the
# remaining workers stay available for subsequent/concurrent detections.
_MAX_GRADIO_WORKERS = 4
_GRADIO_EXECUTOR = ThreadPoolExecutor(
    max_workers=_MAX_GRADIO_WORKERS,
    thread_name_prefix="vehicle-iq-gradio-damage",
)


class GradioDamageProvider:
    """Detect vehicle damage through a public Gradio Space."""

    provider_name = "huggingface_gradio_damage"
    model_name = "huggingface_gradio_damage"

    def __init__(
        self,
        space: str = DEFAULT_SPACE,
        *,
        timeout_seconds: int = 120,
        resize_flag: bool = False,
        thresholds: dict[str, float] | None = None,
        api_name: str = DEFAULT_API_NAME,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.space = space.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.resize_flag = resize_flag
        self.thresholds = dict(thresholds or DEFAULT_THRESHOLDS)
        self.api_name = api_name
        self._client_factory = client_factory or self._default_client_factory
        self._client: Any | None = None
        self._client_lock = threading.Lock()

    # -- client lifecycle --------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    try:
                        self._client = self._client_factory()
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "External damage inference client init failed (%s).",
                            type(exc).__name__,
                        )
                        raise DamageWorkerUnavailable(
                            "External damage inference service is unavailable."
                        ) from None
        return self._client

    @staticmethod
    def _default_client_factory() -> Any:
        # Lazy import keeps the module importable when gradio_client is absent.
        try:
            from gradio_client import Client
        except ImportError as exc:
            raise DamageWorkerUnavailable(
                "External damage inference runtime is unavailable; install gradio_client."
            ) from exc
        return Client(DEFAULT_SPACE)

    # -- protocol ----------------------------------------------------------

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        extension = extension.lower().lstrip(".")
        if extension not in _ALLOWED_EXTENSIONS:
            raise DamageWorkerInvalidResponse("Unsupported image format for damage inference.")
        width, height = self._read_dimensions(content)
        client = self._get_client()
        started = perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="vehicle-iq-gradio-damage-") as temp_dir:
                image_path = Path(temp_dir) / f"image.{extension}"
                image_path.write_bytes(content)
                raw = self._run_with_timeout(self._call_space, client, image_path)
        except DamageWorkerUnavailable:
            raise
        except DamageWorkerTimeout:
            raise
        except DamageWorkerInvalidResponse:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "External damage inference call failed (%s).",
                type(exc).__name__,
            )
            raise DamageWorkerInferenceFailed(_REDACTED_MESSAGE) from None
        elapsed = perf_counter() - started
        return self._parse_result(raw, width, height, elapsed)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _read_dimensions(content: bytes) -> tuple[int, int]:
        try:
            with Image.open(BytesIO(content)) as image:
                return image.size
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise DamageWorkerInvalidResponse("Stored image is corrupt or unsupported.") from exc

    def _run_with_timeout(self, fn: Callable[..., Any], *args: Any) -> Any:
        future = _GRADIO_EXECUTOR.submit(fn, *args)
        try:
            return future.result(timeout=self.timeout_seconds)
        except ExecutorTimeoutError:
            # Best-effort cancel: this only stops a task that has NOT yet begun
            # running. It does NOT interrupt a Gradio request already in flight,
            # so a hung request keeps one pooled worker occupied until it
            # finishes. The bounded shared pool guarantees other subsequent and
            # concurrent detections still get their own worker.
            future.cancel()
            raise DamageWorkerTimeout(
                "External damage inference exceeded its time limit."
            ) from None

    def _call_space(self, client: Any, image_path: Path) -> Any:
        try:
            from gradio_client import handle_file
        except ImportError as exc:
            raise DamageWorkerUnavailable(
                "External damage inference runtime is unavailable; install gradio_client."
            ) from exc
        return client.predict(
            img=handle_file(str(image_path)),
            resize_flag=self.resize_flag,
            **self.thresholds,
            api_name=self.api_name,
        )

    @classmethod
    def _parse_result(
        cls,
        raw: Any,
        width: int,
        height: int,
        elapsed: float,
    ) -> DamageInferenceResult:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise DamageWorkerInvalidResponse(
                "External damage inference returned an invalid response."
            )
        dataframe = raw[1]
        if not isinstance(dataframe, dict):
            raise DamageWorkerInvalidResponse(
                "External damage inference returned an invalid detection table."
            )
        data = dataframe.get("data")
        if not isinstance(data, list):
            raise DamageWorkerInvalidResponse(
                "External damage inference returned an invalid detection table."
            )

        detections: list[DamageDetection] = []
        try:
            for row in data:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    raise ValueError("malformed detection row")
                class_name = row[0]
                confidence = row[1]
                yolo_str = row[2]
                if not isinstance(class_name, str) or not class_name.strip():
                    raise ValueError("missing class name")
                # The Space's "no damage" label is a non-damage sentinel, never
                # a damage detection: skip it so it remains a successful
                # no-damage inference rather than a finding.
                if cls._is_no_damage_label(class_name):
                    continue
                score = float(confidence)
                if not (0 <= score <= 1):
                    raise ValueError("confidence out of range")
                box = cls._parse_normalized_yolo(str(yolo_str))
                x1, y1, x2, y2 = cls._normalized_to_pixel_xyxy(box, width, height)
                detections.append(
                    DamageDetection(
                        class_name=class_name.strip(),
                        confidence=score,
                        bbox_xyxy=(x1, y1, x2, y2),
                    )
                )
        except DamageWorkerInvalidResponse:
            raise
        except (TypeError, ValueError):
            raise DamageWorkerInvalidResponse(
                "External damage inference returned malformed detections."
            ) from None
        return DamageInferenceResult(tuple(detections), elapsed)

    @staticmethod
    def _is_no_damage_label(class_name: str) -> bool:
        normalized = re.sub(r"\s+", " ", class_name).strip().casefold()
        return normalized in NO_DAMAGE_CLASSES

    @staticmethod
    def _parse_normalized_yolo(text: str) -> tuple[float, float, float, float]:
        numbers = [float(value) for value in _NUMBER_RE.findall(text)]
        if len(numbers) != 4:
            raise ValueError("YOLO coordinates must contain four numbers")
        return numbers[0], numbers[1], numbers[2], numbers[3]

    @classmethod
    def _normalized_to_pixel_xyxy(
        cls,
        box: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        x_center, y_center, box_w, box_h = box
        x1 = (x_center - box_w / 2.0) * width
        y1 = (y_center - box_h / 2.0) * height
        x2 = (x_center + box_w / 2.0) * width
        y2 = (y_center + box_h / 2.0) * height
        left = max(0.0, min(float(width), min(x1, x2)))
        right = max(0.0, min(float(width), max(x1, x2)))
        top = max(0.0, min(float(height), min(y1, y2)))
        bottom = max(0.0, min(float(height), max(y1, y2)))
        return left, top, right, bottom
