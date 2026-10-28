"""CPU-only inference adapter for the local YOLO11m car damage checkpoint.

The public contract is ``DamageDetector.predict(image_path)`` returning an
immutable result with actual class labels, confidence values, pixel xyxy boxes,
and elapsed model-inference seconds. Ultralytics is imported lazily so this
module remains importable from Vehicle IQ's main environment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / ".local-storage" / "damage-poc" / "trained.pt"
DEFAULT_YOLO_CONFIG = PROJECT_ROOT / ".local-storage" / "damage-poc" / "ultralytics-config"


class DamageInferenceError(RuntimeError):
    """Raised when the local model cannot produce a valid result."""


class DamageCheckpointMissing(DamageInferenceError):
    """Raised when the configured local checkpoint is absent."""


class DamageRuntimeUnavailable(DamageInferenceError):
    """Raised when the isolated inference runtime is not installed."""


class DamageInputError(ValueError):
    """Raised when the input path is missing or is not a readable image."""


@dataclass(frozen=True)
class DamageDetection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DamageInferenceResult:
    detections: tuple[DamageDetection, ...]
    inference_seconds: float


ModelFactory = Callable[[str], Any]


class DamageDetector:
    """Lazily load and reuse the local checkpoint for CPU predictions."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_CHECKPOINT,
        *,
        image_size: int = 640,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.image_size = image_size
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None
        self._load_lock = Lock()
        self._predict_lock = Lock()

    def predict(self, image_path: Path | str) -> DamageInferenceResult:
        """Run one CPU inference on a readable image file."""
        input_path = Path(image_path).expanduser().resolve()
        self._validate_image(input_path)
        model = self._get_model()

        started = perf_counter()
        try:
            with self._predict_lock:
                results = model.predict(
                    source=str(input_path),
                    device="cpu",
                    imgsz=self.image_size,
                    save=False,
                    verbose=False,
                )
        except Exception as exc:
            raise DamageInferenceError(f"Damage inference failed ({type(exc).__name__}).") from exc
        elapsed = perf_counter() - started

        if not isinstance(results, (list, tuple)) or not results:
            raise DamageInferenceError("Damage model returned no image result.")
        detections = self._parse_result(results[0])
        return DamageInferenceResult(detections=tuple(detections), inference_seconds=elapsed)

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    if not self.checkpoint_path.is_file():
                        raise DamageCheckpointMissing(
                            f"Damage checkpoint not found: {self.checkpoint_path}"
                        )
                    try:
                        self._model = self._model_factory(str(self.checkpoint_path))
                    except DamageInferenceError:
                        raise
                    except Exception as exc:
                        raise DamageInferenceError(
                            f"Unable to load damage checkpoint ({type(exc).__name__})."
                        ) from exc
        return self._model

    @staticmethod
    def _default_model_factory(checkpoint_path: str) -> Any:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_YOLO_CONFIG))
        os.environ.setdefault("ULTRALYTICS_AUTOUPDATE", "False")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DamageRuntimeUnavailable(
                "Damage inference runtime is unavailable; install requirements-damage-inference.txt "
                "in the dedicated damage environment."
            ) from exc
        return YOLO(checkpoint_path, task="detect")

    @staticmethod
    def _validate_image(image_path: Path) -> None:
        if not image_path.is_file():
            raise DamageInputError("Image file does not exist.")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise DamageInputError("Image file is invalid or unreadable.") from exc

    @classmethod
    def _parse_result(cls, result: Any) -> list[DamageDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        try:
            coordinates = cls._to_list(boxes.xyxy)
            confidences = cls._to_list(boxes.conf)
            class_ids = cls._to_list(boxes.cls)
            names = result.names
            if not (len(coordinates) == len(confidences) == len(class_ids)):
                raise ValueError("detection fields have different lengths")

            detections: list[DamageDetection] = []
            for coords, confidence, class_id in zip(coordinates, confidences, class_ids):
                if len(coords) != 4:
                    raise ValueError("bounding box must contain four coordinates")
                index = int(class_id)
                name = names.get(index, str(index)) if isinstance(names, dict) else names[index]
                values = tuple(float(value) for value in coords)
                score = float(confidence)
                if not math.isfinite(score) or not all(math.isfinite(value) for value in values):
                    raise ValueError("detection contains non-finite values")
                detections.append(
                    DamageDetection(
                        class_name=str(name),
                        confidence=score,
                        bbox_xyxy=values,  # coordinates are in original-image pixels
                    )
                )
            return detections
        except Exception as exc:
            raise DamageInferenceError("Damage model returned malformed detections.") from exc

    @staticmethod
    def _to_list(value: Any) -> list[Any]:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return list(value)
