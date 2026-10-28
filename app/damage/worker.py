"""Bridge the API process to the isolated CPU-only YOLO environment."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from threading import Lock
from typing import Protocol

from app.damage.inference import DamageDetection, DamageInferenceResult


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_damage_inference.py"
WORKER_MARKER = "VEHICLE_IQ_DAMAGE_RESULT="
MODEL_NAME = "YOLO11m-Car-Damage-Detector"
_WORKER_LOCK = Lock()
_ALLOWED_EXTENSIONS = {"jpg", "png", "webp"}


class DamageWorkerError(RuntimeError):
    """Base error for the isolated inference process."""


class DamageWorkerUnavailable(DamageWorkerError):
    """The configured isolated runtime or local model is unavailable."""


class DamageWorkerTimeout(DamageWorkerError):
    """The CPU inference process exceeded its configured timeout."""


class DamageWorkerInvalidResponse(DamageWorkerError):
    """The worker did not return the documented result payload."""


class DamageWorkerInferenceFailed(DamageWorkerError):
    """The worker started but could not complete model inference."""


class DamageProvider(Protocol):
    provider_name: str

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult: ...


class IsolatedDamageProvider:
    """Invoke one CPU inference in the dedicated damage virtual environment."""

    provider_name = "yolo11m_local"

    def __init__(
        self,
        python_executable: Path | str,
        checkpoint_path: Path | str,
        *,
        timeout_seconds: int = 180,
        runner_path: Path | str = RUNNER_PATH,
        image_size: int = 640,
    ) -> None:
        self.python_executable = Path(python_executable).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.runner_path = Path(runner_path).expanduser().resolve()
        self.timeout_seconds = timeout_seconds
        self.image_size = image_size

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        extension = extension.lower().lstrip(".")
        if extension not in _ALLOWED_EXTENSIONS:
            raise DamageWorkerInvalidResponse("Unsupported image format for damage inference.")
        if not self.python_executable.is_file():
            raise DamageWorkerUnavailable("Dedicated damage inference environment is unavailable.")
        if not self.checkpoint_path.is_file():
            raise DamageWorkerUnavailable("Local damage checkpoint is unavailable.")
        if not self.runner_path.is_file():
            raise DamageWorkerUnavailable("Damage inference runner is unavailable.")

        try:
            with tempfile.TemporaryDirectory(prefix="vehicle-iq-damage-") as temp_dir:
                image_path = Path(temp_dir) / f"image.{extension}"
                image_path.write_bytes(content)
                command = [
                    str(self.python_executable),
                    str(self.runner_path),
                    str(image_path),
                    "--checkpoint",
                    str(self.checkpoint_path),
                    "--imgsz",
                    str(self.image_size),
                    "--worker-protocol",
                ]
                with _WORKER_LOCK:
                    completed = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        env=self._worker_environment(),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
        except subprocess.TimeoutExpired as exc:
            raise DamageWorkerTimeout("Damage inference exceeded its time limit.") from exc
        except OSError as exc:
            raise DamageWorkerUnavailable("Unable to start the isolated damage inference process.") from exc

        payload = self._extract_payload(completed.stdout)
        if payload is None:
            if completed.returncode != 0:
                raise DamageWorkerInferenceFailed("Damage inference process failed.")
            raise DamageWorkerInvalidResponse("Damage inference worker returned an invalid response.")
        if payload.get("ok") is not True:
            error_code = payload.get("error")
            if error_code in {"checkpoint_missing", "runtime_unavailable"}:
                raise DamageWorkerUnavailable("Local damage model or inference runtime is unavailable.")
            if error_code == "invalid_image":
                raise DamageWorkerInvalidResponse("Damage inference worker rejected the image.")
            raise DamageWorkerInferenceFailed("Damage model inference failed.")
        if completed.returncode != 0:
            raise DamageWorkerInferenceFailed("Damage inference process failed.")
        return self._parse_success(payload)

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        # Keep credentials and unrelated application settings out of the worker.
        safe_names = (
            "PATH",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "LOCALAPPDATA",
        )
        environment = {name: os.environ[name] for name in safe_names if name in os.environ}
        environment["PYTHONUTF8"] = "1"
        environment["YOLO_CONFIG_DIR"] = str(
            PROJECT_ROOT / ".local-storage" / "damage-poc" / "ultralytics-config"
        )
        environment["ULTRALYTICS_AUTOUPDATE"] = "False"
        return environment

    @staticmethod
    def _extract_payload(stdout: str) -> dict | None:
        for line in reversed(stdout.splitlines()):
            if not line.startswith(WORKER_MARKER):
                continue
            try:
                payload = json.loads(line[len(WORKER_MARKER) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None

    @staticmethod
    def _parse_success(payload: dict) -> DamageInferenceResult:
        detections_data = payload.get("detections")
        elapsed = payload.get("inference_seconds")
        if not isinstance(detections_data, list) or not isinstance(elapsed, (int, float)):
            raise DamageWorkerInvalidResponse("Damage inference worker returned an invalid response.")
        if not math.isfinite(float(elapsed)) or elapsed < 0:
            raise DamageWorkerInvalidResponse("Damage inference worker returned an invalid response.")

        detections: list[DamageDetection] = []
        try:
            for item in detections_data:
                if not isinstance(item, dict):
                    raise ValueError
                name = item.get("class")
                confidence = item.get("confidence")
                bbox = item.get("bbox_xyxy_pixels")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError
                if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                    raise ValueError
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError
                coordinates = tuple(float(value) for value in bbox)
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError
                detections.append(
                    DamageDetection(
                        class_name=name,
                        confidence=float(confidence),
                        bbox_xyxy=coordinates,
                    )
                )
        except (TypeError, ValueError):
            raise DamageWorkerInvalidResponse("Damage inference worker returned malformed detections.") from None
        return DamageInferenceResult(tuple(detections), float(elapsed))
