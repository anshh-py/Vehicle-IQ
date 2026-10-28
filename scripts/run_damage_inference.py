"""Run the local YOLO11m damage checkpoint on one image using CPU inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.damage.inference import (
    DEFAULT_CHECKPOINT,
    DamageCheckpointMissing,
    DamageDetector,
    DamageInferenceError,
    DamageInputError,
    DamageRuntimeUnavailable,
)


DEFAULT_IMAGE = PROJECT_ROOT / "test_images" / "Damaged Suzuki Sedan on a Tree-Lined Road.png"
WORKER_MARKER = "VEHICLE_IQ_DAMAGE_RESULT="


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--worker-protocol", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    process = psutil.Process()
    rss_peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample_rss() -> None:
        while not stop.wait(0.02):
            try:
                rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
            except psutil.Error:
                pass

    monitor = threading.Thread(target=sample_rss, daemon=True)
    monitor.start()
    try:
        detector = DamageDetector(args.checkpoint, image_size=args.imgsz)
        result = detector.predict(args.image)
        detections = [
            {
                "class": detection.class_name,
                "confidence": detection.confidence,
                "bbox_xyxy_pixels": list(detection.bbox_xyxy),
            }
            for detection in result.detections
        ]
        rss_peak[0] = max(rss_peak[0], process.memory_info().rss)
        payload = {
            "ok": True,
            "device": "cpu",
            "inference_seconds": result.inference_seconds,
            "rss_peak_sampled_mib": rss_peak[0] / 1024**2,
            "detections": detections,
        }
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps(payload), flush=True)
        else:
            payload.update(image=str(args.image.resolve()), checkpoint=str(args.checkpoint.resolve()))
            print(json.dumps(payload, indent=2))
        return 0
    except DamageCheckpointMissing:
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps({"ok": False, "error": "checkpoint_missing"}), flush=True)
            return 2
        raise
    except DamageRuntimeUnavailable:
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps({"ok": False, "error": "runtime_unavailable"}), flush=True)
            return 3
        raise
    except DamageInputError:
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps({"ok": False, "error": "invalid_image"}), flush=True)
            return 4
        raise
    except DamageInferenceError:
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps({"ok": False, "error": "inference_failed"}), flush=True)
            return 5
        raise
    except Exception:
        if args.worker_protocol:
            print(WORKER_MARKER + json.dumps({"ok": False, "error": "worker_failed"}), flush=True)
            return 6
        raise
    finally:
        stop.set()
        monitor.join(timeout=1)


if __name__ == "__main__":
    raise SystemExit(main())
