import json
from pathlib import Path
import subprocess

import pytest

from app.damage.worker import (
    DamageWorkerInvalidResponse,
    DamageWorkerTimeout,
    IsolatedDamageProvider,
    WORKER_MARKER,
)


def make_provider(tmp_path: Path) -> IsolatedDamageProvider:
    python = tmp_path / "python.exe"
    checkpoint = tmp_path / "trained.pt"
    runner = tmp_path / "runner.py"
    python.write_bytes(b"isolated interpreter placeholder")
    checkpoint.write_bytes(b"local checkpoint placeholder")
    runner.write_text("# runner placeholder", encoding="utf-8")
    return IsolatedDamageProvider(python, checkpoint, runner_path=runner)


def test_worker_adapts_json_result_and_does_not_forward_database_credentials(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    payload = {
        "ok": True,
        "inference_seconds": 2.5,
        "detections": [
            {"class": "dent", "confidence": 0.9, "bbox_xyxy_pixels": [1, 2, 3, 4]}
        ],
    }
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        observed["image_path"] = Path(command[2])
        return subprocess.CompletedProcess(command, 0, WORKER_MARKER + json.dumps(payload), "")

    monkeypatch.setenv("DATABASE_URL", "must-not-be-forwarded")
    monkeypatch.setattr("app.damage.worker.subprocess.run", fake_run)

    result = provider.detect(b"png image data", "png")

    assert result.inference_seconds == 2.5
    assert result.detections[0].class_name == "dent"
    assert result.detections[0].confidence == 0.9
    assert result.detections[0].bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert "must-not-be-forwarded" not in repr(observed["env"])
    assert "DATABASE_URL" not in observed["env"]
    assert not observed["image_path"].exists()
    assert "--worker-protocol" in observed["command"]


def test_worker_maps_timeout_without_leaking_process_details(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], stderr="private worker details")

    monkeypatch.setattr("app.damage.worker.subprocess.run", timeout)

    with pytest.raises(DamageWorkerTimeout, match="exceeded its time limit") as error:
        provider.detect(b"png image data", "png")

    assert "private worker details" not in str(error.value)


def test_worker_rejects_malformed_response(tmp_path, monkeypatch):
    provider = make_provider(tmp_path)
    monkeypatch.setattr(
        "app.damage.worker.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, WORKER_MARKER + "{bad", ""),
    )

    with pytest.raises(DamageWorkerInvalidResponse):
        provider.detect(b"png image data", "png")
