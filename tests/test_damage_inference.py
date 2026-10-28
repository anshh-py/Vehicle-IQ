from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from app.damage.inference import (
    DamageCheckpointMissing,
    DamageDetector,
    DamageInferenceError,
    DamageInputError,
)


class FakeArray:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeModel:
    def __init__(self, results=None, error=None):
        self.results = results
        self.error = error
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.results


def write_test_image(path: Path) -> Path:
    Image.new("RGB", (16, 12), color="white").save(path, format="PNG")
    return path


def fake_result(*, boxes=True):
    if not boxes:
        return SimpleNamespace(boxes=None, names={})
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=FakeArray([[1.0, 2.0, 10.0, 11.0]]),
            conf=FakeArray([0.91]),
            cls=FakeArray([2.0]),
        ),
        names={2: "dent"},
    )


def test_detector_parses_real_result_shape_and_reuses_loaded_model(tmp_path):
    image = write_test_image(tmp_path / "image.png")
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    model = FakeModel(results=[fake_result()])
    factory_calls = []

    detector = DamageDetector(checkpoint, model_factory=lambda path: factory_calls.append(path) or model)
    first = detector.predict(image)
    second = detector.predict(image)

    assert factory_calls == [str(checkpoint.resolve())]
    assert len(model.calls) == 2
    assert all(call["device"] == "cpu" and call["imgsz"] == 640 for call in model.calls)
    assert first.detections[0].class_name == "dent"
    assert first.detections[0].confidence == 0.91
    assert first.detections[0].bbox_xyxy == (1.0, 2.0, 10.0, 11.0)
    assert second.inference_seconds >= 0


def test_detector_reports_missing_checkpoint_without_loading_model(tmp_path):
    image = write_test_image(tmp_path / "image.png")
    detector = DamageDetector(tmp_path / "missing.pt", model_factory=lambda _path: pytest.fail("must not load"))

    with pytest.raises(DamageCheckpointMissing, match="Damage checkpoint not found"):
        detector.predict(image)


def test_detector_rejects_missing_and_corrupt_images(tmp_path):
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    detector = DamageDetector(checkpoint, model_factory=lambda _path: pytest.fail("must not load"))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")

    with pytest.raises(DamageInputError, match="does not exist"):
        detector.predict(tmp_path / "missing.png")
    with pytest.raises(DamageInputError, match="invalid or unreadable"):
        detector.predict(corrupt)


def test_detector_preserves_successful_empty_detection_result(tmp_path):
    image = write_test_image(tmp_path / "image.png")
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    model = FakeModel(results=[fake_result(boxes=False)])
    detector = DamageDetector(checkpoint, model_factory=lambda _path: model)

    result = detector.predict(image)

    assert result.detections == ()


def test_detector_maps_model_failure_to_clear_inference_error(tmp_path):
    image = write_test_image(tmp_path / "image.png")
    checkpoint = tmp_path / "trained.pt"
    checkpoint.write_bytes(b"test-only-checkpoint")
    model = FakeModel(error=RuntimeError("private model details"))
    detector = DamageDetector(checkpoint, model_factory=lambda _path: model)

    with pytest.raises(DamageInferenceError, match=r"Damage inference failed \(RuntimeError\)") as error:
        detector.predict(image)

    assert "private model details" not in str(error.value)
