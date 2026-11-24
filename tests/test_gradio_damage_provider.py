"""Focused tests for the external Gradio damage provider."""
from __future__ import annotations

from io import BytesIO
import time

from PIL import Image
import pytest

from app.damage.gradio_provider import (
    DEFAULT_THRESHOLDS,
    GradioDamageProvider,
)
from app.damage.worker import (
    DamageWorkerInferenceFailed,
    DamageWorkerInvalidResponse,
    DamageWorkerTimeout,
    DamageWorkerUnavailable,
)


def make_png(width: int = 200, height: int = 100) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeClient:
    def __init__(self, result=None, error=None, delay=0.0):
        self.result = result
        self.error = error
        self.delay = delay
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


def dataframe(headers, rows):
    return {"headers": headers, "data": rows, "metadata": None}


YOLO_HEADERS = ["Class", "Confidence", "YOLO [x_center, y_center, w, h]"]


def make_provider(fake, **kwargs):
    return GradioDamageProvider(client_factory=lambda: fake, **kwargs)


def test_successful_response_parses_normalized_yolo_to_pixel_xyxy():
    # Image is 200x100. Normalized [0.5, 0.5, 0.2, 0.4] -> pixel xyxy
    # x1=(0.5-0.1)*200=80, y1=(0.5-0.2)*100=30, x2=120, y2=70.
    result = (
        "/tmp/annotated.webp",
        dataframe(YOLO_HEADERS, [["dent", 0.9, "[0.5, 0.5, 0.2, 0.4]"]]),
        "/tmp/out.csv",
        "/tmp/labels.txt",
    )
    fake = FakeClient(result=result)
    provider = make_provider(fake)

    outcome = provider.detect(make_png(), "png")

    assert len(outcome.detections) == 1
    det = outcome.detections[0]
    assert det.class_name == "dent"
    assert det.confidence == pytest.approx(0.9)
    assert det.bbox_xyxy == pytest.approx((80.0, 30.0, 120.0, 70.0))
    assert outcome.inference_seconds >= 0
    assert fake.calls[0]["resize_flag"] is False
    assert fake.calls[0]["api_name"] == "/run_detection"
    assert fake.calls[0]["t0"] == DEFAULT_THRESHOLDS["t0"]


def test_supported_damage_classes_are_preserved():
    rows = [
        ["dent", 0.9, "[0.5, 0.5, 0.2, 0.4]"],
        ["paint scratch", 0.8, "[0.5, 0.5, 0.2, 0.4]"],
        ["broken lamp", 0.7, "[0.5, 0.5, 0.2, 0.4]"],
        ["torn", 0.6, "[0.5, 0.5, 0.2, 0.4]"],
        ["hole", 0.5, "[0.5, 0.5, 0.2, 0.4]"],
        ["lost parts", 0.4, "[0.5, 0.5, 0.2, 0.4]"],
    ]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")

    assert [d.class_name for d in outcome.detections] == [
        "dent", "paint scratch", "broken lamp", "torn", "hole", "lost parts"
    ]
    assert [d.confidence for d in outcome.detections] == pytest.approx(
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    )


def test_unknown_class_is_preserved_not_dropped():
    rows = [["weird_class", 0.55, "[0.5, 0.5, 0.2, 0.4]"]]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")

    assert [d.class_name for d in outcome.detections] == ["weird_class"]
    assert outcome.detections[0].confidence == pytest.approx(0.55)


def test_empty_detection_response_returns_empty_result():
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, []), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")

    assert outcome.detections == ()
    assert outcome.inference_seconds >= 0


def test_only_no_damage_class_is_filtered_to_empty_result():
    # A clean vehicle yields only the Space's "no damage" sentinel, which must
    # NOT surface as a DamageDetection.
    rows = [["no damage", 0.989, "[0.5, 0.5, 1.0, 1.0]"]]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")

    assert outcome.detections == ()
    assert outcome.inference_seconds >= 0


def test_no_damage_case_and_whitespace_insensitive():
    rows = [["No   Damage", 0.99, "[0.5, 0.5, 1.0, 1.0]"]]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")
    assert outcome.detections == ()


def test_no_damage_does_not_override_real_damage():
    # Both "no damage" AND genuine damage present: damage must remain, and the
    # "no damage" sentinel must not mask it.
    rows = [
        ["no damage", 0.99, "[0.5, 0.5, 1.0, 1.0]"],
        ["dent", 0.82, "[0.5, 0.5, 0.2, 0.4]"],
    ]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(), "png")

    assert len(outcome.detections) == 1
    assert outcome.detections[0].class_name == "dent"
    assert outcome.detections[0].confidence == pytest.approx(0.82)


def test_timeout_raises_damage_worker_timeout():
    fake = FakeClient(
        result=("", dataframe(YOLO_HEADERS, []), "", ""),
        delay=5.0,
    )
    provider = make_provider(fake, timeout_seconds=1)

    with pytest.raises(DamageWorkerTimeout):
        provider.detect(make_png(), "png")


def test_timeout_does_not_permanently_block_subsequent_detections():
    # A slow call will time out but keep running in the background executor.
    slow = FakeClient(
        result=("", dataframe(YOLO_HEADERS, []), "", ""),
        delay=3.0,
    )
    provider = make_provider(slow, timeout_seconds=1)

    with pytest.raises(DamageWorkerTimeout):
        provider.detect(make_png(), "png")

    # A subsequent, fast call must succeed quickly via another pooled worker;
    # it must not block behind the still-running timed-out call.
    fast = FakeClient(result=(
        "",
        dataframe(YOLO_HEADERS, [["dent", 0.5, "[0.5, 0.5, 0.2, 0.4]"]]),
        "",
        "",
    ))
    provider._client = None
    provider._client_factory = lambda: fast

    started = time.perf_counter()
    outcome = provider.detect(make_png(), "png")
    elapsed = time.perf_counter() - started

    assert len(outcome.detections) == 1
    assert elapsed < 2.0


def test_provider_uses_shared_bounded_executor():
    from app.damage.gradio_provider import _GRADIO_EXECUTOR, _MAX_GRADIO_WORKERS

    provider = make_provider(FakeClient())
    assert not hasattr(provider, "_executor")
    assert _GRADIO_EXECUTOR._max_workers <= _MAX_GRADIO_WORKERS
    assert _GRADIO_EXECUTOR._max_workers >= 2


def test_client_factory_failure_maps_to_unavailable():
    def broken_factory():
        raise OSError("cannot reach hugginface hub")

    provider = GradioDamageProvider(client_factory=broken_factory)

    with pytest.raises(DamageWorkerUnavailable):
        provider.detect(make_png(), "png")


def test_provider_failure_never_reports_zero_damage():
    fake = FakeClient(error=RuntimeError("space worker crashed"))
    provider = make_provider(fake)

    with pytest.raises(DamageWorkerInferenceFailed):
        provider.detect(make_png(), "png")


@pytest.mark.parametrize(
    "raw",
    [
        ("/tmp/a.webp",),  # fewer than 2 return values
        ("/tmp/a.webp", ["not", "a", "dict"], "/tmp/a.csv", "/tmp/a.txt"),
        ("/tmp/a.webp", {"headers": YOLO_HEADERS, "data": "nope"}, "/tmp/a.csv", "/tmp/a.txt"),
        ("/tmp/a.webp", dataframe(YOLO_HEADERS, [["dent"]]), "/tmp/a.csv", "/tmp/a.txt"),
        ("/tmp/a.webp", dataframe(YOLO_HEADERS, [["dent", 1.5, "[0.5, 0.5, 0.2, 0.4]"]]), "/tmp/a.csv", "/tmp/a.txt"),
        ("/tmp/a.webp", dataframe(YOLO_HEADERS, [["dent", 0.5, "[0.5, 0.5, 0.2]"]]), "/tmp/a.csv", "/tmp/a.txt"),
    ],
)
def test_malformed_api_response_is_rejected(raw):
    fake = FakeClient(result=raw)
    provider = make_provider(fake)

    with pytest.raises(DamageWorkerInvalidResponse):
        provider.detect(make_png(), "png")


def test_coordinates_are_clamped_to_image_bounds():
    # Center (0.8, 0.5) with normalized size (2.0, 2.0) overflows both edges.
    rows = [["dent", 0.8, "[0.8, 0.5, 2.0, 2.0]"]]
    fake = FakeClient(result=(
        "/tmp/a.webp", dataframe(YOLO_HEADERS, rows), "/tmp/a.csv", "/tmp/a.txt"
    ))
    outcome = make_provider(fake).detect(make_png(200, 100), "png")

    x1, y1, x2, y2 = outcome.detections[0].bbox_xyxy
    assert x1 == 0.0 and y1 == 0.0
    assert x2 == 200.0 and y2 == 100.0


def test_unsupported_extension_is_rejected():
    provider = make_provider(FakeClient())
    with pytest.raises(DamageWorkerInvalidResponse):
        provider.detect(make_png(), "gif")


def test_corrupt_image_is_rejected_without_touching_provider():
    called = {"value": False}

    def tracking_factory():
        called["value"] = True
        return FakeClient()

    provider = GradioDamageProvider(client_factory=tracking_factory)
    with pytest.raises(DamageWorkerInvalidResponse):
        provider.detect(b"not an image", "png")
    assert called["value"] is False
