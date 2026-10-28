from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import DamageWorkerUnavailable
from app.db.base import Base
from app.db.models import DamageFinding, UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.api.v1.damage import get_damage_provider
from app.storage.images import LocalImageStorage, get_image_storage


def png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 20), color=(200, 200, 200)).save(output, format="PNG")
    return output.getvalue()


class FakeDamageProvider:
    provider_name = "yolo11m_local"

    def __init__(self, result=None, error=None):
        self.result = result or DamageInferenceResult(
            detections=(
                DamageDetection("dent", 0.91, (1.0, 2.0, 10.0, 11.0)),
                DamageDetection("broken_lamp", 0.84, (11.0, 3.0, 20.0, 15.0)),
            ),
            inference_seconds=1.25,
        )
        self.error = error
        self.calls = []

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        self.calls.append((content, extension))
        if self.error:
            raise self.error
        return self.result


class DamageAPIFixture:
    def __init__(self, client, engine, assessment_id, image_id, provider, storage, content):
        self.client = client
        self.engine = engine
        self.assessment_id = assessment_id
        self.image_id = image_id
        self.provider = provider
        self.storage = storage
        self.content = content


@contextmanager
def damage_api(tmp_path, *, content=None, provider=None) -> Generator[DamageAPIFixture, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    storage = LocalImageStorage(tmp_path / "damage-images")
    content = png_bytes() if content is None else content
    image_key = storage.save(content, "png")
    with Session(engine) as session:
        assessment = VehicleAssessment()
        session.add(assessment)
        session.flush()
        image = UploadedImage(
            assessment_id=assessment.id,
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key=image_key,
            original_filename="vehicle.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="test-sha256",
        )
        session.add(image)
        session.commit()
        assessment_id, image_id = str(assessment.id), str(image.id)

    fake = provider or FakeDamageProvider()

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_image_storage] = lambda: storage
    app.dependency_overrides[get_damage_provider] = lambda: fake
    try:
        with TestClient(app) as client:
            yield DamageAPIFixture(client, engine, assessment_id, image_id, fake, storage, content)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.fixture
def damage_api_fixture(tmp_path):
    with damage_api(tmp_path) as api:
        yield api


def count_findings(engine) -> int:
    with Session(engine) as session:
        return session.scalar(select(func.count()).select_from(DamageFinding)) or 0


def test_damage_endpoint_persists_only_model_results(damage_api_fixture):
    api = damage_api_fixture

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "yolo11m_local"
    assert payload["status"] == "completed"
    assert payload["reused"] is False
    assert [finding["damage_type"] for finding in payload["findings"]] == ["dent", "broken_lamp"]
    assert [finding["confidence"] for finding in payload["findings"]] == [0.91, 0.84]
    assert payload["findings"][0]["component"] is None
    assert payload["findings"][0]["bounding_box"] == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 10.0,
        "y2": 11.0,
        "coordinate_space": "original_image_pixels",
    }
    assert api.provider.calls == [(api.content, "png")]
    assert count_findings(api.engine) == 2


def test_repeated_damage_request_reuses_existing_findings(damage_api_fixture):
    api = damage_api_fixture
    path = f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"

    first = api.client.post(path)
    second = api.client.post(path)

    assert first.status_code == second.status_code == 200
    assert second.json()["reused"] is True
    assert second.json()["findings"] == first.json()["findings"]
    assert len(api.provider.calls) == 1
    assert count_findings(api.engine) == 2


def test_damage_endpoint_returns_404_for_image_outside_assessment(damage_api_fixture):
    api = damage_api_fixture
    with Session(api.engine) as session:
        other_assessment = VehicleAssessment()
        session.add(other_assessment)
        session.commit()
        other_assessment_id = str(other_assessment.id)

    response = api.client.post(
        f"/api/v1/assessments/{other_assessment_id}/images/{api.image_id}/damage"
    )

    assert response.status_code == 404
    assert api.provider.calls == []
    assert count_findings(api.engine) == 0


def test_damage_endpoint_returns_404_for_missing_image(damage_api_fixture):
    api = damage_api_fixture

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/images/{uuid4()}/damage"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found for assessment."
    assert api.provider.calls == []


def test_damage_endpoint_rejects_unsupported_stored_image_format(tmp_path):
    output = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(output, format="BMP")
    with damage_api(tmp_path, content=output.getvalue()) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )

        assert response.status_code == 422
        assert response.json()["detail"] == "Stored image is corrupt or unsupported."
        assert api.provider.calls == []
        assert count_findings(api.engine) == 0


def test_damage_endpoint_rejects_corrupt_stored_image(tmp_path):
    with damage_api(tmp_path, content=b"not an image") as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )
        assert response.status_code == 422
        assert api.provider.calls == []
        assert count_findings(api.engine) == 0


def test_damage_endpoint_does_not_persist_worker_failure(tmp_path):
    with damage_api(tmp_path, provider=FakeDamageProvider(error=DamageWorkerUnavailable())) as api:
        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/images/{api.image_id}/damage"
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "Damage inference is unavailable."
        assert count_findings(api.engine) == 0
