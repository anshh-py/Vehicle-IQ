from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.damage.inference import DamageDetection, DamageInferenceResult
from app.damage.worker import DamageWorkerUnavailable
from app.db.base import Base
from app.db.models import DamageFinding, EvidenceRecord, UploadedImage, VehicleAssessment
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


class SequenceDamageProvider(FakeDamageProvider):
    def __init__(self, results):
        super().__init__()
        self.results = iter(results)

    def detect(self, content: bytes, extension: str) -> DamageInferenceResult:
        self.calls.append((content, extension))
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


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


def add_stored_image(api, content: bytes) -> str:
    object_key = api.storage.save(content, "png")
    with Session(api.engine) as session:
        image = UploadedImage(
            assessment_id=UUID(api.assessment_id),
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key=object_key,
            original_filename="additional-vehicle.png",
            content_type="image/png",
            byte_size=len(content),
            sha256="additional-test-sha256",
        )
        session.add(image)
        session.commit()
        return str(image.id)


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


def test_batch_damage_processes_ten_owned_images_and_report_aggregates_findings(damage_api_fixture):
    api = damage_api_fixture
    image_ids = [api.image_id] + [add_stored_image(api, api.content) for _ in range(9)]

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": image_ids},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["processed_count"] == 10
    assert payload["failed_count"] == 0
    assert len(payload["results"]) == 10
    assert all(item["status"] == "completed" and len(item["findings"]) == 2 for item in payload["results"])
    assert {finding["image_id"] for item in payload["results"] for finding in item["findings"]} == set(image_ids)
    assert count_findings(api.engine) == 20
    assert len(api.provider.calls) == 10

    repeated = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": image_ids},
    )
    assert repeated.status_code == 200
    assert all(item["reused"] for item in repeated.json()["results"])
    assert count_findings(api.engine) == 20
    assert len(api.provider.calls) == 10

    report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
    assert report.status_code == 200
    report_data = report.json()["report_data"]
    assert report_data["stages"]["damage_detection"]["finding_count"] == 20
    assert report_data["stages"]["damage_detection"]["images_with_findings"] == 10


def test_batch_damage_rejects_more_than_ten_and_duplicate_image_ids(damage_api_fixture):
    api = damage_api_fixture
    too_many = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [str(uuid4()) for _ in range(11)]},
    )
    duplicate = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id, api.image_id]},
    )

    assert too_many.status_code == duplicate.status_code == 422
    assert api.provider.calls == []


def test_batch_damage_reports_cross_assessment_image_failure(damage_api_fixture):
    api = damage_api_fixture
    with Session(api.engine) as session:
        other_assessment = VehicleAssessment()
        session.add(other_assessment)
        session.commit()
        other_assessment_id = str(other_assessment.id)

    response = api.client.post(
        f"/api/v1/assessments/{other_assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["results"][0]["error"] == "Image not found for assessment."
    assert api.provider.calls == []
    assert count_findings(api.engine) == 0


def test_batch_damage_empty_detection_is_success_without_fabricated_findings(damage_api_fixture):
    api = damage_api_fixture
    api.provider.result = DamageInferenceResult(detections=(), inference_seconds=0.2)

    response = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["results"][0]["findings"] == []
    assert count_findings(api.engine) == 0
    assert response.json()["results"][0]["reused"] is False

    repeated = api.client.post(
        f"/api/v1/assessments/{api.assessment_id}/damage",
        json={"image_ids": [api.image_id]},
    )
    assert repeated.status_code == 200
    assert repeated.json()["results"][0]["reused"] is True
    assert len(api.provider.calls) == 1
    with Session(api.engine) as session:
        marker = session.scalar(select(EvidenceRecord).where(EvidenceRecord.evidence_type == "damage_detection"))
        assert marker.evidence_data == {"status": "completed", "detection_count": 0}

    report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
    assert report.status_code == 200
    assert report.json()["report_data"]["stages"]["damage_detection"]["status"] == "completed_no_findings"


def test_batch_damage_continues_after_corrupt_image_and_inference_failure(tmp_path):
    with damage_api(tmp_path, provider=FakeDamageProvider()) as api:
        corrupt_id = add_stored_image(api, b"not an image")
        second_id = add_stored_image(api, api.content)
        third_id = add_stored_image(api, api.content)
        api.provider = SequenceDamageProvider(
            [DamageWorkerUnavailable("worker unavailable"), api.provider.result]
        )
        app.dependency_overrides[get_damage_provider] = lambda: api.provider

        response = api.client.post(
            f"/api/v1/assessments/{api.assessment_id}/damage",
            json={"image_ids": [corrupt_id, second_id, third_id]},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "partial"
        assert body["processed_count"] == 1
        assert body["failed_count"] == 2
        assert body["results"][0]["error"] == "Stored image is corrupt or unsupported."
        assert body["results"][1]["error"] == "Damage inference is unavailable."
        assert len(body["results"][2]["findings"]) == 2
        assert len(api.provider.calls) == 2
        assert count_findings(api.engine) == 2

        report = api.client.post(f"/api/v1/assessments/{api.assessment_id}/report")
        assert report.status_code == 200
        assert report.json()["report_data"]["stages"]["damage_detection"]["status"] == "partial"
