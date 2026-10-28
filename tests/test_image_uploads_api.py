from collections.abc import Generator
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, event, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.v1.images import get_image_upload_limit
from app.db.base import Base
from app.db.models import UploadedImage, VehicleAssessment
from app.db.session import get_db
from app.main import app
from app.storage.images import ImageStorageError, LocalImageStorage, get_image_storage


def image_bytes(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), color=(25, 80, 150)).save(output, format=image_format)
    return output.getvalue()


@pytest.fixture
def upload_api(tmp_path) -> Generator[tuple[TestClient, str, LocalImageStorage, object], None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as session:
        assessment = VehicleAssessment()
        session.add(assessment)
        session.commit()
        assessment_id = str(assessment.id)

    storage = LocalImageStorage(tmp_path / "image-storage")

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_image_storage] = lambda: storage
    app.dependency_overrides[get_image_upload_limit] = lambda: 10 * 1024 * 1024
    try:
        with TestClient(app) as client:
            yield client, assessment_id, storage, engine
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.mark.parametrize(
    ("image_format", "content_type", "extension"),
    [
        ("JPEG", "image/jpeg", ".jpg"),
        ("PNG", "image/png", ".png"),
        ("WEBP", "image/webp", ".webp"),
    ],
)
def test_uploads_valid_supported_image_and_lists_metadata(upload_api, image_format, content_type, extension):
    client, assessment_id, storage, _engine = upload_api
    content = image_bytes(image_format)

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("../../user-name.txt", content, "application/octet-stream")},
    )

    assert response.status_code == 201
    metadata = response.json()
    assert metadata["assessment_id"] == assessment_id
    assert metadata["content_type"] == content_type
    assert metadata["byte_size"] == len(content)
    assert metadata["original_filename"] == "user-name.txt"
    assert "object_key" not in metadata
    assert "storage_bucket" not in metadata

    stored_files = list(storage.root.iterdir())
    assert len(stored_files) == 1
    assert stored_files[0].suffix == extension
    assert stored_files[0].read_bytes() == content

    listed = client.get(f"/api/v1/assessments/{assessment_id}/images")
    assert listed.status_code == 200
    assert listed.json() == [metadata]


def test_rejects_invalid_content_even_when_headers_claim_png(upload_api):
    client, assessment_id, storage, _engine = upload_api

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("photo.png", b"this is not a PNG", "image/png")},
    )

    assert response.status_code == 422
    assert not storage.root.exists()


def test_rejects_upload_over_configured_limit(upload_api):
    client, assessment_id, storage, _engine = upload_api
    app.dependency_overrides[get_image_upload_limit] = lambda: 32

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("large.png", b"x" * 33, "image/png")},
    )

    assert response.status_code == 413
    assert not storage.root.exists()


def test_missing_assessment_does_not_store_file(upload_api):
    client, _assessment_id, storage, _engine = upload_api
    missing_id = "00000000-0000-0000-0000-000000000001"

    response = client.post(
        f"/api/v1/assessments/{missing_id}/images",
        files={"file": ("photo.png", image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 404
    assert not storage.root.exists()


def test_listing_images_for_missing_assessment_returns_404(upload_api):
    client, _assessment_id, _storage, _engine = upload_api

    response = client.get("/api/v1/assessments/00000000-0000-0000-0000-000000000001/images")

    assert response.status_code == 404


def test_database_failure_removes_stored_file_and_hides_details(upload_api, monkeypatch):
    client, assessment_id, storage, engine = upload_api

    def fail_commit(_session):
        raise SQLAlchemyError("database private details")

    monkeypatch.setattr(Session, "commit", fail_commit)
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("photo.png", image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to save uploaded image."
    assert "database private details" not in response.text
    assert list(storage.root.iterdir()) == []
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_storage_failure_returns_safe_error_without_database_record(upload_api):
    client, assessment_id, _storage, engine = upload_api

    class BrokenStorage:
        def save(self, _content: bytes, _extension: str) -> str:
            raise ImageStorageError("private filesystem details")

        def delete(self, _key: str) -> None:
            pass

    app.dependency_overrides[get_image_storage] = BrokenStorage
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("photo.png", image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Unable to save uploaded image."
    assert "private filesystem details" not in response.text
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_local_storage_generates_unique_keys_and_rejects_traversal(tmp_path):
    storage = LocalImageStorage(tmp_path / "storage")
    first_key = storage.save(b"first", "png")
    second_key = storage.save(b"second", "png")

    assert first_key != second_key
    with pytest.raises(ImageStorageError):
        storage.delete("../../outside.png")
