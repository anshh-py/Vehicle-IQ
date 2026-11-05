from collections.abc import Generator
from io import BytesIO
from uuid import UUID, uuid4

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


def image_bytes(image_format: str) -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), color=(25, 80, 150)).save(output, format=image_format)
    return output.getvalue()


@pytest.fixture
def upload_api() -> Generator[tuple[TestClient, str, object], None, None]:
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

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_image_upload_limit] = lambda: 10 * 1024 * 1024
    try:
        with TestClient(app) as client:
            yield client, assessment_id, engine
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.mark.parametrize(
    ("image_format", "content_type"),
    [
        ("JPEG", "image/jpeg"),
        ("PNG", "image/png"),
        ("WEBP", "image/webp"),
    ],
)
def test_uploads_valid_supported_image_and_lists_metadata(upload_api, image_format, content_type):
    client, assessment_id, engine = upload_api
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
    with Session(engine) as session:
        image = session.scalar(select(UploadedImage))
        assert image is not None
        assert image.storage_provider == "postgresql"
        assert image.content == content

    statements: list[str] = []

    def collect_sql(_connection, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", collect_sql)
    try:
        listed = client.get(f"/api/v1/assessments/{assessment_id}/images")
    finally:
        event.remove(engine, "before_cursor_execute", collect_sql)
    assert listed.status_code == 200
    assert listed.json() == [metadata]
    list_query = next(sql for sql in statements if "FROM uploaded_images" in sql)
    assert "uploaded_images.content," not in list_query

    downloaded = client.get(
        f"/api/v1/assessments/{assessment_id}/images/{metadata['id']}/content"
    )
    assert downloaded.status_code == 200
    assert downloaded.content == content
    assert downloaded.headers["content-type"].startswith(content_type)
    assert downloaded.headers["cache-control"] == "private, no-store"
    assert downloaded.headers["x-content-type-options"] == "nosniff"


def test_image_content_requires_matching_assessment(upload_api):
    client, assessment_id, _engine = upload_api
    upload = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("photo.png", image_bytes("PNG"), "image/png")},
    )
    assert upload.status_code == 201

    other_assessment = client.post("/api/v1/assessments", json={})
    assert other_assessment.status_code == 201
    response = client.get(
        f"/api/v1/assessments/{other_assessment.json()['id']}/images/{upload.json()['id']}/content"
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Image content not found."}


def test_rejects_invalid_content_even_when_headers_claim_png(upload_api):
    client, assessment_id, engine = upload_api

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("photo.png", b"this is not a PNG", "image/png")},
    )

    assert response.status_code == 422
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_rejects_upload_over_configured_limit(upload_api):
    client, assessment_id, engine = upload_api
    app.dependency_overrides[get_image_upload_limit] = lambda: 32

    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("large.png", b"x" * 33, "image/png")},
    )

    assert response.status_code == 413
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_missing_assessment_does_not_store_image(upload_api):
    client, _assessment_id, engine = upload_api
    missing_id = "00000000-0000-0000-0000-000000000001"

    response = client.post(
        f"/api/v1/assessments/{missing_id}/images",
        files={"file": ("photo.png", image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 404
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_listing_images_for_missing_assessment_returns_404(upload_api):
    client, _assessment_id, _engine = upload_api

    response = client.get("/api/v1/assessments/00000000-0000-0000-0000-000000000001/images")

    assert response.status_code == 404


def test_database_failure_rolls_back_image_bytes_and_hides_details(upload_api, monkeypatch):
    client, assessment_id, engine = upload_api

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
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None


def test_legacy_metadata_without_database_bytes_is_not_treated_as_migrated(upload_api):
    client, assessment_id, engine = upload_api
    with Session(engine) as session:
        image = UploadedImage(
            assessment_id=UUID(assessment_id),
            storage_provider="local",
            storage_bucket="vehicle-images",
            object_key="legacy-file.png",
            original_filename="legacy.png",
            content_type="image/png",
            byte_size=10,
        )
        session.add(image)
        session.commit()
        image_id = str(image.id)

    listed = client.get(f"/api/v1/assessments/{assessment_id}/images")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == image_id
    response = client.get(f"/api/v1/assessments/{assessment_id}/images/{image_id}/content")
    assert response.status_code == 404
    assert response.json() == {"detail": "Image content not found."}


def test_upload_persists_explicit_image_category(upload_api):
    client, assessment_id, engine = upload_api
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("front.png", image_bytes("PNG"), "image/png")},
        data={"image_type": "vehicle_exterior"},
    )
    assert response.status_code == 201
    assert response.json()["image_type"] == "vehicle_exterior"
    assert response.json()["classification_source"] == "user_input"
    assert response.json()["damage_eligible"] is True
    with Session(engine) as session:
        image = session.scalar(select(UploadedImage))
        decision = image.image_metadata["damage_eligibility"]
        assert decision["image_type"] == "vehicle_exterior"
        assert decision["classification_source"] == "user_input"
        assert decision["damage_eligible"] is True


def test_upload_rejects_unrecognized_explicit_image_category(upload_api):
    client, assessment_id, engine = upload_api
    response = client.post(
        f"/api/v1/assessments/{assessment_id}/images",
        files={"file": ("front.png", image_bytes("PNG"), "image/png")},
        data={"image_type": "truck"},
    )
    assert response.status_code == 422
    with Session(engine) as session:
        assert session.scalar(select(UploadedImage)) is None
