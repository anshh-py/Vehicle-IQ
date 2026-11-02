from collections.abc import Generator
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, _connection_record):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)

    def override_get_db():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_create_retrieve_and_return_only_public_assessment_fields(client):
    created = client.post(
        "/api/v1/assessments",
        json={
            "vin": "TEST1234567890123",
            "registration_number": "KA01AB1234",
            "vehicle_details": {"make": "Example", "year": 2022},
        },
    )

    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "pending"
    assert body["vin"] == "TEST1234567890123"
    assert body["vehicle_details"]["make"] == "Example"
    assert set(body) == {
        "id",
        "status",
        "vin",
        "registration_number",
        "vehicle_details",
        "asking_price_inr",
        "created_at",
        "updated_at",
        "vehicle_details_provenance",
        "asking_price_inr_provenance",
    }
    assert body["vehicle_details_provenance"] == {
        "make": "user_input",
        "year": "user_input",
    }
    assert body["asking_price_inr_provenance"] is None

    retrieved = client.get(f"/api/v1/assessments/{body['id']}")
    assert retrieved.status_code == 200
    assert retrieved.json() == body


def test_asking_price_is_optional_nullable_and_persisted(client):
    omitted = client.post("/api/v1/assessments", json={})
    explicit_null = client.post("/api/v1/assessments", json={"asking_price_inr": None})
    supplied = client.post(
        "/api/v1/assessments",
        json={"asking_price_inr": "845000.50", "vehicle_details": {"make": "Example"}},
    )

    assert omitted.status_code == explicit_null.status_code == supplied.status_code == 201
    assert omitted.json()["asking_price_inr"] is None
    assert explicit_null.json()["asking_price_inr"] is None
    assert supplied.json()["asking_price_inr"] == "845000.50"
    assert supplied.json()["asking_price_inr_provenance"] == "user_input"
    assert supplied.json()["vehicle_details"]["make"] == "Example"
    assert supplied.json()["vehicle_details"]["asking_price_inr"] == "845000.50"
    assert client.get(f"/api/v1/assessments/{supplied.json()['id']}").json()["asking_price_inr"] == "845000.50"


@pytest.mark.parametrize("value", [0, -1, "100000000.01", "1.001", "not-a-price"])
def test_asking_price_rejects_invalid_values(client, value):
    response = client.post("/api/v1/assessments", json={"asking_price_inr": value})

    assert response.status_code == 422


def test_asking_price_patch_updates_clears_and_survives_vehicle_details_patch(client):
    created = client.post("/api/v1/assessments", json={"asking_price_inr": 500000}).json()
    assessment_url = f"/api/v1/assessments/{created['id']}"

    update = client.patch(assessment_url, json={"asking_price_inr": "525000.25"})
    details_update = client.patch(assessment_url, json={"vehicle_details": {"make": "Example"}})
    clear = client.patch(assessment_url, json={"asking_price_inr": None})

    assert update.status_code == details_update.status_code == clear.status_code == 200
    assert update.json()["asking_price_inr"] == "525000.25"
    assert details_update.json()["asking_price_inr"] == "525000.25"
    assert details_update.json()["vehicle_details"]["make"] == "Example"
    assert clear.json()["asking_price_inr"] is None


def test_list_assessments_paginates_and_reports_total(client):
    ids = []
    for index in range(3):
        response = client.post("/api/v1/assessments", json={"vehicle_details": {"index": index}})
        assert response.status_code == 201
        ids.append(response.json()["id"])

    response = client.get("/api/v1/assessments?limit=2&offset=1")

    assert response.status_code == 200
    assert response.json()["total"] == 3
    assert response.json()["limit"] == 2
    assert response.json()["offset"] == 1
    assert len(response.json()["items"]) == 2
    assert len({item["id"] for item in response.json()["items"]}) == 2


def test_patch_updates_only_permitted_fields(client):
    created = client.post("/api/v1/assessments", json={"vin": "OLDVIN"}).json()

    response = client.patch(
        f"/api/v1/assessments/{created['id']}",
        json={"status": "processing", "registration_number": "MH12XY0001"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "processing"
    assert response.json()["vin"] == "OLDVIN"
    assert response.json()["registration_number"] == "MH12XY0001"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"status": None},
        {"status": "unknown"},
        {"id": "00000000-0000-0000-0000-000000000000"},
        {"vin": "V" * 18},
    ],
)
def test_patch_rejects_invalid_or_unpermitted_fields(client, payload):
    created = client.post("/api/v1/assessments", json={}).json()

    response = client.patch(f"/api/v1/assessments/{created['id']}", json=payload)

    assert response.status_code == 422


def test_create_rejects_unpermitted_fields(client):
    response = client.post("/api/v1/assessments", json={"status": "completed"})

    assert response.status_code == 422


def test_missing_and_malformed_assessment_ids(client):
    missing_id = "00000000-0000-0000-0000-000000000001"

    assert client.get(f"/api/v1/assessments/{missing_id}").status_code == 404
    assert client.patch(f"/api/v1/assessments/{missing_id}", json={"status": "failed"}).status_code == 404
    assert client.get("/api/v1/assessments/not-a-uuid").status_code == 422


def test_assessment_routes_appear_in_openapi_docs(client):
    paths = client.get("/openapi.json").json()["paths"]

    assert set(paths["/api/v1/assessments"]) == {"get", "post"}
    assert set(paths["/api/v1/assessments/{assessment_id}"]) == {"get", "patch"}


@pytest.mark.parametrize(
    ("method", "path", "operation", "expected_detail"),
    [
        ("get", "/api/v1/assessments?limit=1", "list_assessments", "Unable to list assessments."),
        ("post", "/api/v1/assessments", "create_assessment", "Unable to create assessment."),
    ],
)
def test_assessment_database_errors_are_logged_safely_and_keep_generic_responses(
    client, caplog, method, path, operation, expected_detail
):
    class DriverFailure(Exception):
        sqlstate = "08006"

        def __str__(self):
            return (
                "connection refused; "
                "postgresql://dbuser:super-secret@db.example/vehicle_iq "
                "password=super-secret token=secret-token"
            )

    class FailingSession:
        def add(self, _item):
            pass

        def scalar(self, _statement):
            raise OperationalError("statement", {}, DriverFailure("private connection detail"))

        def commit(self):
            raise OperationalError("statement", {}, DriverFailure("private connection detail"))

        def rollback(self):
            pass

    def override_get_db():
        yield FailingSession()

    app.dependency_overrides[get_db] = override_get_db
    caplog.set_level("ERROR", logger="app.assessments")
    try:
        response = client.request(method, path, json={} if method == "post" else None)
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 500
    assert response.json() == {"detail": expected_detail}

    record = next(record for record in caplog.records if record.name == "app.assessments")
    event = json.loads(record.getMessage())
    assert event == {
        "event": "assessment_database_error",
        "operation": operation,
        "request_id": response.headers["x-request-id"],
        "exception_class": "OperationalError",
        "underlying_exception_class": "DriverFailure",
        "sqlstate": "08006",
        "error_category": "connection_refused",
        "error_message": "Database connection was refused.",
    }
    assert "super-secret" not in caplog.text
    assert "secret-token" not in caplog.text
    assert "dbuser" not in caplog.text
    assert "db.example" not in caplog.text
    assert "vehicle_iq" not in caplog.text


@pytest.mark.parametrize(
    ("driver_message", "expected_category", "expected_message"),
    [
        ("password authentication failed for user 'private-user'", "authentication", "Database authentication failed."),
        ('could not translate host name "private-host" to address: name resolution failed', "dns", "Database host name could not be resolved."),
        ("connection timed out", "timeout", "Database connection timed out."),
        ("SSL certificate verify failed", "ssl_tls", "Database TLS/SSL negotiation failed."),
        ("network is unreachable", "network", "Database network connection failed."),
        ("server sent an unrecognized startup parameter", "other", "Database operation failed; driver details were omitted."),
    ],
)
def test_database_error_messages_map_to_safe_diagnostic_categories(
    client, caplog, driver_message, expected_category, expected_message
):
    from app.api.v1.assessments import _safe_database_message

    category, safe_message = _safe_database_message(driver_message)

    assert (category, safe_message) == (expected_category, expected_message)
