from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
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
    }

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
