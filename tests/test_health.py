from fastapi.testclient import TestClient
import logging

from app.main import app
from app.db.session import get_db

client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["name"] == "Vehicle IQ"


def test_liveness():
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_failure_is_logged_without_database_details(caplog):
    class BrokenDatabase:
        def execute(self, _statement):
            raise RuntimeError("private database connection detail")

    app.dependency_overrides[get_db] = lambda: BrokenDatabase()
    caplog.set_level(logging.WARNING, logger="app.main")
    try:
        response = client.get("/health/ready")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Database unavailable"}
    assert "RuntimeError" in caplog.text
    assert "private database connection detail" not in caplog.text
