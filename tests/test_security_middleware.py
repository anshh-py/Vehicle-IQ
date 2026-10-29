from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.core.config import settings
from app.main import app


def test_api_remains_open_in_development_without_token(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "api_access_token", None)

    response = TestClient(app).get("/api/v1/not-a-route")

    assert response.status_code == 404


def test_configured_api_token_is_required_and_validated(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "api_access_token", SecretStr("test-only-token"))
    client = TestClient(app)

    denied = client.get("/api/v1/not-a-route")
    allowed_to_route = client.get(
        "/api/v1/not-a-route", headers={"Authorization": "Bearer test-only-token"}
    )

    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == "Bearer"
    assert "test-only-token" not in denied.text
    assert allowed_to_route.status_code == 404


def test_production_api_fails_closed_when_token_is_missing(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "production")
    monkeypatch.setattr(settings, "api_access_token", None)

    response = TestClient(app).get("/api/v1/not-a-route")

    assert response.status_code == 503
    assert response.json() == {"detail": "API access is not configured."}


def test_upload_request_body_is_capped_before_multipart_parsing(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "api_access_token", None)
    monkeypatch.setattr(settings, "max_image_upload_bytes", 32 * 1024)

    response = TestClient(app).post(
        "/api/v1/assessments/00000000-0000-0000-0000-000000000001/images",
        files={"file": ("large.png", b"x" * (100 * 1024), "image/png")},
    )

    assert response.status_code == 413
    assert response.json() == {
        "detail": "Image upload request exceeds the configured size limit."
    }


def test_api_rate_limit_returns_429_and_does_not_throttle_health(monkeypatch):
    monkeypatch.setattr(settings, "app_env", "development")
    monkeypatch.setattr(settings, "api_access_token", None)
    monkeypatch.setattr(settings, "api_rate_limit_per_minute", 2)
    client = TestClient(app, client=("203.0.113.77", 12345))

    first = client.get("/api/v1/not-a-route")
    second = client.get("/api/v1/not-a-route")
    limited = client.get("/api/v1/not-a-route")
    health = client.get("/health/live")

    assert first.status_code == second.status_code == 404
    assert limited.status_code == 429
    assert limited.headers["retry-after"]
    assert limited.json() == {"detail": "API request rate limit exceeded."}
    assert health.status_code == 200
