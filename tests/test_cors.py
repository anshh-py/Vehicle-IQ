import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://vehicle-iq-frontend.vercel.app",
    ],
)
def test_assessment_create_preflight_allows_frontend_origins(origin):
    with TestClient(app) as client:
        response = client.options(
            "/api/v1/assessments",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin
    assert "POST" in response.headers["access-control-allow-methods"]
    assert response.headers["access-control-allow-headers"] == (
        "authorization,content-type"
    )
    assert "access-control-allow-credentials" not in response.headers
