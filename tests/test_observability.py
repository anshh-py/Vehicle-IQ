import json
import logging

from fastapi.testclient import TestClient

from app.main import app


def test_request_id_and_safe_route_timing_are_logged(caplog):
    caplog.set_level(logging.INFO, logger="app.request")
    secret_marker = "private-registration-query"

    response = TestClient(app).get(f"/health/live?registration={secret_marker}")

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32
    record = next(record for record in caplog.records if record.name == "app.request")
    event = json.loads(record.message)
    assert event["event"] == "http_request"
    assert event["request_id"] == request_id
    assert event["method"] == "GET"
    assert event["route"] == "/health/live"
    assert event["status_code"] == 200
    assert event["duration_ms"] >= 0
    assert secret_marker not in record.message
