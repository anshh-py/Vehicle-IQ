import pytest

from app.core.config import settings


@pytest.fixture(autouse=True)
def disable_live_tavily_requests(monkeypatch):
    monkeypatch.setattr(settings, "tavily_api_key", None)
