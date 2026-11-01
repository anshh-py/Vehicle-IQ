import httpx
import pytest
from pydantic import SecretStr

from app.core.config import settings
from app.storage.images import ImageStorageError, SupabaseImageStorage, get_image_storage


def test_supabase_storage_upload_read_and_delete_use_private_authenticated_api():
    calls = []

    def handler(request):
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(200, content=b"image-bytes")
        return httpx.Response(200, json={"Key": "object-key"})

    storage = SupabaseImageStorage(
        "https://project.supabase.co",
        "vehicle-images",
        "test-secret-key",
        transport=httpx.MockTransport(handler),
    )

    key = storage.save(b"image-bytes", "png")
    assert key.endswith(".png")
    assert storage.read(key) == b"image-bytes"
    storage.delete(key)

    assert [request.method for request in calls] == ["POST", "GET", "DELETE"]
    assert all(request.url.host == "project.supabase.co" for request in calls)
    assert all(request.headers["apikey"] == "test-secret-key" for request in calls)
    assert all(request.headers["authorization"] == "Bearer test-secret-key" for request in calls)
    assert calls[0].headers["x-upsert"] == "false"


def test_supabase_storage_rejects_unsafe_keys_and_unsupported_extensions():
    storage = SupabaseImageStorage(
        "https://project.supabase.co", "vehicle-images", "test-secret-key",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
    )

    with pytest.raises(ImageStorageError, match="Invalid storage key"):
        storage.read("../secret.png")
    with pytest.raises(ImageStorageError, match="Unsupported storage extension"):
        storage.save(b"image", "exe")


def test_supabase_storage_bounds_download_and_hides_network_errors():
    storage = SupabaseImageStorage(
        "https://project.supabase.co",
        "vehicle-images",
        "test-secret-key",
        max_image_bytes=4,
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=b"12345")),
    )
    key = "a" * 32 + ".png"

    with pytest.raises(ImageStorageError, match="request failed safely"):
        storage.read(key)


def test_storage_factory_selects_supabase_only_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "image_storage_provider", "supabase")
    monkeypatch.setattr(settings, "supabase_url", "https://project.supabase.co")
    monkeypatch.setattr(settings, "supabase_storage_bucket", "vehicle-images")
    monkeypatch.setattr(settings, "supabase_service_role_key", SecretStr("test-secret-key"))

    storage = get_image_storage()

    assert isinstance(storage, SupabaseImageStorage)
    assert storage.provider_name == "supabase"
    assert storage.bucket == "vehicle-images"


def test_storage_factory_fails_safely_instead_of_falling_back_to_local(monkeypatch):
    monkeypatch.setattr(settings, "image_storage_provider", "supabase")
    monkeypatch.setattr(settings, "supabase_url", None)
    monkeypatch.setattr(settings, "supabase_storage_bucket", None)
    monkeypatch.setattr(settings, "supabase_service_role_key", None)

    with pytest.raises(ImageStorageError, match="not configured"):
        get_image_storage()
