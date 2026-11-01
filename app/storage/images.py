from __future__ import annotations

from pathlib import Path
from typing import Protocol
from urllib.parse import quote
from uuid import uuid4

import httpx

from app.core.config import settings
from app.core.http import ExternalResponseTooLarge, request_with_response_limit


class ImageStorageError(Exception):
    """Raised when an image cannot be saved to or removed from storage."""


class ImageStorage(Protocol):
    provider_name: str
    bucket: str

    def save(self, content: bytes, extension: str) -> str: ...

    def read(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class LocalImageStorage:
    """Local development adapter storing images under an isolated root directory."""

    _ALLOWED_EXTENSIONS = {"jpg", "png", "webp"}
    provider_name = "local"
    bucket = "vehicle-images"

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()

    def save(self, content: bytes, extension: str) -> str:
        if extension.lower() not in self._ALLOWED_EXTENSIONS:
            raise ImageStorageError("Unsupported storage extension.")
        extension = extension.lower()

        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            key = f"{uuid4().hex}.{extension}"
            path = self._path_for_key(key)
            try:
                with path.open("xb") as image_file:
                    image_file.write(content)
                    image_file.flush()
                return key
            except FileExistsError:
                continue
            except OSError as exc:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise ImageStorageError("Unable to store image.") from exc
        raise ImageStorageError("Unable to allocate a unique image key.")

    def delete(self, key: str) -> None:
        path = self._path_for_key(key)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise ImageStorageError("Unable to remove stored image.") from exc

    def read(self, key: str) -> bytes:
        path = self._path_for_key(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ImageStorageError("Unable to read stored image.") from exc

    def _path_for_key(self, key: str) -> Path:
        if not key or Path(key).name != key or "/" in key or "\\" in key:
            raise ImageStorageError("Invalid storage key.")
        stem, separator, extension = key.partition(".")
        if not separator or len(stem) != 32 or any(char not in "0123456789abcdef" for char in stem):
            raise ImageStorageError("Invalid storage key.")
        if extension not in self._ALLOWED_EXTENSIONS:
            raise ImageStorageError("Invalid storage key.")

        path = (self.root / key).resolve()
        if path.parent != self.root:
            raise ImageStorageError("Invalid storage key.")
        return path


class SupabaseImageStorage:
    """Private-bucket adapter using the Supabase Storage HTTP API."""

    provider_name = "supabase"
    _CONTENT_TYPES = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}

    def __init__(
        self,
        project_url: str,
        bucket: str,
        service_role_key: str,
        *,
        timeout_seconds: float = 30.0,
        max_image_bytes: int = 10 * 1024 * 1024,
        max_response_bytes: int = 2 * 1024 * 1024,
        transport: httpx.BaseTransport | None = None,
    ):
        self.project_url = project_url.rstrip("/")
        self.bucket = bucket
        self._service_role_key = service_role_key
        self.timeout_seconds = timeout_seconds
        self.max_image_bytes = max_image_bytes
        self.max_response_bytes = max_response_bytes
        self.transport = transport

    def save(self, content: bytes, extension: str) -> str:
        extension = extension.lower()
        content_type = self._CONTENT_TYPES.get(extension)
        if content_type is None:
            raise ImageStorageError("Unsupported storage extension.")
        key = f"{uuid4().hex}.{extension}"
        response = self._request(
            "POST",
            self._object_url(key),
            max_bytes=self.max_response_bytes,
            content=content,
            headers={"Content-Type": content_type, "x-upsert": "false"},
        )
        if not response.is_success:
            raise ImageStorageError("Unable to store image in Supabase Storage.")
        return key

    def read(self, key: str) -> bytes:
        self._validate_key(key)
        response = self._request("GET", self._object_url(key), max_bytes=self.max_image_bytes)
        if response.status_code == 404:
            raise ImageStorageError("Stored image was not found.")
        if not response.is_success:
            raise ImageStorageError("Unable to read image from Supabase Storage.")
        return response.content

    def delete(self, key: str) -> None:
        self._validate_key(key)
        response = self._request("DELETE", self._object_url(key), max_bytes=self.max_response_bytes)
        if response.status_code == 404:
            return
        if not response.is_success:
            raise ImageStorageError("Unable to remove image from Supabase Storage.")

    def _request(self, method: str, url: str, *, max_bytes: int, **kwargs) -> httpx.Response:
        headers = {
            "apikey": self._service_role_key,
            "Authorization": f"Bearer {self._service_role_key}",
            **kwargs.pop("headers", {}),
        }
        try:
            with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                return request_with_response_limit(
                    client,
                    method,
                    url,
                    max_bytes=max_bytes,
                    headers=headers,
                    **kwargs,
                )
        except (httpx.TimeoutException, httpx.RequestError, ExternalResponseTooLarge) as exc:
            raise ImageStorageError("Supabase Storage request failed safely.") from exc

    def _object_url(self, key: str) -> str:
        return (
            f"{self.project_url}/storage/v1/object/"
            f"{quote(self.bucket, safe='')}/{quote(key, safe='')}"
        )

    @staticmethod
    def _validate_key(key: str) -> None:
        if (
            not key
            or Path(key).name != key
            or "/" in key
            or "\\" in key
            or len(key) not in {36, 37}
            or key[32] != "."
            or key[:32].lower() != key[:32]
            or any(char not in "0123456789abcdef" for char in key[:32])
            or key[33:] not in {"jpg", "png", "webp"}
        ):
            raise ImageStorageError("Invalid storage key.")


def get_image_storage() -> ImageStorage:
    if settings.image_storage_provider == "supabase":
        if not settings.supabase_url or not settings.supabase_storage_bucket or not settings.supabase_service_role_key:
            raise ImageStorageError("Supabase Storage is not configured.")
        return SupabaseImageStorage(
            settings.supabase_url,
            settings.supabase_storage_bucket,
            settings.supabase_service_role_key.get_secret_value(),
            timeout_seconds=settings.supabase_storage_timeout_seconds,
            max_image_bytes=settings.max_image_upload_bytes,
            max_response_bytes=settings.max_external_response_bytes,
        )
    return LocalImageStorage(settings.image_storage_dir)
