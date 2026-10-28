from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.core.config import settings


class ImageStorageError(Exception):
    """Raised when an image cannot be saved to or removed from storage."""


class ImageStorage(Protocol):
    def save(self, content: bytes, extension: str) -> str: ...

    def read(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...


class LocalImageStorage:
    """Local development adapter storing images under an isolated root directory."""

    _ALLOWED_EXTENSIONS = {"jpg", "png", "webp"}

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


def get_image_storage() -> ImageStorage:
    return LocalImageStorage(settings.image_storage_dir)
