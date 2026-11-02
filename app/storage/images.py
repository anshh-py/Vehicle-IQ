from __future__ import annotations

from app.db.models import UploadedImage


class ImageContentUnavailable(Exception):
    """Raised when an image record has no bytes stored in the database."""


def read_image_content(image: UploadedImage) -> bytes:
    if image.content is None:
        raise ImageContentUnavailable("Image bytes are not present in the database.")
    return bytes(image.content)
