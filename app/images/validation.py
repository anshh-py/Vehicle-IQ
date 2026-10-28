from io import BytesIO
import warnings

from PIL import Image


SUPPORTED_IMAGE_FORMATS = {
    "JPEG": ("jpg", "image/jpeg"),
    "PNG": ("png", "image/png"),
    "WEBP": ("webp", "image/webp"),
}


class InvalidImageError(ValueError):
    """Raised when bytes are not a supported, structurally valid image."""


def inspect_image(content: bytes) -> tuple[str, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                image_format = image.format
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    raise ValueError("Unsupported image format")
                image.verify()
    except Exception as exc:
        raise InvalidImageError("Invalid or unsupported image") from exc
    return SUPPORTED_IMAGE_FORMATS[image_format]
