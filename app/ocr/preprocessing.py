"""Provider-neutral image enhancement before OCR."""

from io import BytesIO

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


_MIN_LONG_EDGE = 1000
_MAX_LONG_EDGE = 2600


def preprocess_ocr_image(content: bytes) -> bytes:
    """Orient, scale, denoise, and enhance an image without cropping its content."""
    try:
        with Image.open(BytesIO(content)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            longest_edge = max(image.size)
            if longest_edge < _MIN_LONG_EDGE:
                scale = _MIN_LONG_EDGE / longest_edge
                image = image.resize(
                    (round(image.width * scale), round(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            elif longest_edge > _MAX_LONG_EDGE:
                scale = _MAX_LONG_EDGE / longest_edge
                image = image.resize(
                    (round(image.width * scale), round(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )

        image = ImageOps.autocontrast(image, cutoff=1)
        image = image.filter(ImageFilter.MedianFilter(size=3))
        image = ImageEnhance.Contrast(image).enhance(1.25)
        image = ImageEnhance.Sharpness(image).enhance(1.4)
        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()
    except Exception as exc:
        raise ValueError("OCR image preprocessing failed.") from exc
