"""Bounded document decoding, called under the service lock for PDFium thread safety."""

import io
import math
import warnings
from collections.abc import Iterator
from contextlib import closing

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError

from nfse.config import Settings


class DocumentError(Exception):
    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def check_dimensions(width: int, height: int, settings: Settings) -> None:
    if width <= 0 or height <= 0 or width * height > settings.max_pixels:
        raise DocumentError("Page exceeds the configured pixel limit.", 413)


def iter_pages(data: bytes, settings: Settings) -> Iterator[Image.Image]:
    if not data:
        raise DocumentError("The uploaded file is empty.", 400)
    if data[:1024].lstrip().startswith(b"%PDF-"):
        yield from _pdf_pages(data, settings)
    else:
        yield from _image_pages(data, settings)


def _image_pages(data: bytes, settings: Settings) -> Iterator[Image.Image]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG", "WEBP", "TIFF", "BMP"}:
                    raise DocumentError("Use JPEG, PNG, WebP, TIFF, BMP or PDF.", 415)
                if getattr(source, "n_frames", 1) != 1:
                    raise DocumentError("Multi-frame images are not supported; upload a PDF.", 415)
                check_dimensions(*source.size, settings)
                image = ImageOps.exif_transpose(source)
                try:
                    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                        with (
                            image.convert("RGBA") as rgba,
                            Image.new("RGBA", rgba.size, "white") as background,
                            Image.alpha_composite(background, rgba) as composite,
                        ):
                            image_rgb = composite.convert("RGB")
                    else:
                        image_rgb = image.convert("RGB")
                finally:
                    image.close()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise DocumentError("Image exceeds the pixel limit.", 413) from exc
    except UnidentifiedImageError as exc:
        raise DocumentError("Unsupported or unreadable document.", 415) from exc
    except (OSError, ValueError) as exc:
        raise DocumentError("The image is corrupted or unreadable.") from exc
    try:
        yield image_rgb
    finally:
        image_rgb.close()


def _pdf_pages(data: bytes, settings: Settings) -> Iterator[Image.Image]:
    try:
        document = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise DocumentError("The PDF is corrupted or password-protected.") from exc
    with document:
        if len(document) == 0:
            raise DocumentError("The PDF has no pages.")
        if len(document) > settings.max_pages:
            raise DocumentError("PDF exceeds the configured page limit.", 413)
        scale = settings.pdf_dpi / 72
        for index in range(len(document)):
            try:
                with closing(document[index]) as page:
                    width, height = page.get_size()
                    if not math.isfinite(width * height):
                        raise DocumentError("Invalid PDF page dimensions.")
                    check_dimensions(math.ceil(width * scale), math.ceil(height * scale), settings)
                    with closing(page.render(scale=scale)) as bitmap, bitmap.to_pil() as rendered:
                        image = rendered.convert("RGB")
            except pdfium.PdfiumError as exc:
                raise DocumentError("The PDF page could not be rendered.") from exc
            try:
                yield image
            finally:
                image.close()
