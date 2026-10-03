import io

import pytest
from PIL import Image

from nfse.documents import DocumentError, iter_pages


def test_image_decode(image_bytes, settings):
    for image in iter_pages(image_bytes, settings):
        assert image.mode == "RGB"
        assert image.size == (100, 60)


@pytest.mark.parametrize(
    ("data", "status"), [(b"", 400), (b"not an image", 415), (b"%PDF-broken", 422)]
)
def test_invalid_documents(data, status, settings):
    with pytest.raises(DocumentError) as error:
        list(iter_pages(data, settings))
    assert error.value.status_code == status


def test_image_pixel_limit(image_bytes, settings):
    settings.max_pixels = 100
    with pytest.raises(DocumentError, match="pixel limit"):
        list(iter_pages(image_bytes, settings))


def test_transparency_composited_on_white(settings):
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(buffer, format="PNG")
    for image in iter_pages(buffer.getvalue(), settings):
        assert image.getpixel((0, 0)) == (255, 255, 255)


def test_exif_orientation(settings):
    buffer = io.BytesIO()
    image = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif()
    exif[274] = 6
    image.save(buffer, format="JPEG", exif=exif)
    for page in iter_pages(buffer.getvalue(), settings):
        assert page.size == (20, 40)


def test_pdf_preserves_pages_and_colors(pdf_bytes, settings):
    colors = [image.getpixel((0, 0)) for image in iter_pages(pdf_bytes, settings)]
    assert len(colors) == 2
    assert colors[0][0] > 200 and colors[0][2] < 20
    assert colors[1][2] > 200 and colors[1][0] < 20


def test_pdf_page_limit(pdf_bytes, settings):
    settings.max_pages = 1
    with pytest.raises(DocumentError, match="page limit"):
        list(iter_pages(pdf_bytes, settings))


def test_pdf_pixel_limit(pdf_bytes, settings):
    settings.max_pixels = 100
    with pytest.raises(DocumentError, match="pixel limit"):
        list(iter_pages(pdf_bytes, settings))


def test_multiframe_tiff_is_explicitly_rejected(settings):
    buffer = io.BytesIO()
    image = Image.new("RGB", (20, 20))
    image.save(buffer, format="TIFF", save_all=True, append_images=[image])
    with pytest.raises(DocumentError, match="Multi-frame"):
        list(iter_pages(buffer.getvalue(), settings))
