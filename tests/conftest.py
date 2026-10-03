import io

import pytest
from PIL import Image

from nfse.config import Settings


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture
def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 60), "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def pdf_bytes():
    buffer = io.BytesIO()
    first = Image.new("RGB", (100, 60), "red")
    second = Image.new("RGB", (100, 60), "blue")
    first.save(buffer, format="PDF", save_all=True, append_images=[second])
    return buffer.getvalue()
