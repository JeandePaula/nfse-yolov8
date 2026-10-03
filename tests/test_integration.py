import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from nfse.api import create_app
from nfse.config import ROOT
from nfse.service import ExtractionService

pytestmark = pytest.mark.integration


def test_real_checkpoint_image_pdf_and_portuguese_ocr(settings):
    image = Image.new("RGB", (900, 180), "white")
    font = ImageFont.truetype("DejaVuSans.ttf", 48)
    ImageDraw.Draw(image).text((20, 40), "NOTA FISCAL 12345", fill="black", font=font)
    extractor = ExtractionService(settings)
    text = extractor._ocr(np.asarray(image), "nf-numero")
    assert "12345" in text
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    result = extractor.extract(buffer.getvalue())
    assert result.pages[0].width == 900
    buffer = io.BytesIO()
    image.save(buffer, format="PDF", save_all=True, append_images=[image])
    result = extractor.extract(buffer.getvalue())
    assert [page.page for page in result.pages] == [1, 2]


def test_real_app_lifespan_and_extraction(settings, image_bytes):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/extract", files={"file": ("data.bin", image_bytes)})
        assert response.status_code == 200
        page = response.json()["pages"][0]
        assert (page["width"], page["height"]) == (100, 60)
        assert all(field["text"] == "" for field in page["fields"])


def test_historical_example_produces_text(settings):
    # Annotated historical image: integration fixture only, not an accuracy benchmark.
    settings.image_size = 1280
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/extract", files={"file": ("example.png", (ROOT / "output_image.png").read_bytes())}
        )
        assert response.status_code == 200
        fields = response.json()["pages"][0]["fields"]
        assert fields and any(field["text"] for field in fields)
