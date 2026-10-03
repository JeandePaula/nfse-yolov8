from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from nfse import service
from nfse.documents import DocumentError


@pytest.fixture
def extractor(monkeypatch, settings):
    monkeypatch.setattr(service.pytesseract, "get_languages", lambda config: ["por"])
    model = Mock()
    boxes = [
        SimpleNamespace(xyxy=np.array([[-2, -4, 200, 80]]), cls=np.array(0), conf=np.array(0.9)),
        SimpleNamespace(xyxy=np.array([[10, 10, 30, 30]]), cls=np.array(0), conf=np.array(0.8)),
        SimpleNamespace(xyxy=np.array([[0, 0, 0, 0]]), cls=np.array(0), conf=np.array(0.9)),
    ]
    model.predict.return_value = [SimpleNamespace(boxes=boxes, names={0: "nf-numero"})]
    monkeypatch.setattr(service, "YOLO", lambda path: model)
    return service.ExtractionService(settings)


def test_clips_crops_skips_empty_and_preserves_duplicates(extractor, monkeypatch, image_bytes):
    monkeypatch.setattr(extractor, "_ocr", lambda crop, label: "123")
    result = extractor.extract(image_bytes)
    fields = result.pages[0].fields
    assert len(fields) == 2
    assert fields[0].bbox == (0, 0, 100, 60)
    assert fields[1].bbox == (10, 10, 30, 30)
    assert [field.label for field in fields] == ["nf-numero", "nf-numero"]


def test_busy_requests_are_rejected(extractor, image_bytes):
    with service._PROCESS_LOCK, pytest.raises(service.ServiceBusyError):
        extractor.extract(image_bytes)


def test_failure_releases_service_lock(extractor, image_bytes, monkeypatch):
    with pytest.raises(DocumentError):
        extractor.extract(b"invalid")
    monkeypatch.setattr(extractor, "_ocr", lambda crop, label: "123")
    assert extractor.extract(image_bytes).pages


def test_pdf_rgb_is_converted_to_bgr(extractor, monkeypatch, pdf_bytes):
    monkeypatch.setattr(extractor, "_ocr", lambda crop, label: "123")
    result = extractor.extract(pdf_bytes)
    assert len(result.pages) == 2
    bgr = extractor.model.predict.call_args_list[0].kwargs["source"]
    assert bgr[0, 0, 2] > 200 and bgr[0, 0, 0] < 20


def test_multiline_ocr_and_timeout(extractor, monkeypatch):
    mock = Mock(return_value="  first\nsecond  ")
    monkeypatch.setattr(service.pytesseract, "image_to_string", mock)
    text = extractor._ocr(np.zeros((20, 40, 3), dtype=np.uint8), "servico-descricao")
    assert text == "first\nsecond"
    assert mock.call_args.kwargs["config"] == "--psm 6"
    mock.side_effect = RuntimeError("Tesseract process timeout")
    with pytest.raises(service.OCRTimeoutError):
        extractor._ocr(np.zeros((20, 40, 3), dtype=np.uint8), "nf-numero")


def test_very_narrow_crop_remains_nonempty(extractor, monkeypatch):
    mock = Mock(return_value="")
    monkeypatch.setattr(service.pytesseract, "image_to_string", mock)
    extractor._ocr(np.zeros((5000, 1, 3), dtype=np.uint8), "nf-numero")
    assert mock.call_args.args[0].shape == (2400, 1)
