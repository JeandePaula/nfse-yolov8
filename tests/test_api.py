import asyncio

import pytest
from fastapi.testclient import TestClient

from nfse import api
from nfse.documents import DocumentError
from nfse.schemas import DetectedField, ExtractionResult, PageResult
from nfse.service import OCRTimeoutError, ServiceBusyError


class FakeExtractor:
    def __init__(self, settings):
        self.calls = 0

    def extract(self, data):
        self.calls += 1
        if not data:
            raise DocumentError("The uploaded file is empty.", 400)
        return ExtractionResult(
            pages=[
                PageResult(
                    page=number,
                    width=100,
                    height=60,
                    fields=[
                        DetectedField(
                            label="nf-numero", text=str(number), confidence=0.9, bbox=(1, 2, 30, 20)
                        )
                    ],
                )
                for number in (1, 2)
            ]
        )


@pytest.fixture
def client(monkeypatch, settings):
    monkeypatch.setattr(api, "ExtractionService", FakeExtractor)
    settings.max_file_mb = 1
    with TestClient(api.create_app(settings)) as client:
        yield client


def test_health_and_model_reuse(client, image_bytes):
    assert client.get("/health").json() == {"status": "ok"}
    for _ in range(2):
        response = client.post("/extract", files={"file": ("a.png", image_bytes, "image/png")})
        assert response.status_code == 200
        assert [p["fields"][0]["text"] for p in response.json()["pages"]] == ["1", "2"]
    assert client.app.state.extractor.calls == 2


def test_legacy_contract(client, image_bytes):
    response = client.post("/extract_data", files={"file": ("a.png", image_bytes)})
    assert response.status_code == 200
    assert response.json() == {"nf-numero": "2"}


def test_empty_upload(client):
    assert client.post("/extract", files={"file": ("a.png", b"")}).status_code == 400


def test_missing_file(client):
    assert client.post("/extract").status_code == 422


@pytest.mark.parametrize("size", [1024 * 1024 + 1, 2 * 1024 * 1024])
def test_file_and_request_limits(client, size):
    response = client.post("/extract", files={"file": ("a.png", b"x" * size)})
    assert response.status_code == 413
    assert client.app.state.extractor.calls == 0


@pytest.mark.parametrize(
    ("error", "status"),
    [(DocumentError("broken"), 422), (ServiceBusyError(), 503), (OCRTimeoutError(), 504)],
)
def test_processing_errors(client, monkeypatch, error, status):
    def fail(data):
        raise error

    monkeypatch.setattr(client.app.state.extractor, "extract", fail)
    response = client.post("/extract", files={"file": ("a.png", b"data")})
    assert response.status_code == status
    if status == 503:
        assert response.headers["Retry-After"] == "1"


def test_unexpected_error_does_not_leak_details(client, monkeypatch):
    def fail(data):
        raise RuntimeError("private path")

    monkeypatch.setattr(client.app.state.extractor, "extract", fail)
    response = client.post("/extract", files={"file": ("a.png", b"data")})
    assert response.status_code == 500
    assert "private path" not in response.text


def test_startup_failure_is_visible(monkeypatch, settings):
    def fail(settings):
        raise RuntimeError("Model not found")

    monkeypatch.setattr(api, "ExtractionService", fail)
    with pytest.raises(RuntimeError, match="Model not found"), TestClient(api.create_app(settings)):
        pass


def test_openapi_documents_both_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/extract" in paths
    assert paths["/extract_data"]["post"]["deprecated"]


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")]])
def test_chunked_body_limit_does_not_trust_content_length(headers):
    async def run():
        chunks = iter(
            [
                {"type": "http.request", "body": b"ab", "more_body": True},
                {"type": "http.request", "body": b"cd", "more_body": False},
            ]
        )
        sent = []

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)

        async def consumer(scope, receive, send):
            while (await receive()).get("more_body", False):
                pass
            raise AssertionError("Oversized body reached the handler")

        middleware = api.UploadLimitMiddleware(consumer, max_bytes=3)
        await middleware({"type": "http", "headers": headers}, receive, send)
        assert sent[0]["status"] == 413

    asyncio.run(run())
