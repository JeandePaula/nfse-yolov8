import logging
from contextlib import asynccontextmanager, suppress
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from nfse.config import Settings
from nfse.documents import DocumentError
from nfse.schemas import ExtractionResult
from nfse.service import ExtractionService, OCRTimeoutError, ServiceBusyError

logger = logging.getLogger(__name__)


class _BodyTooLarge(MultiPartException):
    pass


class UploadLimitMiddleware:
    """Bound the request body before multipart parsing writes uploads to disk."""

    def __init__(self, app: ASGIApp, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        size = 0
        exceeded = False

        async def limited_receive() -> Message:
            nonlocal size, exceeded
            message = await receive()
            if message["type"] == "http.request":
                size += len(message.get("body", b""))
                if size > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge("Request exceeds the upload limit.")
            return message

        async def limited_send(message: Message) -> None:
            # Multipart parsing may convert the exception to 400; emit our 413 instead.
            if not exceeded:
                await send(message)

        with suppress(_BodyTooLarge):
            await self.app(scope, limited_receive, limited_send)
        if exceeded:
            await JSONResponse({"detail": "Request exceeds the upload limit."}, status_code=413)(
                scope, receive, send
            )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.extractor = await run_in_threadpool(ExtractionService, settings)
        yield
        del app.state.extractor

    app = FastAPI(title="NFS-e Extractor", version="2.0.0", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, max_bytes=settings.max_file_bytes + 64 * 1024)

    @app.get("/health", tags=["Operations"])
    def health(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "extractor"):
            raise HTTPException(503, "Extractor is not ready.")
        return {"status": "ok"}

    def process(request: Request, file: UploadFile) -> ExtractionResult:
        try:
            data = file.file.read(settings.max_file_bytes + 1)
            if len(data) > settings.max_file_bytes:
                raise HTTPException(413, "File exceeds the configured size limit.")
            return request.app.state.extractor.extract(data)
        except DocumentError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        except ServiceBusyError as exc:
            raise HTTPException(
                503, "Extractor is busy. Retry shortly.", {"Retry-After": "1"}
            ) from exc
        except OCRTimeoutError as exc:
            raise HTTPException(504, "OCR exceeded its time limit.") from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Document extraction failed")
            raise HTTPException(500, "Document extraction failed.") from exc
        finally:
            file.file.close()

    @app.post("/extract", response_model=ExtractionResult, tags=["Extraction"])
    def extract(request: Request, file: Annotated[UploadFile, File()]) -> ExtractionResult:
        """Extract all fields, retaining page numbers, duplicate labels and detection scores."""
        return process(request, file)

    @app.post("/extract_data", response_model=dict[str, str], tags=["Extraction"], deprecated=True)
    def extract_legacy(request: Request, file: Annotated[UploadFile, File()]) -> dict[str, str]:
        """Compatibility response: duplicate labels use the last detection on the last page."""
        result = process(request, file)
        return {field.label: field.text for page in result.pages for field in page.fields}

    return app


app = create_app()
