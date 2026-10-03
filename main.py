"""Compatibility entry point: uvicorn main:app."""

from nfse.api import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("nfse.api:app", host="0.0.0.0", port=8000)
