from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NFSE_", env_file=".env", extra="ignore")

    model_path: Path = ROOT / "best-nfse.pt"
    device: str = "cpu"
    confidence: float = Field(default=0.25, gt=0, le=1)
    image_size: int = Field(default=640, ge=32, le=2048, multiple_of=32)
    ocr_language: str = "por"
    ocr_timeout: float = Field(default=15, gt=0, le=120)
    max_file_mb: int = Field(default=5, ge=1, le=100)
    max_pages: int = Field(default=10, ge=1, le=100)
    max_pixels: int = Field(default=20_000_000, ge=1)
    max_detections: int = Field(default=100, ge=1, le=300)
    pdf_dpi: int = Field(default=150, ge=72, le=300)

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024
