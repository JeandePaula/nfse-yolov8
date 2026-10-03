from pydantic import BaseModel, Field


class DetectedField(BaseModel):
    label: str
    text: str
    confidence: float = Field(ge=0, le=1, description="YOLO detection score, not OCR accuracy")
    bbox: tuple[int, int, int, int] = Field(description="x1, y1, x2, y2 in rendered page pixels")


class PageResult(BaseModel):
    page: int
    width: int
    height: int
    fields: list[DetectedField]


class ExtractionResult(BaseModel):
    pages: list[PageResult]
