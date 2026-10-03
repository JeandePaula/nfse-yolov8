import math
from contextlib import closing
from threading import Lock

import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

from nfse.config import Settings
from nfse.documents import iter_pages
from nfse.schemas import DetectedField, ExtractionResult, PageResult

# PDFium and YOLO predictor state must not be used concurrently in the same process.
_PROCESS_LOCK = Lock()


class ServiceBusyError(Exception):
    pass


class OCRTimeoutError(Exception):
    pass


class ExtractionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.model_path.is_file():
            raise RuntimeError(f"Model not found: {settings.model_path}")
        languages = set(pytesseract.get_languages(config=""))
        missing = set(settings.ocr_language.split("+")) - languages
        if missing:
            raise RuntimeError(f"Install missing Tesseract languages: {', '.join(sorted(missing))}")
        # Load only trusted operator-configured checkpoints, never uploaded model files.
        self.model = YOLO(str(settings.model_path))

    def extract(self, data: bytes) -> ExtractionResult:
        if not _PROCESS_LOCK.acquire(blocking=False):
            raise ServiceBusyError
        try:
            pages = []
            with closing(iter_pages(data, self.settings)) as images:
                for number, image in enumerate(images, start=1):
                    # Ultralytics numpy inputs are BGR; Pillow/PDFium provide RGB.
                    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                    result = self.model.predict(
                        source=bgr,
                        conf=self.settings.confidence,
                        imgsz=self.settings.image_size,
                        device=self.settings.device,
                        max_det=self.settings.max_detections,
                        verbose=False,
                        save=False,
                    )[0]
                    fields = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            coords = box.xyxy[0].tolist()
                            if not all(math.isfinite(value) for value in coords):
                                continue
                            x1 = max(0, min(image.width, math.floor(coords[0])))
                            y1 = max(0, min(image.height, math.floor(coords[1])))
                            x2 = max(0, min(image.width, math.ceil(coords[2])))
                            y2 = max(0, min(image.height, math.ceil(coords[3])))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            label = result.names[int(box.cls.item())]
                            text = self._ocr(bgr[y1:y2, x1:x2], label)
                            fields.append(
                                DetectedField(
                                    label=label,
                                    text=text,
                                    confidence=float(box.conf.item()),
                                    bbox=(x1, y1, x2, y2),
                                )
                            )
                    pages.append(
                        PageResult(
                            page=number, width=image.width, height=image.height, fields=fields
                        )
                    )
            return ExtractionResult(pages=pages)
        finally:
            _PROCESS_LOCK.release()

    def _ocr(self, crop: np.ndarray, label: str) -> str:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scale = min(2.0, 2400 / max(gray.shape))
        size = (max(1, round(gray.shape[1] * scale)), max(1, round(gray.shape[0] * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_CUBIC)
        mode = 6 if "descricao" in label or "description" in label else 7
        try:
            return pytesseract.image_to_string(
                gray,
                lang=self.settings.ocr_language,
                config=f"--psm {mode}",
                timeout=self.settings.ocr_timeout,
            ).strip()
        except pytesseract.TesseractError:
            raise
        except RuntimeError as exc:
            raise OCRTimeoutError from exc
