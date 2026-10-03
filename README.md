# NFS-e Extractor

[Português](README-pt.md) · [CI](https://github.com/JeandePaula/nfse-yolov8/actions)

A small FastAPI service that detects NFS-e fields with YOLO and reads them with Portuguese Tesseract OCR. Images and PDFs are processed locally. No database, external OCR service or task queue is required.

## Run on Linux / WSL

Requires Python 3.12–3.14 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/JeandePaula/nfse-yolov8.git
cd nfse-yolov8
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-por libgl1 libglib2.0-0
uv sync --locked
uv run uvicorn nfse.api:app --host 127.0.0.1 --port 8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs). `uv run python main.py` and `uv run uvicorn main:app` remain supported. The model is loaded once at startup; missing weights or OCR languages cause a clear startup failure.

## Docker

```bash
docker build -t nfse .
docker run --rm -p 127.0.0.1:8000:8000 nfse
```

The container includes the checkpoint, CPU PyTorch, PDFium and Portuguese Tesseract, runs as an unprivileged user, and checks `/health`. Set options with `--env-file .env` after copying `.env.example`. Custom model paths must exist inside the container, for example through a read-only volume mount.

## API

```bash
curl --fail-with-body http://127.0.0.1:8000/extract -F 'file=@nota.pdf'
curl --fail-with-body http://127.0.0.1:8000/extract -F 'file=@nota.png'
curl http://127.0.0.1:8000/health
```

`POST /extract` preserves every detection, including repeated labels and separate PDF pages. Example schema (illustrative values):

```json
{
  "pages": [
    {
      "page": 1,
      "width": 1240,
      "height": 1754,
      "fields": [
        {
          "label": "nf-numero",
          "text": "00000001",
          "confidence": 0.94,
          "bbox": [910, 80, 1150, 135]
        }
      ]
    }
  ]
}
```

Labels are the checkpoint's original class names, listed in `data.yaml`; existing spellings are preserved. Coordinates refer to the rendered, EXIF-corrected page. Confidence is the **detector score**, not an OCR accuracy score. No detections produce an empty `fields` array.

`POST /extract_data` retains the original `{label: text}` contract. As before, the last detection wins for duplicate labels, including across pages. New integrations should use `/extract`.

Accepted input: single-frame JPEG, PNG, WebP, TIFF, BMP, and multi-page PDF. Bytes determine the format, independently of the filename or MIME header. Corrupt, encrypted or unsupported documents return errors; multi-frame images must first be converted to PDF.

| Status | Meaning |
| --- | --- |
| 400 | Empty file or malformed multipart request |
| 413 | File, request, page count or pixel limit exceeded |
| 415 | Unsupported or unrecognizable format |
| 422 | Unreadable document or missing upload field |
| 503 | Busy worker; retry after the `Retry-After` interval |
| 504 | OCR timeout |

## Configuration

Copy `.env.example` to `.env` to override defaults. Environment variables take precedence.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NFSE_MODEL_PATH` | repository `best-nfse.pt` | Trusted local checkpoint |
| `NFSE_DEVICE` | `cpu` | Inference device |
| `NFSE_CONFIDENCE` | `0.25` | Detection threshold |
| `NFSE_IMAGE_SIZE` | `640` | YOLO input size, multiple of 32 |
| `NFSE_OCR_LANGUAGE` | `por` | Installed Tesseract language(s) |
| `NFSE_OCR_TIMEOUT` | `15` | Timeout in seconds per detected region |
| `NFSE_MAX_FILE_MB` | `5` | File limit in MiB |
| `NFSE_MAX_PAGES` | `10` | Maximum PDF pages |
| `NFSE_MAX_PIXELS` | `20000000` | Maximum pixels per page |
| `NFSE_MAX_DETECTIONS` | `100` | Maximum detections per page |
| `NFSE_PDF_DPI` | `150` | PDF render resolution |

The HTTP body is limited to the file allowance plus 64 KiB for multipart overhead, including chunked uploads. PDF pages are decoded one at a time. Detection and OCR run outside the async event loop. One extraction runs per process to protect YOLO and PDFium state; overlapping requests receive 503 instead of accumulating work. Additional Uvicorn workers load independent model copies and use more RAM.

There is no built-in authentication. Use a trusted network or an authenticated reverse proxy for deployment. Uploads may be temporarily spooled by the multipart parser and are closed after the request. The application does not save annotated invoices or extracted text; proxy/access-log policies remain the operator's responsibility. EXIF orientation is handled, but arbitrary scanned-page rotation is not corrected. The old keyword-based “orientation validation” was removed because it did not reliably validate orientation or invoice authenticity.

## Model and training

The bundled `best-nfse.pt` is the original YOLOv8 NFS-e checkpoint. Its limited training data still limits accuracy, and it can produce false detections even on blank pages. This rewrite updates the software, not the model's learned quality. No labeled dataset is included, so no retraining or accuracy claim is made. `result/` and `output_image.png` are historical artifacts, not benchmarks for this version.

New training defaults to the current [YOLO26 nano model](https://docs.ultralytics.com/models/yolo26/). Prepare distinct training and validation sets in YOLO format:

```text
datasets/nfse/
  images/train/  images/val/
  labels/train/  labels/val/
```

Each label file contains `class_id x_center y_center width height`, normalized to 0–1. `data.yaml` preserves the 38 original class IDs. Its `path` is resolved according to Ultralytics dataset settings; use an absolute dataset path when storing data elsewhere. Training and validation must not use the same samples.

```bash
uv run python train.py --data data.yaml --epochs 100 --device cpu
# Continue training from the existing architecture instead:
uv run python train.py --model best-nfse.pt --data data.yaml
```

Ultralytics downloads the base weights on the first training run. Point `NFSE_MODEL_PATH` to the resulting `best.pt` after evaluating it on held-out invoices. The committed lock uses CPU wheels. For CUDA, change the two PyTorch sources in `pyproject.toml` using the [official installation selector](https://pytorch.org/get-started/locally/), regenerate `uv.lock`, and then select `--device 0` / `NFSE_DEVICE=0`.

## Development

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
# Fast tests without loading actual model weights / invoking OCR:
uv run pytest -q -m 'not integration'
```

Integration tests require Portuguese Tesseract and DejaVu Sans (`fonts-dejavu-core` on Ubuntu). They exercise the real checkpoint, image/PDF inference, OCR and application startup. They verify functionality, not invoice extraction accuracy. CI runs on Python 3.12 and 3.14 and builds/starts the Docker image.

`nfse/api.py` handles HTTP and startup, `config.py` settings, `documents.py` bounded decoding, `service.py` detection/OCR, and `schemas.py` response types. `train.py` is the independent training command. Runtime dependencies and development tools live in `pyproject.toml`; `uv.lock` pins the complete resolution. To update, run `uv lock --upgrade`, `uv sync --locked`, then the checks above. Dependabot checks monthly.

Versions were resolved from current stable releases on 2026-10-03: FastAPI 0.142.2, Ultralytics 8.4.172, PyTorch 2.14.1, NumPy 2.5.3, OpenCV 5.0.0.93 and PDFium bindings 5.13.0. The lockfile is authoritative.

## License

The repository's original MIT license is retained in [LICENSE](LICENSE). Dependencies and model artifacts have their own terms; see [Ultralytics licensing](https://docs.ultralytics.com/help/license/).
