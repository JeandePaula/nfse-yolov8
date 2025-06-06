from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np

from image_utils import preprocess_image, convert_pdf_to_images
from ocr_utils import extract_text
from validation import orientation_is_valid

app = FastAPI()

# Maximum allowed file size (5 MB)
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@app.post("/extract_data")
async def extract_and_save_text_from_regions(file: UploadFile = File(...)):
    # Check the file size
    file_size = len(await file.read())
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds the 5 MB limit.")
    
    # Reopen the file for further processing
    file.file.seek(0)
    
    # Check if the file is an image or a PDF
    if file.content_type == 'application/pdf':
        pdf_bytes = await file.read()
        images = convert_pdf_to_images(pdf_bytes)
    elif file.content_type.startswith('image/'):
        img_bytes = await file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        images = [cv2.imdecode(img_np, cv2.IMREAD_COLOR)]
    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Only images or PDFs are allowed.")
    
    extracted_data = {}
    model = YOLO("best-nfse.pt")

    for img in images:
        if not orientation_is_valid(img):
            raise HTTPException(status_code=400, detail="NFS is not valid")

        results = model(img)
        results[0].save("./output_image.png")
        for i, result in enumerate(results):
            for j, box in enumerate(result.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = x1 + 1, x2 + 1
                roi = img[y1:y2, x1:x2]
                roi = preprocess_image(roi)
                text = extract_text(roi)
                class_id = int(box.cls[0])
                label = model.names[class_id]
                extracted_data[label] = text

    return extracted_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
