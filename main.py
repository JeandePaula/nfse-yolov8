from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import pytesseract
import numpy as np
from pdf2image import convert_from_bytes

app = FastAPI()

# Tamanho máximo do arquivo permitido (5 MB)
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Função para pré-processar a imagem usando OpenCV
def preprocess_image(image):
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Função para extrair texto usando pytesseract
def extract_text(image, changeConfigToOrientation=False):
    custom_config = '--psm 6 --oem 3' if changeConfigToOrientation else '--psm 7'
    text = pytesseract.image_to_string(image, lang='por', config=custom_config).strip()
    return text

def orientation_is_valid(image):
    keywords = [
        "PREFEITURA", "PRESTADOR", "TOMADOR", "VALOR", 
        "NOTA", "FISCAL", "CPF", "CNPJ",
    ]
    img = preprocess_image(image)
    text = extract_text(img, changeConfigToOrientation=True)
    return any(keyword.lower() in text.lower() for keyword in keywords)

# Função para converter PDF em imagens
def convert_pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return [np.array(image) for image in images]

@app.post("/extract_data")
async def extract_and_save_text_from_regions(file: UploadFile = File(...)):
    # Verifica o tamanho do arquivo
    file_size = len(await file.read())
    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds the 5 MB limit.")
    
    # Reabre o arquivo para processamento posterior
    file.file.seek(0)
    
    # Verifica se o arquivo é uma imagem ou um PDF
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
