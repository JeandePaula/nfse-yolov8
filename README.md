# NFS-e Yolov8

[Leia em Português](README-pt.md)

## Overview
This project provides an API for data extraction from images and PDFs using YOLO, FastAPI, and Tesseract.

## Requirements
- Python >= 3.11

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/JeandePaula/nfse-yolov8.git
    cd nfse-yolov8
    ```

2. **Create a virtual environment (optional but recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/MacOS
    venv\Scripts\activate  # For Windows
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

   Or manually install:
    ```bash
    pip install fastapi>=0.112.1 uvicorn>=0.30.6 pytesseract>=0.3.13 ultralytics>=8.2.77 numpy>=1.26.4,<2.0.0 opencv-python>=4.10.0,<5.0.0 python-multipart>=0.0.9 pdf2image>=1.16.3
    ```

## Running the Application

1. **Start the FastAPI server**:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

2. **Send a POST request**:
    You can send a `POST` request to the `/extract_data` endpoint using `curl` or an HTTP client like Postman. Example using `curl`:

    - For an image:
        ```bash
        curl -X POST "http://0.0.0.0:8000/extract_data" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/nf-image.jpg"
        ```

    - For a PDF:
        ```bash
        curl -X POST "http://0.0.0.0:8000/extract_data" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/nf-pdf.pdf"
        ```

## Expected Output

1. **Generated Image**:
   After processing, an image named `output_image.png` will be saved in the root of the project, as per the following example:
   ![Output Image](output_image.png)

2. **Sample JSON Output**:
    ```json
    {
        "service-description": "“o e mil e quientos reais)",
        "iss-value": "75,00",
        "nf-number": "00000001",
        "credit-value": "22,50",
        "service-value": "R$ 1.500,00",
        "nf-issue-date": "1/6/2006 12:01:40",
        "total-deductions": "0,00",
        "tax-percentage-rate": "5,00%",
        "nf-verification-code": "TPUG-ZK2J",
        "municipal-registration": "—",
        "address": "Address: AV SAPOPEMBA 0520 - VILA GUARANI - ZIP: 03374-001",
        "document-time": "20060601 1200",
        "service-code-description": "05690 - HIGHER EDUCATION, UNDERGRADUATE COURSES, AND OTHER COURSES",
        "calculation-base-value": "1.500,00",
        "issuer-name": "'PF/CNPJ: 64.167.648/0001-33",
        "issuer-municipal-registration": "1.311.306-8",
        "recipient-document": "DPF/CNPJ: 007.725.827-73",
        "recipient-email": "-rmail: sandromunizrochaQBinternet.com.br",
        "recipient-state": "SP",
        "recipient-name": "SANDRO MUNIZ ROCHA",
        "issuer-city": "São Paulo",
        "issuer-state": "sP"
    }
    ```

## Important Note
**Note:** The model used in this project was not properly trained due to insufficient images of Service Invoices (NFS). This may result in discrepancies or inaccuracies in the extracted data. It is recommended to improve the model's training with a broader and more diverse dataset to enhance accuracy.

## Contribution
Feel free to open issues and submit pull requests to improve this project.

## License
This project is licensed under the terms of the MIT license.
