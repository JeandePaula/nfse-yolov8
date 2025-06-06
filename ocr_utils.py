import pytesseract


def extract_text(image, orientation_mode: bool = False) -> str:
    """Run Tesseract OCR on the provided image."""
    config = '--psm 6 --oem 3' if orientation_mode else '--psm 7'
    text = pytesseract.image_to_string(image, lang='por', config=config).strip()
    return text
