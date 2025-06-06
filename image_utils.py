import cv2
import numpy as np
from pdf2image import convert_from_bytes


def preprocess_image(image):
    """Apply preprocessing steps to improve OCR results."""
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def convert_pdf_to_images(pdf_bytes):
    """Convert a PDF file into a list of images."""
    images = convert_from_bytes(pdf_bytes)
    return [np.array(image) for image in images]
