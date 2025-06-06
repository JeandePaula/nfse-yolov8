from typing import List

from image_utils import preprocess_image
from ocr_utils import extract_text


KEYWORDS: List[str] = [
    "PREFEITURA", "PRESTADOR", "TOMADOR", "VALOR",
    "NOTA", "FISCAL", "CPF", "CNPJ",
]


def orientation_is_valid(image) -> bool:
    """Check if the orientation of the NFS-e is correct based on keywords."""
    img = preprocess_image(image)
    text = extract_text(img, orientation_mode=True)
    return any(keyword.lower() in text.lower() for keyword in KEYWORDS)
