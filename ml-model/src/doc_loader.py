import os
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

VALID_EXTS = ('.pdf', '.jpg', '.jpeg', '.png', '.tiff')

def scan_files(root_dir):
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                paths.append(os.path.join(root, f))
    return paths

def extract_text(file_path: str) -> str:
    ext = file_path.lower().split('.')[-1]
    text = ""
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = "".join(p.extract_text() or "" for p in pdf.pages)
        if len(text.strip()) < 50:
            images = convert_from_path(file_path)
            text = "".join(pytesseract.image_to_string(img, lang='rus+eng') for img in images)
    else:
        text = pytesseract.image_to_string(Image.open(file_path), lang='rus+eng')
    return text.strip()
