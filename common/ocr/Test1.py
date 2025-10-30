"""
Compares OCR results from Tesseract, EasyOCR, and PaddleOCR
and saves them to 'ocr_results.txt'.

Requirements (already installed in your env):
- pytesseract
- easyocr
- paddleocr==2.6.1
- paddlepaddle==2.6.2
"""

import os
import pytesseract
import easyocr
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image

# ---- Compatibility patch for numpy ≥1.20 ----
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
# ---------------------------------------------

# === File paths ===
ENG_PATH = "/Users/Ekaterina/PycharmProjects/AI_Synapse/ocr_samples/eng_sample.png"
KOR_PATH = "/Users/Ekaterina/PycharmProjects/AI_Synapse/ocr_samples/kor_sample.png"

# === Output file ===
OUTPUT_FILE = "ocr_results.txt"

# === Create output folder ===
dir_path = os.path.dirname(OUTPUT_FILE)
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

# === Initialize OCR engines ===
print("🔹 Initializing OCR engines...")

paddle_en = PaddleOCR(lang='en', use_angle_cls=True)
paddle_ko = PaddleOCR(lang='korean', use_angle_cls=True)
easy_reader = easyocr.Reader(['en', 'ko'])

# === Run tests ===
results = []

def run_tesseract(image_path, lang):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang=lang).strip()

def run_easyocr(image_path):
    return "\n".join(easy_reader.readtext(image_path, detail=0))

def run_paddleocr(image_path, ocr_engine):
    res = ocr_engine.ocr(image_path)
    return "\n".join([line[1][0] for line in res[0]]) if res and len(res[0]) > 0 else "(no text)"

# === English ===
print("🔹 Running OCR on English sample...")
tess_en = run_tesseract(ENG_PATH, 'eng')
easy_en = run_easyocr(ENG_PATH)
paddle_en_res = run_paddleocr(ENG_PATH, paddle_en)

# === Korean ===
print("🔹 Running OCR on Korean sample...")
tess_ko = run_tesseract(KOR_PATH, 'kor')
easy_ko = run_easyocr(KOR_PATH)
paddle_ko_res = run_paddleocr(KOR_PATH, paddle_ko)

# === Combine Results ===
results.append("===== OCR COMPARISON REPORT =====\n")
results.append("🗂 English sample:\n")
results.append(f"Tesseract:\n{tess_en}\n")
results.append(f"EasyOCR:\n{easy_en}\n")
results.append(f"PaddleOCR:\n{paddle_en_res}\n")

results.append("\n🗂 Korean sample:\n")
results.append(f"Tesseract:\n{tess_ko}\n")
results.append(f"EasyOCR:\n{easy_ko}\n")
results.append(f"PaddleOCR:\n{paddle_ko_res}\n")

# === Save to file ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print("\n✅ OCR comparison completed!")
print(f"Results saved to: {os.path.abspath(OUTPUT_FILE)}")
