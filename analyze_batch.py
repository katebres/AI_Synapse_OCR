"""
Batch OCR analyzer for PDF and image files.
Reads data from data/raw, saves results to data/processed.
"""

import os
import pathlib
import shutil
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import pandas as pd

# === Directory setup ===
BASE_DIR = pathlib.Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
IMG_TMP = OUT_DIR / "tmp_images"
TSV_DIR = OUT_DIR / "tsv"
TXT_DIR = OUT_DIR / "text"
SUMMARY_CSV = OUT_DIR / "summary.csv"

for d in [OUT_DIR, IMG_TMP, TSV_DIR, TXT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Convert PDF to images ===
def pdf_to_images(pdf_path):
    pages = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    stem = pdf_path.stem
    for i, img in enumerate(pages, 1):
        out_path = IMG_TMP / f"{stem}_p{i:03d}.png"
        img.save(out_path, "PNG")
        image_paths.append(out_path)
    return image_paths

# === OCR single image ===
def ocr_image(img_path, lang="kor+eng"):
    text = pytesseract.image_to_string(Image.open(img_path), lang=lang)
    tsv = pytesseract.image_to_data(Image.open(img_path), lang=lang, output_type=pytesseract.Output.DATAFRAME)
    tsv = tsv.dropna(subset=["text"]).reset_index(drop=True)
    return text, tsv

# === Process one file ===
def process_file(path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        image_paths = pdf_to_images(path)
    elif ext in [".jpg", ".jpeg", ".png", ".tif", ".bmp"]:
        image_paths = [path]
    else:
        print(f"Skipping {path.name} (unsupported type)")
        return []

    results = []
    for idx, img in enumerate(image_paths, 1):
        text, tsv = ocr_image(img)
        base = path.stem

        txt_out = TXT_DIR / f"{base}_p{idx:03d}.txt"
        txt_out.write_text(text, encoding="utf-8")

        tsv_out = TSV_DIR / f"{base}_p{idx:03d}.tsv"
        tsv.to_csv(tsv_out, sep="\t", index=False)

        results.append({
            "file": path.name,
            "page_or_image": idx,
            "txt_path": str(txt_out),
            "tsv_path": str(tsv_out),
            "preview": text[:200].replace("\n", " ")
        })
    return results

# === Main ===
def main():
    items = []
    for file_path in RAW_DIR.iterdir():
        if file_path.is_file():
            print(f"Processing {file_path.name} ...")
            items.extend(process_file(file_path))

    pd.DataFrame(items).to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
    shutil.rmtree(IMG_TMP, ignore_errors=True)
    print("\n✅ All files processed!")
    print(f"Summary → {SUMMARY_CSV}")
    print(f"Text files → {TXT_DIR}")
    print(f"TSV files → {TSV_DIR}")

if __name__ == "__main__":
    main()