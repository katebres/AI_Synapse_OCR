"""
Diagnostic and summary rebuild script.
Reads OCR results from data/processed/text and tsv folders
and regenerates a correct summary.csv with statistics.
"""

import pandas as pd
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
TSV_DIR = BASE_DIR / "data" / "processed" / "tsv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "summary.csv"

items = []

for txt_path in sorted(TEXT_DIR.glob("*.txt")):
    text = txt_path.read_text(encoding="utf-8").strip()
    word_count = len(text.split())
    line_count = text.count("\n") + 1
    preview = text[:180].replace("\n", " ")

    # Try to find matching TSV file
    tsv_path = TSV_DIR / (txt_path.stem + ".tsv")
    conf_mean = None
    if tsv_path.exists():
        try:
            df = pd.read_csv(tsv_path, sep="\t")
            if "conf" in df.columns:
                conf_mean = round(df["conf"].replace(-1, pd.NA).dropna().mean(), 2)
        except Exception:
            conf_mean = None

    items.append({
        "file": txt_path.stem.split("_p")[0],
        "page_or_image": txt_path.stem.split("_p")[-1],
        "txt_path": str(txt_path),
        "tsv_path": str(tsv_path) if tsv_path.exists() else "",
        "word_count": word_count,
        "line_count": line_count,
        "avg_confidence": conf_mean,
        "preview": preview
    })

if not items:
    print("⚠️ No text files found in processed/text — please check OCR output.")
else:
    df = pd.DataFrame(items)
    df.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Rebuilt summary saved to: {SUMMARY_PATH}")
    print(df.head(5).to_string(index=False))
