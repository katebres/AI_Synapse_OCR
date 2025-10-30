"""
Analyze OCR text outputs: language, amounts, dates, and financial terms.
Input : data/processed/summary.csv
Output: data/processed/text_analysis.xlsx
"""

import re
import pathlib
import pandas as pd
from langdetect import detect, DetectorFactory
from collections import Counter

# Ensure consistent results
DetectorFactory.seed = 42

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SUMMARY_PATH = PROCESSED_DIR / "summary.csv"
OUT_PATH = PROCESSED_DIR / "text_analysis.xlsx"

# --- Load summary file ---
df = pd.read_csv(SUMMARY_PATH)
print(f"Loaded {len(df)} entries from summary.csv")

# --- Keyword sets for quick term detection ---
FIN_KEYWORDS = [
    "swap", "fx", "rate", "coupon", "notional", "barrier", "index",
    "callable", "equity", "maturity", "settlement", "forward", "hedge"
]

CURRENCY_CODES = ["USD", "EUR", "KRW", "JPY", "AUD", "HKD", "GBP", "CNY"]

def extract_info(text):
    """Return dict of extracted features for a single document."""
    info = {}

    # --- Detect language ---
    try:
        info["language"] = detect(text)
    except Exception:
        info["language"] = "unknown"

    # --- Find dates (multiple formats) ---
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b\d{4}[.-]\d{1,2}[.-]\d{1,2}\b",
        r"\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4}\b",
    ]
    dates = []
    for pat in date_patterns:
        dates += re.findall(pat, text)
    info["dates_found"] = ", ".join(sorted(set(dates)))[:150]

    # --- Find monetary values ---
    amounts = re.findall(r"[\$€₩¥]?\s?\d[\d,]*(?:\.\d+)?", text)
    info["amount_samples"] = ", ".join(amounts[:5])

    # --- Find currencies explicitly ---
    found_curr = [c for c in CURRENCY_CODES if c in text]
    info["currencies"] = ", ".join(sorted(set(found_curr)))

    # --- Financial keywords (frequency) ---
    words = re.findall(r"[A-Za-z]+", text.lower())
    found = [w for w in words if w in FIN_KEYWORDS]
    info["top_terms"] = ", ".join([f"{w}({n})" for w, n in Counter(found).most_common(5)]) or "-"

    return info


records = []
for _, row in df.iterrows():
    txt_path = pathlib.Path(row["txt_path"])
    if not txt_path.exists():
        continue
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    features = extract_info(text)
    record = {
        "file": row["file"],
        "page_or_image": row["page_or_image"],
        "language": features["language"],
        "dates_found": features["dates_found"],
        "currencies": features["currencies"],
        "amount_samples": features["amount_samples"],
        "top_terms": features["top_terms"],
        "word_count": row.get("word_count", None),
        "avg_confidence": row.get("avg_confidence", None),
        "preview": row.get("preview", "")[:200],
    }
    records.append(record)

# --- Save results ---
out_df = pd.DataFrame(records)
out_df.to_excel(OUT_PATH, index=False)
print(f"✅ Analysis complete. Excel saved to: {OUT_PATH}")
print(out_df.head(5).to_string(index=False))
