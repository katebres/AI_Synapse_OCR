"""
Summarize all OCR text analysis results into one row per document.
Input: data/processed/text_analysis.xlsx (page-level)
Output: data/processed/document_summary.xlsx (document-level)
"""

import pandas as pd
import re
from datetime import datetime
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROCESSED = BASE_DIR / "data" / "processed"
INPUT_PATH = PROCESSED / "text_analysis.xlsx"
OUTPUT_PATH = PROCESSED / "document_summary.xlsx"

# --- Load the analyzed data ---
df = pd.read_excel(INPUT_PATH)
print(f"Loaded {len(df)} page entries")

# --- Helper functions ---
def combine_unique(values):
    """Join unique non-null strings into one, separated by commas."""
    vals = [v for v in values if isinstance(v, str) and v.strip()]
    uniq = sorted(set(", ".join(vals).split(", ")))
    return ", ".join([v for v in uniq if v])[:300]

def extract_dates(text):
    """Extract all valid dates and return earliest/latest."""
    date_formats = [
        "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%d.%m.%Y", "%Y.%m.%d", "%d %b %Y"
    ]
    found = []
    for t in str(text).split(","):
        t = t.strip()
        for fmt in date_formats:
            try:
                found.append(datetime.strptime(t, fmt))
                break
            except Exception:
                pass
    if not found:
        return None, None
    return min(found).strftime("%Y-%m-%d"), max(found).strftime("%Y-%m-%d")

def average(values):
    vals = [v for v in values if pd.notna(v)]
    return round(sum(vals)/len(vals), 2) if vals else None

def sum_words(values):
    vals = [v for v in values if pd.notna(v)]
    return int(sum(vals)) if vals else 0

# --- Group pages by file ---
summaries = []
for file_name, group in df.groupby("file"):
    all_dates = combine_unique(group["dates_found"])
    earliest, latest = extract_dates(all_dates)
    currencies = combine_unique(group["currencies"])
    keywords = combine_unique(group["top_terms"])
    sample_amounts = combine_unique(group["amount_samples"])
    mean_conf = average(group["avg_confidence"])
    total_words = sum_words(group["word_count"])

    summaries.append({
        "document": file_name,
        "pages": len(group),
        "total_words": total_words,
        "avg_confidence": mean_conf,
        "currencies": currencies,
        "earliest_date": earliest,
        "latest_date": latest,
        "keywords": keywords,
        "amount_samples": sample_amounts,
        "languages_detected": combine_unique(group["language"]),
    })

# --- Export summary ---
summary_df = pd.DataFrame(summaries)
summary_df = summary_df.sort_values("document").reset_index(drop=True)
summary_df.to_excel(OUTPUT_PATH, index=False)

print(f"✅ Document-level summary saved to: {OUTPUT_PATH}")
print(summary_df.head(10).to_string(index=False))
