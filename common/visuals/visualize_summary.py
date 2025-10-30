"""
visualize_summary.py
Generates charts from document_summary.xlsx for reporting and analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import pathlib
from collections import Counter

# --- Paths ---
BASE_DIR = pathlib.Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "document_summary.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
df = pd.read_excel(INPUT_FILE)
print(f"Loaded {len(df)} documents from {INPUT_FILE}")

# --- 1. Bar Chart: Average Confidence ---
plt.figure(figsize=(8, 5))
plt.bar(df["document"], df["avg_confidence"], color="skyblue")
plt.title("Average OCR Confidence per Document")
plt.xlabel("Document")
plt.ylabel("Average Confidence (%)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_confidence_per_doc.png")
plt.close()

# --- 2. Bar Chart: Total Words ---
plt.figure(figsize=(8, 5))
plt.bar(df["document"], df["total_words"], color="lightgreen")
plt.title("Total Word Count per Document")
plt.xlabel("Document")
plt.ylabel("Total Words")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "total_words_per_doc.png")
plt.close()

# --- 3. Pie Chart: Currency Distribution (robust for single counts) ---
all_currencies = []

for val in df["currencies"].dropna():
    clean = str(val).replace(";", ",").replace("/", ",").replace("\n", ",")
    for c in clean.split(","):
        code = c.strip().upper()
        # remove non-letters, extra spaces
        code = ''.join(ch for ch in code if ch.isalpha())
        if len(code) == 3:
            all_currencies.append(code)

if not all_currencies:
    print("⚠️ No currency data found.")
else:
    from collections import Counter
    import pandas as pd
    currency_counts = Counter(all_currencies)

    # Create DataFrame for reference
    df_currency = pd.DataFrame.from_dict(currency_counts, orient="index", columns=["Count"])
    df_currency["Percentage"] = (df_currency["Count"] / df_currency["Count"].sum() * 100).round(1)
    df_currency = df_currency.sort_values("Count", ascending=False)

    # Save table
    df_currency.to_csv(OUTPUT_DIR / "currency_distribution_summary.csv")

    # Create pie chart
    labels = [f"{idx} ({row.Count})" for idx, row in df_currency.iterrows()]
    plt.figure(figsize=(6, 6))
    plt.pie(df_currency["Count"], labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Detected Currency Distribution (Count & %)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "currency_distribution.png")
    plt.close()
    print("Chart and summary CSV updated.")
