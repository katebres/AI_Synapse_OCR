import re
import pandas as pd
import pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
TEXT_DIR = BASE_DIR / "data" / "processed" / "text"
OUT_FILE = BASE_DIR / "data" / "processed" / "field_extraction.xlsx"


def find_field(patterns, text):
    """Return first regex match for a list of patterns."""
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            if match.lastindex:
                return match.group(1).strip()
            else:
                return match.group(0).strip()  # return full match
    return None


records = []

for txt_file in sorted(TEXT_DIR.glob("*.txt")):
    text = txt_file.read_text(encoding="utf-8", errors="ignore")

    # --- Trade / Effective Dates ---
    trade_date = find_field([
        r"(?:Trade|Strike|Initial Valuation|Effective)\s*Date[:\s]+([A-Za-z]+\s*\d{1,2},\s*\d{4})",
        r"(?:Trade|Strike|Initial Valuation|Effective)\s*Date[:\s]+([\d]{1,2}[./-][\d]{1,2}[./-][\d]{2,4})"
    ], text)

    # --- Smarter Notional Amount extraction ---
    notional = find_field([
        # look for explicit currency + number pattern
        r"(?:USD|KRW|EUR|JPY|GBP|CNY|AUD|HKD)[\s:]*([₩$€¥]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?)",
        # fallback to standard numeric followed by 'amount' or 'notional'
        r"([₩$€¥]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?=.*?(?:Notional|Amount|USD|KRW))",
        # final fallback: phrase only
        r"(Equity\s*Notional\s*Amount)"
    ], text)

    # if still None, try to find numeric value within 2 lines after "Equity Notional"
    if not notional:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if "Equity Notional" in line:
                next_lines = " ".join(lines[i:i + 3])
                match = re.search(r"([₩$€¥]?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?)", next_lines)
                if match:
                    notional = match.group(1)
                    break

    # --- Rate or Percentage ---
    rate = find_field([
        r"\[?([\d.]+\s?%)\]?",
        r"Rate[:\s]*([\d.]+\s?%)"
    ], text)

    # --- Currency codes and pairs ---
    currency = find_field([
        r"\b(USD|KRW|EUR|JPY|GBP|CNY|AUD|HKD)\b",
        r"\b(KRW/USD|USD/KRW|EUR/USD|USD/EUR)\b"
    ], text)

    # --- Counterparty (with cleanup filter) ---
    counterparty = find_field([
        r"(?:Counterparty|Party\s*A|Seller|Buyer)[:\s]*([A-Za-z&.\s]{3,50})"
    ], text)

    # Cleanup: remove false matches from disclaimers
    if counterparty and any(w in counterparty.lower() for w in ["capacity", "adviser", "principal"]):
        counterparty = None

    # --- Append all extracted fields to the records list ---
    records.append({
        "file": txt_file.stem.split("_p")[0],
        "page": txt_file.stem.split("_p")[-1],
        "Trade_or_Effective_Date": trade_date,
        "Notional": notional,
        "Currency": currency,
        "Rate": rate,
        "Counterparty": counterparty,
    })

# --- Save the extracted data ---
df = pd.DataFrame(records)
df.to_excel(OUT_FILE, index=False)
print(f"Field extraction complete. Results saved to: {OUT_FILE}")
print(df.head(5).to_string(index=False))
