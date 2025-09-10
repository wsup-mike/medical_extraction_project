# src/main.py
import os
import re
import json
import pandas as pd
from typing import Any, Dict, List

from dotenv import load_dotenv
from extractor import extract_info_with_openai
from icd_mapper import get_icd_codes

load_dotenv()  # loads OPENAI_API_KEY from .env

ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_PATH = os.path.join(ROOT, "data", "transcriptions.csv")
OUT_PATH  = os.path.join(ROOT, "structured_output.csv")

# Optional: limit rows for cheap test runs (export ROW_LIMIT=10)
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))

def normalize_icd_codes(raw: Any) -> List[str]:
    """
    Accepts list or string (possibly with ```json fences). Returns a clean list of codes.
    """
    if raw is None:
        return ["Unknown"]
    if isinstance(raw, list):
        return [str(c).strip() for c in raw if str(c).strip()]

    s = str(raw).strip()
    if not s:
        return ["Unknown"]

    # Strip Markdown fences like ```json ... ```
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE)

    # Try JSON parse (expecting a list)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(c).strip() for c in parsed if str(c).strip()]
    except Exception:
        pass

    # Fallback: split on commas/whitespace
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    return parts if parts else ["Unknown"]

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Could not find input CSV at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"medical_specialty", "transcription"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required column(s): {sorted(missing)}")

    if ROW_LIMIT > 0:
        df = df.head(ROW_LIMIT)

    print("Sample input:")
    print(df.head())

    rows: List[Dict[str, Any]] = []
    icd_cache: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        specialty = str(row["medical_specialty"])
        transcription = str(row["transcription"])

        # Extract fields
        try:
            extracted = extract_info_with_openai(transcription)  # {"Age": "...", "recommended_treatment": "..."}
        except Exception as e:
            extracted = {"Age": "Unknown", "recommended_treatment": f"ERROR: {e}"}

        treatment = extracted.get("recommended_treatment", "Unknown")

        # Cache ICD lookups by treatment string
        if treatment not in icd_cache:
            try:
                raw_codes = get_icd_codes(treatment)  # may be list OR a string with ```json
                icd_cache[treatment] = normalize_icd_codes(raw_codes)
            except Exception as e:
                icd_cache[treatment] = [f"ERROR: {e}"]

        codes = icd_cache[treatment]

        rows.append({
            "medical_specialty": specialty,
            "Age": extracted.get("Age", "Unknown"),
            "recommended_treatment": treatment,
            "icd_codes": ", ".join(codes),  # write as CSV-friendly string
        })

    df_structured = pd.DataFrame(rows)
    print("\nStructured sample:")
    print(df_structured.head(10))

    df_structured.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Rows processed: {len(df_structured)}")

if __name__ == "__main__":
    main()