# src/main.py
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any

from extractor import extract_info_with_openai
from icd_mapper import get_icd_codes

load_dotenv()  # loads OPENAI_API_KEY from .env

ROOT = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_PATH = os.path.join(ROOT, "data", "transcriptions.csv")
OUT_PATH  = os.path.join(ROOT, "structured_output.csv")

# Optional: limit rows for cheap test runs: export ROW_LIMIT=10
ROW_LIMIT = int(os.getenv("ROW_LIMIT", "0"))

def main():
    # Load input CSV (requires columns: medical_specialty, transcription)
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

    rows: list[Dict[str, Any]] = []
    icd_cache: Dict[str, list[str]] = {}   # cache by treatment string

    for idx, row in df.iterrows():
        specialty = str(row["medical_specialty"])
        transcription = str(row["transcription"])

        try:
            extracted = extract_info_with_openai(transcription)  # {"Age": "...", "recommended_treatment": "..."}
        except Exception as e:
            # Keep going; record a stub so you can audit later
            extracted = {"Age": "Unknown", "recommended_treatment": f"ERROR: {e}"}

        treatment = extracted.get("recommended_treatment", "Unknown")

        # Cache ICD lookups to save cost on repeated treatments
        if treatment not in icd_cache:
            try:
                icd_cache[treatment] = get_icd_codes(treatment)
            except Exception as e:
                icd_cache[treatment] = [f"ERROR: {e}"]

        icd_codes = icd_cache[treatment]

        rows.append({
            "medical_specialty": specialty,
            "Age": extracted.get("Age", "Unknown"),
            "recommended_treatment": treatment,
            # write as comma-separated string for CSV
            "icd_codes": ", ".join(icd_codes) if isinstance(icd_codes, list) else str(icd_codes),
        })

    df_structured = pd.DataFrame(rows)
    print("\nStructured sample:")
    print(df_structured.head(10))

    df_structured.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Rows processed: {len(df_structured)}")

if __name__ == "__main__":
    main()