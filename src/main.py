# src/main.py
import os
import pandas as pd
from dotenv import load_dotenv

from extractor import extract_info_with_openai
from icd_mapper import get_icd_codes

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "transcriptions.csv")
OUT_PATH  = os.path.join(os.path.dirname(__file__), "..", "structured_output.csv")

def main():
    # Load input CSV (requires columns: medical_specialty, transcription)
    df = pd.read_csv("data/transcriptions.csv")
    print(df.head()) # quick check to see if it loaded correctly

    required = {"medical_specialty", "transcription"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required column(s): {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        specialty = row["medical_specialty"]
        transcription = row["transcription"]

        extracted = extract_info_with_openai(transcription)
        icd_codes = get_icd_codes(extracted["recommended_treatment"])

        rows.append({
            "medical_specialty": specialty,
            "Age": extracted["Age"],
            "recommended_treatment": extracted["recommended_treatment"],
            "icd_codes": icd_codes
        })

    df_structured = pd.DataFrame(rows)
    print(df_structured.head(10))
    df_structured.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")

if __name__ == "__main__":
    main()