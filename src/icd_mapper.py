# src/icd_mapper.py
import json
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def normalize_icd_codes(raw: str) -> List[str]:
    """
    Clean and normalize the model's ICD output into a Python list of strings.
    Handles cases with Markdown fences, 'json' tags, or raw text.
    """
    if not raw:
        return []

    # Remove code fences like ```json ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip("json").strip()

    # Try JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(code).strip() for code in parsed]
        if isinstance(parsed, str):
            return [parsed.strip()]
    except Exception:
        pass

    # Fallback: split by comma or whitespace
    return [c.strip() for c in cleaned.replace("\n", ",").split(",") if c.strip()]

def get_icd_codes(treatment: str) -> List[str]:
    """Ask the model for ICD-10 codes given a treatment/procedure string."""
    if not treatment or treatment == "Unknown":
        return []

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Provide the ICD-10 codes for the treatment: '{treatment}'. "
                "Return only a JSON list of codes, no explanation."
            )
        }],
        temperature=0
    )

    raw = response.choices[0].message.content
    return normalize_icd_codes(raw)