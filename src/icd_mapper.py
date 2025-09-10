# src/icd_mapper.py
import json
from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

client = OpenAI()

PROMPT = (
    "Return a JSON array of likely ICD-10 codes that correspond to the given "
    "treatment/procedure. Do not include explanations or text—only a JSON array "
    "of strings. If uncertain, return an empty JSON array.\n\n"
    "Treatment: {treatment}"
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def get_icd_codes(treatment: str) -> List[str]:
    if not treatment or treatment == "Unknown":
        return ["Unknown"]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT.format(treatment=treatment)}],
        temperature=0
    )
    raw = resp.choices[0].message.content

    # Try to parse a JSON array; fallback to raw text
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(c).strip() for c in data]
    except Exception:
        pass
    return [raw.strip() if raw else "Unknown"]