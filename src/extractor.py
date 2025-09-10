# src/extractor.py
import json
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

# Tool schema: the fields we want out
FUNCTION_DEFINITION = [{
    "type": "function",
    "function": {
        "name": "extract_medical_data",
        "description": "Extract Age and recommended_treatment from a medical transcription.",
        "parameters": {
            "type": "object",
            "properties": {
                "Age": {
                    "type": "string",
                    "description": "Patient age as a string (e.g., '45'). Use 'Unknown' if not mentioned."
                },
                "recommended_treatment": {
                    "type": "string",
                    "description": "The recommended treatment/procedure. Use 'Unknown' if not mentioned."
                }
            },
            "required": ["Age", "recommended_treatment"]
        }
    }
}]

SYS_MSG = (
    "You are a careful healthcare data extractor. "
    "Return both fields exactly as specified. If a field is missing, set it to 'Unknown'. "
    "Never invent data."
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def extract_info_with_openai(transcription: str) -> Dict[str, str]:
    """Return {'Age': str, 'recommended_treatment': str} using OpenAI function calling."""
    messages = [
        {"role": "system", "content": SYS_MSG},
        {"role": "user", "content": (
            "Extract the patient's Age and recommended_treatment from this transcription.\n\n"
            f"```{transcription}```"
        )}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=FUNCTION_DEFINITION,
        tool_choice="auto",
        temperature=0  # deterministic-ish
    )
    msg = resp.choices[0].message

    # If the model decided to call our tool, parse the args (JSON string)
    if getattr(msg, "tool_calls", None):
        try:
            args = json.loads(msg.tool_calls[0].function.arguments)
        except Exception:
            args = {}
    else:
        # Fallback: try to read plain text (rare if tool schema is provided)
        args = {}

    age = (str(args.get("Age", "Unknown")) or "Unknown").strip()
    tx  = (str(args.get("recommended_treatment", "Unknown")) or "Unknown").strip()
    return {"Age": age, "recommended_treatment": tx}