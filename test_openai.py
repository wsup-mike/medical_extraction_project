from openai import OpenAI, RateLimitError
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()

try:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello, can you confirm my API setup works?"}]
    )
    print("Model reply:", resp.choices[0].message.content)
except RateLimitError as e:
    print("The request reached the API, but your project has insufficient quota.")
    print("Details:", e)
except Exception as e:
    print("Unexpected error:", e)