# Medical Extraction Project

Minimal wrapper that:
- reads `data/transcriptions.csv`
- extracts `Age` and `recommended_treatment` via OpenAI function calling
- maps treatments to ICD-10 codes
- writes `structured_output.csv`

## Run
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Add your key to .env (not committed):
# OPENAI_API_KEY=sk-proj-...
python -m src.main