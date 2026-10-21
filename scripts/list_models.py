import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config           # loads .env
from openai import OpenAI

base = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=base)
print("Models available at", base)
for m in sorted(client.models.list().data, key=lambda m: m.id):
    print(" ", m.id)
