import os, json, time, hashlib
from pathlib import Path
from google import genai
from google.genai import types
from src import config

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
_CACHE = config.ROOT / ".cache" / "llm"
_CACHE.mkdir(parents=True, exist_ok=True)

def chat(system: str, messages: list[dict], temperature: float = 0.2,
         use_cache: bool = True, retries: int = 5) -> str:
    """messages: [{"role": "user"|"model", "text": "..."}]"""
    key = hashlib.sha256(json.dumps(
        [MODEL, system, messages, temperature], sort_keys=True).encode()).hexdigest()
    f = _CACHE / f"{key}.json"
    if use_cache and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["text"]

    contents = [types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
                for m in messages]
    cfg = types.GenerateContentConfig(system_instruction=system,
                                      temperature=temperature)
    for attempt in range(retries):
        try:
            r = _client.models.generate_content(model=MODEL, contents=contents, config=cfg)
            text = r.text or ""
            if use_cache:
                f.write_text(json.dumps({"text": text}), encoding="utf-8")
            return text
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (429, 500, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 2)   # 2s, 4s, 8s, 16s
                continue
            raise