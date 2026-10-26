"""Provider-agnostic LLM wrapper (any OpenAI-compatible API: Groq, Cerebras,
OpenRouter, Gemini's OpenAI-compat endpoint, a local Ollama server, ...).
Configured entirely through env vars so switching provider = editing .env.
"""
import os, json, time, hashlib
from openai import OpenAI, APIConnectionError, APITimeoutError
from src import config
from src.agent.ratelimit import RateLimiter

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
_CACHE = config.ROOT / ".cache" / "llm"
_CACHE.mkdir(parents=True, exist_ok=True)

LLM_MAX_RPM = int(os.getenv("LLM_MAX_RPM", "30"))      # requests per minute we allow ourselves (0 = off); Gemini free tier: 12-15
_limiter = RateLimiter(LLM_MAX_RPM, 60.0)

LAST_CACHED = False      # True if the most recent chat() call was served from disk cache
_client = None


def _get():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=BASE_URL)
    return _client


def _retry_after(e):
    headers = getattr(getattr(e, "response", None), "headers", None)
    try:
        return float(headers.get("retry-after")) if headers else None
    except (TypeError, ValueError):
        return None


def chat(system: str, messages: list[dict], temperature: float = 0.2,
         use_cache: bool = True, retries: int = 6) -> str:
    """messages: [{"role": "user"|"model"|"assistant", "text": "..."}]"""
    global LAST_CACHED
    key = hashlib.sha256(json.dumps(
        [BASE_URL, MODEL, system, messages, temperature], sort_keys=True).encode()).hexdigest()
    f = _CACHE / f"{key}.json"
    if use_cache and f.exists():
        LAST_CACHED = True
        return json.loads(f.read_text(encoding="utf-8"))["text"]
    LAST_CACHED = False

    msgs = [{"role": "system", "content": system}] + [
        {"role": "assistant" if m["role"] in ("model", "assistant") else "user",
         "content": m["text"]} for m in messages]

    for attempt in range(retries):
        waited = _limiter.acquire()                  # pace ourselves BEFORE the request (cache hits above never reach this)
        if waited >= 2:
            print(f"   (rate limiter: waited {waited:.0f}s to stay under LLM_MAX_RPM={LLM_MAX_RPM})", flush=True)
        try:
            r = _get().chat.completions.create(model=MODEL, messages=msgs,
                                               temperature=temperature)
            text = r.choices[0].message.content or ""
            if use_cache:
                f.write_text(json.dumps({"text": text}), encoding="utf-8")
            return text
        except Exception as e:
            code = getattr(e, "status_code", None)
            transient = code in (429, 500, 502, 503) or isinstance(
                e, (APIConnectionError, APITimeoutError))
            if transient and attempt < retries - 1:
                wait = _retry_after(e) or min(2 ** attempt * 3, 60)
                if wait > 120:   # daily quota, not a blip: fail fast with a clear message
                    raise RuntimeError(
                        f"Provider asks to wait {wait:.0f}s (likely a DAILY quota). "
                        f"Change LLM_MODEL/LLM_BASE_URL in .env or retry later.") from e
                print(f"   ! LLM error {code}; retry {attempt + 1}/{retries - 1} in {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue
            raise
