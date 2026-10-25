"""Thin wrappers around Gnani's STT and TTS (the ONLY speech services allowed).

Contract (docs.gnani.ai): STT REST takes WAV/MP3/OGG/FLAC/AAC/M4A up to 60 s and an explicit language_code
(en-IN, hi-IN, ...); format "transcribe" converts spoken numbers/dates to digits (en-IN and hi-IN only).
TTS takes text, voice, model and language ("en-IN", "hi-IN", "hi-en", "auto"). One GNANI_API_KEY covers both.
"""
import inspect, logging, os, shutil, sys, tempfile, time, wave
from pathlib import Path

logging.getLogger("gnani").setLevel(os.getenv("GNANI_LOG_LEVEL", "WARNING"))     # the SDK logs 2 INFO lines per request

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a"}
MAX_CLIP_SECONDS = 55          # REST limit is 60 s; stay below it


_TRANSIENT_WORDS = ("429", "503", "timeout", "timed out", "temporarily", "connection aborted", "connection reset", "connection error",
                    "max retries exceeded", "getaddrinfo", "name resolution", "unreachable", "remote end closed", "forcibly closed")


def _transient(e: Exception) -> bool:
    """Rate limits, 5xx, and flaky-network errors are retried; configuration errors (ValueError) never are."""
    if isinstance(e, ValueError):
        return False
    code = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    name = type(e).__name__.lower()
    if isinstance(e, (ConnectionError, TimeoutError)) or "connection" in name or "timeout" in name:
        return True
    return any(k in str(e).lower() for k in _TRANSIENT_WORDS)


def _accepts(fn, name: str) -> bool:
    try:
        ps = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in ps or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps.values())


MAX_PARTS = 12                 # safety net: at most ~11 minutes of audio is ever sent for one file


def split_wav(path: Path, out_dir: Path, seconds: int = MAX_CLIP_SECONDS) -> list[Path]:
    """WAV clips longer than the REST limit are cut into <= 55 s pieces; other formats are passed through.

    The duration comes from the REAL file size, not the header: streaming TTS output often declares a
    huge placeholder length (0xFFFFFFFF), which once made a 1-second clip look like 24 hours.
    """
    if path.suffix.lower() != ".wav":
        return [path]
    try:
        with wave.open(str(path), "rb") as w:
            rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frame = max(ch * width, 1)
            n = min(w.getnframes(), path.stat().st_size // frame)          # header claim vs bytes that actually exist
            if rate <= 0 or n / rate <= seconds:
                return [path]
            parts, step = [], rate * seconds
            for k, start in enumerate(range(0, n, step)):
                if k >= MAX_PARTS:
                    break
                w.setpos(start)
                frames = w.readframes(min(step, n - start))
                if not frames:                                               # nothing left: never send an empty clip
                    break
                p = out_dir / f"{path.stem}_part{k}.wav"
                with wave.open(str(p), "wb") as o:
                    o.setnchannels(ch); o.setsampwidth(width); o.setframerate(rate); o.writeframes(frames)
                parts.append(p)
            return parts or [path]
    except (wave.Error, EOFError, OSError):
        return [path]


class GnaniSTT:
    def __init__(self, client=None, fmt: str | None = None):
        self._client = client
        self.fmt = (fmt or os.getenv("GNANI_STT_FORMAT", "transcribe")).lower()      # measured: digits + dates beat spoken words

    def _c(self):
        if self._client is None:
            from gnani.stt import GnaniSTTClient
            self._client = GnaniSTTClient()
            logging.getLogger("gnani").setLevel(os.getenv("GNANI_LOG_LEVEL", "WARNING"))
        return self._client

    def _one(self, path: Path, language_code: str) -> str:
        kwargs = {"language_code": language_code}
        c = self._c()
        if self.fmt != "verbatim" and _accepts(c.transcribe, "format"):
            kwargs["format"] = self.fmt
        for attempt in range(4):
            try:
                r = c.transcribe(str(path), **kwargs)
                text = r.get("transcript", "") if isinstance(r, dict) else getattr(r, "transcript", "")
                return (text or "").strip()
            except Exception as e:
                if _transient(e) and attempt < 3:
                    time.sleep(2 ** (attempt + 1))                     # 2 s, 4 s, 8 s
                    continue
                raise
        return ""

    def transcribe(self, path, language_code: str = "en-IN") -> str:
        path = Path(path)
        tmp = Path(tempfile.mkdtemp(prefix="stt_"))
        try:
            return " ".join(t for t in (self._one(p, language_code) for p in split_wav(path, tmp)) if t).strip()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


VOICES = {   # timbre-v2.5 catalog: Kaveri (English), Nalini / Deepak (Hindi), Poorvi (Hinglish), Pranav / Shlok / Trupti (English)
    "agent": {"en": "Kaveri", "hi": "Nalini", "hi-en": "Poorvi"},
    "customer": {"en": "Pranav", "hi": "Deepak", "hi-en": "Deepak"},      # the simulated caller must sound different from the agent
}
SAFE_MODEL = "timbre-v2.5"        # accepted by every SDK version; timbre-v3.0 only exists in newer ones (opt-in via GNANI_TTS_MODEL)


def lang_candidates(lang: str) -> list[str]:
    """Language codes to try for a reply language. Hinglish is 'hi-en' on newer SDKs, 'auto' on older ones."""
    if lang == "en":
        return ["en-IN"]
    if lang == "hi":
        return ["hi-IN"]
    pref = os.getenv("GNANI_TTS_LANG_HINGLISH", "").strip()
    return list(dict.fromkeys([c for c in (pref, "hi-en", "auto") if c]))


class GnaniTTS:
    def __init__(self, client=None, model: str | None = None, voice: str | None = None, role: str = "agent"):
        self._client = client
        self.role = role
        self.model = model or os.getenv("GNANI_TTS_MODEL", SAFE_MODEL)
        self.voice = voice or (os.getenv("GNANI_TTS_VOICE") if role == "agent" else None)
        self.last_used = None            # (model, voice, language code) that actually produced the last audio
        self.errors: dict = {}           # (model, voice, code) -> why it was rejected; such options are skipped afterwards

    def _c(self):
        if self._client is None:
            from gnani.tts import GnaniTTSClient
            self._client = GnaniTTSClient()
            logging.getLogger("gnani").setLevel(os.getenv("GNANI_LOG_LEVEL", "WARNING"))
        return self._client

    def _voice_for(self, lang: str) -> str:
        if self.voice:
            return self.voice
        env = {"en": "GNANI_TTS_VOICE_EN", "hi": "GNANI_TTS_VOICE_HI", "hi-en": "GNANI_TTS_VOICE_HINGLISH"}[lang]
        return (os.getenv(env) if self.role == "agent" else None) or VOICES[self.role].get(lang, "Kaveri")

    def _synth(self, text: str, code: str, out_path: Path, model: str, voice: str) -> Path:
        """One request with a concrete language CODE (en-IN, hi-IN, hi-en, auto)."""
        c = self._c()
        for attempt in range(4):
            try:
                audio = c.synthesize(text, voice=voice, model=model, language=code, output_file=str(out_path))
                if not out_path.exists() and isinstance(audio, (bytes, bytearray)):
                    out_path.write_bytes(bytes(audio))
                if not out_path.exists():
                    raise RuntimeError("TTS returned no audio")
                return out_path
            except Exception as e:
                if _transient(e) and attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise

    def synthesize(self, text: str, lang: str, out_path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lang = lang if lang in ("en", "hi", "hi-en") else "en"
        options = list(dict.fromkeys([(self.model, self._voice_for(lang)), (SAFE_MODEL, VOICES[self.role][lang])]))
        last = None
        for model, voice in options:
            for code in lang_candidates(lang):
                key = (model, voice, code)
                if key in self.errors:
                    continue
                try:
                    path = self._synth(text, code, out_path, model, voice)
                    self.last_used = key
                    return path
                except Exception as e:
                    last = e
                    if isinstance(e, ValueError):                      # unsupported model / voice / language: remember it
                        self.errors[key] = f"{type(e).__name__}: {e}"
                    print(f"[tts] {model}/{voice}/{code} failed ({type(e).__name__}: {str(e)[:140]}); trying the next option", file=sys.stderr, flush=True)
        raise last or RuntimeError(f"no working TTS option for {lang}: {self.errors}")
