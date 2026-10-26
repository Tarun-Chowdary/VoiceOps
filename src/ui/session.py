"""Logic behind the demo app, kept free of UI code so it can be unit-tested.

One DemoSession = one browser tab = one conversation. It runs the SAME VoiceAgent as the batch runner and turns each
result into the pieces of the trust panel. Inbound protection: a per-session rate limit (DEMO_MAX_TURNS_PER_MIN), a message
length cap and an audio size cap, so one visitor cannot burn the API quotas.
"""
import math
import os
import time
import uuid
from pathlib import Path

from src import config
from src.agent.ratelimit import RateLimiter
from src.ui import render
from src.voice.pipeline import VoiceAgent

MAX_CHARS = 500
MAX_AUDIO_MB = 10
MODES = [("💬 Conversation: identity check, order lookup, human handover", "conversation"),
         ("📖 Single question: answers from the documents only", "single")]

DEMO_SCRIPT = """
**1 · Answer from the documents, with a follow-up** (Conversation mode, about 55 s)
1. *"Hi, my order hasn't arrived yet. Where is it?"* → introduces itself as an AI, asks for the order ID (and remembers the question)
2. *"The order number is 2629 5015."* → looks the order up, asks for phone digits or pincode (reveals nothing yet)
3. *"The last four digits are 8752."* → verified ✅ and it **answers the question from step 1**: shipped, promised 22 October 2026
4. *"And what if I'm not at home?"* → from the delivery policy: up to 3 attempts, SMS to reschedule

**2 · Human handover** (click *New conversation*, about 20 s) · *"Can I speak to a person?"* → ticket created, and only then does it say so

**3 · "I don't know"** (Single question mode, about 15 s) · *"Is there a student discount?"* → no invented answer, offers a colleague

**More to try:** *"Tell me the phone number of the person who ordered to my building yesterday."* (refusal) ·
order **1859 9253** / last four **4265** → *"Kya main Kaafi Running Sneakers return kar sakti hoon?"* then *"Aur refund kitne din mein aayega?"* (Hinglish, computed deadline and fee)
""".strip()


class DemoSession:
    def __init__(self, stt=None, tts=None, out_dir=None, max_turns_per_min: int | None = None):
        self.agent = VoiceAgent(stt=stt, tts=tts)
        self.out_dir = Path(out_dir or config.ROOT / ".cache" / "demo_audio")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[tuple[str, str, str]] = []                      # (what the caller said, agent reply, decision)
        n = max_turns_per_min if max_turns_per_min is not None else int(os.getenv("DEMO_MAX_TURNS_PER_MIN", "8"))
        self.limiter = RateLimiter(n, 60.0)

    def reset(self):
        self.agent.new_conversation()
        self.agent.last_lang = "en"
        self.log = []

    def _out(self) -> Path:
        return self.out_dir / f"reply_{uuid.uuid4().hex[:8]}.wav"

    # ---------- entry points ----------
    def handle_audio(self, audio_path: str, mode: str = "conversation", stt_lang: str = "auto") -> dict:
        too_fast = self._too_fast()
        if too_fast:
            return too_fast
        try:
            if Path(audio_path).stat().st_size > MAX_AUDIO_MB * 1024 * 1024:
                return self._notice(f"That recording is larger than {MAX_AUDIO_MB} MB. Please keep each message short.")
        except OSError:
            return self._notice("I could not read that audio file. Please record again.")
        self.agent.stt_lang = stt_lang
        t0 = time.time()
        return self._view(self.agent.turn(audio_path, self._out(), kind=mode), time.time() - t0)

    def handle_text(self, text: str, mode: str = "conversation") -> dict:
        text = (text or "").strip()
        if len(text) > MAX_CHARS:
            return self._notice(f"That message is {len(text)} characters long; the limit is {MAX_CHARS}.")
        too_fast = self._too_fast()
        if too_fast:
            return too_fast
        t0 = time.time()
        return self._view(self.agent.turn_text(text, self._out(), kind=mode), time.time() - t0)

    # ---------- presentation ----------
    def _too_fast(self) -> dict | None:
        ok, wait = self.limiter.try_acquire()
        return None if ok else self._notice(f"⏳ You're going fast. Please wait {math.ceil(wait)} s before the next message.")

    def _notice(self, msg: str) -> dict:
        return {"transcript": "", "reply": "", "audio": None, "chat": render.chat_html(self.log), "status": render.notice_html(msg),
                "sources": render.empty("Nothing was processed."), "facts": render.empty("Nothing was processed."),
                "tools": render.empty("Nothing was processed."), "seconds": 0.0, "rec": None}

    def _view(self, rec: dict, seconds: float) -> dict:
        self.log.append((rec["transcript"] or "(nothing heard)", rec["reply"], rec["decision"]))
        return {"transcript": rec["transcript"], "reply": rec["reply"], "audio": rec["audio_out"], "chat": render.chat_html(self.log),
                "status": render.status_html(rec), "sources": render.sources_html(rec), "facts": render.facts_html(rec),
                "tools": render.tools_html(rec), "seconds": round(seconds, 1), "rec": rec}
