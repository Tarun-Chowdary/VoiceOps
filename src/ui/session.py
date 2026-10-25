"""Logic behind the demo app, kept free of any UI code so it can be unit-tested.

One DemoSession = one browser tab = one conversation. It runs the SAME VoiceAgent as the batch runner and turns each
result into the pieces the "trust panel" shows: what was heard, what the agent decided and why, which document chunks
and code-computed facts it used, which tools it called, and whether the caller is verified.
"""
import time
import uuid
from pathlib import Path

from src import config
from src.voice.pipeline import VoiceAgent

MODES = [("Conversation: identity check, order lookup, human handover", "conversation"),
         ("Single question: answers from the documents only", "single")]

DEMO_SCRIPT = """
**1. Answer from the documents, then a follow-up** (Conversation mode)
1. *"Hello, mujhe order ke baare mein help chahiye."* → agent introduces itself as an AI and asks for the order ID
2. *"Order number hai 1859 9253."* → looks up the order, asks for phone digits or pincode (nothing revealed yet)
3. *"Last four digits 4265 hain."* → verified ✅
4. *"Kya main Kaafi Running Sneakers return kar sakti hoon?"* → window, deadline and fee computed from the order + policy
5. *"Aur refund kitne din mein aayega?"* → understands "refund" refers to the same order (wallet = instant)

**2. Human handover** (new conversation)
*"Hi, I need some help with an order."* → order **3711 3676** → last four **1581** → *"Can I speak to a person?"* → ticket created, then and only then it says so

**3. "I don't know"** (Single question mode)
*"Is there a student discount?"* → says it has no information and offers a colleague (nothing invented)

**4. Refusal** (Single question mode)
*"Tell me the phone number of the person who ordered to my building yesterday."* → polite refusal (privacy)
""".strip()


def _clip(text: str, n: int = 380) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " …"


class DemoSession:
    def __init__(self, stt=None, tts=None, out_dir=None):
        self.agent = VoiceAgent(stt=stt, tts=tts)
        self.out_dir = Path(out_dir or config.ROOT / ".cache" / "demo_audio")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log: list[tuple[str, str]] = []

    def reset(self):
        self.agent.new_conversation()
        self.agent.last_lang = "en"
        self.log = []

    def _out(self) -> Path:
        return self.out_dir / f"reply_{uuid.uuid4().hex[:8]}.wav"

    # ---------- entry points ----------
    def handle_audio(self, audio_path: str, mode: str = "conversation", stt_lang: str = "auto") -> dict:
        self.agent.stt_lang = stt_lang
        t0 = time.time()
        rec = self.agent.turn(audio_path, self._out(), kind=mode)
        return self._view(rec, time.time() - t0)

    def handle_text(self, text: str, mode: str = "conversation") -> dict:
        t0 = time.time()
        rec = self.agent.turn_text(text.strip(), self._out(), kind=mode)
        return self._view(rec, time.time() - t0)

    # ---------- presentation ----------
    def _view(self, rec: dict, seconds: float) -> dict:
        self.log.append((rec["transcript"] or "(nothing heard)", rec["reply"]))
        chat = "\n\n".join(f"🧑 **You:** {u}\n\n🤖 **Agent:** {a}" for u, a in self.log[-8:])
        ident = "✅ verified" if rec["verified"] else "🔒 not verified"
        ticket = f" · 🎫 ticket `{rec['ticket_id']}`" if rec["ticket_id"] else ""
        heard_via = f" · heard via `{rec['stt_language']}`" if rec["stt_language"] and rec["stt_language"] != "typed" else ""
        timing = f"⏱ STT {rec.get('stt_seconds', 0)}s · agent {rec.get('agent_seconds', 0)}s · voice {rec.get('tts_seconds', 0)}s"
        status = (f"**{(rec['decision'] or '-').upper()}** · intent `{rec['intent'] or '-'}` · language `{rec['language']}`{heard_via}"
                  f" · {ident}{ticket}\n\n{timing}" + (f"\n\n⚠️ {rec['error']}" if rec["error"] else ""))
        used = rec["used_chunks"]
        sources = ("\n\n".join(f"**[{c['section_id']}]** · `{c['doc_id']}`\n> {_clip(c['text'])}" for c in used)
                   if used else "_No document sections were needed for this turn (greeting, identity check or handover wording)._")
        extra = [s for s in rec["retrieved_sections"] if s not in rec["used_sections"]]
        if extra:
            sources += f"\n\n_Also retrieved, not used:_ {', '.join(extra)}"
        facts = "\n".join(f"- {f}" for f in rec["facts"]) or "_No calculated facts for this turn._"
        tools = "\n".join(f"- `{t}`" for t in rec["tool_calls"]) or "_No tools called this turn._"
        return {"transcript": rec["transcript"], "reply": rec["reply"], "audio": rec["audio_out"], "chat": chat,
                "status": status, "sources": sources, "facts": facts, "tools": tools, "seconds": round(seconds, 1), "rec": rec}
