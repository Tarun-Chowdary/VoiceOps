"""One voice turn: audio -> Gnani STT -> (normalise, understand, retrieve, answer / converse) -> Gnani TTS -> audio.

Failure policy: a turn ALWAYS ends with something spoken. Unclear or empty audio gets a polite "please repeat"
(and does NOT count as a failed identity check); an internal error gets a spoken apology instead of silence.
"""
import re, time
from pathlib import Path

from src.agent.lang import detect_language, tpl
from src.voice.gnani_io import GnaniSTT, GnaniTTS
from src.voice.langid import choose, plausibility
from src.voice.speech_text import to_speech

FILLERS = {"uh", "um", "umm", "hmm", "er", "ah", "oh", "hm"}
FAST_PATH = 0.75     # measured on real Gnani output: English 0.94, Hinglish 0.80 (en-IN is right) vs Hindi speech 0.47 (needs hi-IN)


def is_unclear(text: str) -> bool:
    toks = [t for t in re.findall(r"[\w\u0900-\u097F]+", (text or "").lower())]
    return not toks or all(t in FILLERS for t in toks)


def chunks_for(ids):
    from src.agent.rag import _ctx
    by = _ctx()["by_id"]
    return [{"section_id": i, "doc_id": by[i].doc_id, "text": by[i].text} for i in ids if i in by]


class VoiceAgent:
    def __init__(self, stt=None, tts=None, stt_lang: str = "auto"):
        self.stt = stt or GnaniSTT()
        self.tts = tts or GnaniTTS()
        self.stt_lang = stt_lang                 # "auto" | "en-IN" | "hi-IN"
        self.conv = None
        self.last_lang = "en"

    def new_conversation(self):
        self.conv = None

    def listen(self, audio_path) -> tuple[str, str]:
        if self.stt_lang != "auto":
            return self.stt_lang, self.stt.transcribe(audio_path, self.stt_lang)
        t_en = self.stt.transcribe(audio_path, "en-IN")
        if t_en.strip() and plausibility(t_en) >= FAST_PATH:        # clearly English / Hinglish: one STT request is enough
            return "en-IN", t_en
        try:
            t_hi = self.stt.transcribe(audio_path, "hi-IN")        # empty, implausible or Hindi speech: ask hi-IN too and compare
        except Exception:
            t_hi = ""
        return choose(t_en, t_hi)

    def _agent(self, text: str, kind: str) -> dict:
        if kind == "conversation":
            if self.conv is None:
                from src.agent.conversation import Conversation
                self.conv = Conversation()
            return self.conv.respond(text)
        from src.agent import rag
        return rag.answer(text)

    def _speak(self, reply: str, lang: str, out_audio) -> tuple[str | None, str]:
        try:
            self.tts.synthesize(to_speech(reply, lang), lang, out_audio)
            return str(out_audio), ""
        except Exception as e:
            return None, f"TTS failed: {type(e).__name__}: {e}"

    def turn(self, audio_path, out_audio, kind: str = "single") -> dict:
        rec = {"audio_in": str(audio_path), "transcript": "", "stt_language": "", "reply": "", "decision": "",
               "intent": "", "language": self.last_lang, "used_sections": [], "retrieved_sections": [],
               "used_chunks": [], "tool_calls": [], "ticket_id": None, "audio_out": None, "error": ""}
        t0 = time.time()
        try:
            code, text = self.listen(audio_path)
            rec.update(transcript=text, stt_language=code, stt_seconds=round(time.time() - t0, 2))
        except Exception as e:
            code, text = "", ""
            rec["error"] = f"STT failed: {type(e).__name__}: {e}"
        t1 = time.time()
        if is_unclear(text):
            lang = self.last_lang
            rec.update(reply=tpl(lang, "didnt_catch" if not rec["error"] else "error"), decision="clarify", intent="unclear_audio", language=lang)
        else:
            try:
                out = self._agent(text, kind)
                lang = out.get("language") or detect_language(out["reply"], self.last_lang)
                lang = lang if lang in ("en", "hi", "hi-en") else detect_language(out["reply"], self.last_lang)
                self.last_lang = lang
                rec.update(reply=out["reply"], decision=out.get("decision", ""), intent=out.get("intent", ""), language=lang,
                           used_sections=out.get("used_sections", []), retrieved_sections=out.get("retrieved", []),
                           used_chunks=chunks_for(out.get("used_sections", [])), tool_calls=out.get("tool_calls", []),
                           ticket_id=out.get("ticket_id"))
            except Exception as e:
                lang = self.last_lang
                rec.update(reply=tpl(lang, "error"), decision="clarify", intent="error", language=lang,
                           error=(rec["error"] + " | " if rec["error"] else "") + f"agent failed: {type(e).__name__}: {e}")
        rec["agent_seconds"] = round(time.time() - t1, 2)
        t2 = time.time()
        rec["audio_out"], err = self._speak(rec["reply"], rec["language"], out_audio)
        if err:
            rec["error"] = (rec["error"] + " | " if rec["error"] else "") + err
        rec["tts_seconds"] = round(time.time() - t2, 2)
        return rec
