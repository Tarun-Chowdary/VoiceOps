import json, os, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

import pytest
from fakes import make_fake
from src.agent import llm
from src.voice import gnani_io
from src.voice.langid import choose, plausibility
from src.voice.pipeline import VoiceAgent, is_unclear
from src.voice.speech_text import to_speech
import run as batch


def silent_wav(path: Path, seconds: float = 0.2, rate: int = 8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class FakeSTT:
    def __init__(self, table, fail_on=()):
        self.table, self.fail_on, self.calls = table, set(fail_on), []

    def transcribe(self, path, language_code="en-IN"):
        name = Path(path).name
        self.calls.append((name, language_code))
        if name in self.fail_on:
            raise RuntimeError("boom")
        return self.table.get(name, {}).get(language_code, "")


class FakeTTS:
    def __init__(self, fail=False):
        self.fail, self.spoken = fail, []

    def synthesize(self, text, lang, out_path):
        if self.fail:
            raise RuntimeError("tts down")
        self.spoken.append((text, lang))
        return silent_wav(Path(out_path))


@pytest.fixture(autouse=True)
def _cleanup_tickets():
    yield
    from src import config
    f = config.DATASET_DIR / "tools" / "tickets_created.csv"
    if f.exists():
        f.unlink()


# ---------------- speech rewrite ----------------
def test_to_speech_spells_identifiers_and_hides_internal_ids():
    s = to_speech("Order 18599253 refund RRN 315633825001 for SKU ELE-0012 is at QC (RET-7) on 22 October 2026 (Thursday).")
    assert "1859 9253" in s and "3156 3382 5001" in s and "E L E 0 0 1 2" in s and "Q C" in s and "R R N" in s
    assert "RET-7" not in s and "22 October 2026, Thursday" in s


def test_to_speech_leaves_rupee_amounts_and_normal_numbers_to_gnani():
    s = to_speech("The fee is ₹10,000 and the window is 15 days. Pincode 440010.")
    assert "₹10,000" in s and "15 days" in s and "440010" in s


# ---------------- language choice ----------------
def test_choose_defaults_to_en_in_and_uses_hi_in_only_when_needed():
    assert choose("mujhe refund ke baare mein jaanna hai", "मुझे रिफंड के बारे में जानना है")[0] == "en-IN"
    assert choose("", "मुझे रिफंड के बारे में जानना है")[0] == "hi-IN"
    assert choose("xqz blorf zzk wmpt", "मुझे रिफंड के बारे में जानना है")[0] == "hi-IN"
    assert choose("", "") == ("en-IN", "")
    assert plausibility("how long does a refund to upi take") > 0.8


def test_unclear_audio_detection():
    assert is_unclear("") and is_unclear("  uh  um ") and not is_unclear("4265") and not is_unclear("refund")


# ---------------- voice turns ----------------
def test_single_question_turn_produces_text_chunks_and_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake(generate_reply="Only one coupon per order."))
    a = silent_wav(tmp_path / "q1.wav")
    agent = VoiceAgent(FakeSTT({"q1.wav": {"en-IN": "can i use two coupon codes on one order"}}), FakeTTS())
    r = agent.turn(a, tmp_path / "o.wav")
    assert r["reply"] == "Only one coupon per order." and r["audio_out"] and Path(r["audio_out"]).exists()
    assert r["used_sections"] and r["used_chunks"][0]["text"] and not r["error"]


def test_unclear_audio_gets_a_spoken_retry_and_does_not_burn_a_verification_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    a1, a2, a3, a4 = (silent_wav(tmp_path / f"{n}.wav") for n in ("a1", "a2", "a3", "a4"))
    stt = FakeSTT({"a1.wav": {"en-IN": "Hello, mujhe order ke baare mein help chahiye."},
                   "a2.wav": {"en-IN": "Order number hai 1859 9253."},
                   "a3.wav": {"en-IN": " "},                                    # noise / silence
                   "a4.wav": {"en-IN": "Last four digits 4265 hain."}})
    tts = FakeTTS()
    ag = VoiceAgent(stt, tts)
    for k, p in enumerate((a1, a2), 1):
        ag.turn(p, tmp_path / f"o{k}.wav", kind="conversation")
    r3 = ag.turn(a3, tmp_path / "o3.wav", kind="conversation")
    assert r3["intent"] == "unclear_audio" and "dobara" in r3["reply"] and r3["audio_out"]
    r4 = ag.turn(a4, tmp_path / "o4.wav", kind="conversation")
    assert r4["tool_calls"] == ["lookup_order(order_id=18599253, verification_value=4265)"] and ag.conv.s.verified
    assert ag.conv.s.attempts == 0


def test_agent_exception_still_ends_with_speech(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("LLM down")
    monkeypatch.setattr(llm, "chat", boom)
    a = silent_wav(tmp_path / "q.wav")
    r = VoiceAgent(FakeSTT({"q.wav": {"en-IN": "how long do refunds take"}}), FakeTTS()).turn(a, tmp_path / "o.wav")
    assert r["audio_out"] and "went wrong" in r["reply"] and "agent failed" in r["error"]


def test_tts_failure_is_reported_not_hidden(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    a = silent_wav(tmp_path / "q.wav")
    r = VoiceAgent(FakeSTT({"q.wav": {"en-IN": "return window for electronics"}}), FakeTTS(fail=True)).turn(a, tmp_path / "o.wav")
    assert r["audio_out"] is None and "TTS failed" in r["error"] and r["reply"]


# ---------------- batch run ----------------
def test_plan_groups_only_explicit_conversation_files(tmp_path):
    names = ["question_001.wav", "question_002.wav", "PC901_t1.wav", "PC901_t2.wav", "PC901_t3.wav", "other-turn1.wav", "other-turn2.wav"]
    files = [tmp_path / n for n in names]
    jobs = batch.plan(files)
    kinds = {j[1]: (j[0], [t for t, _ in j[2]]) for j in jobs}
    assert kinds["PC901"] == ("conversation", [1, 2, 3]) and kinds["other"] == ("conversation", [1, 2])
    assert kinds["question_001"][0] == "single" and kinds["question_002"][0] == "single"     # not glued into one fake conversation
    forced = batch.plan([tmp_path / "x_1.wav", tmp_path / "x_2.wav"], mode="conversation")
    assert forced[0][0] == "conversation"
    assert all(j[0] == "single" for j in batch.plan([tmp_path / "PC901_t1.wav", tmp_path / "PC901_t2.wav"], mode="single"))


def test_batch_run_writes_text_audio_and_chunks_for_every_file(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake(generate_reply="Only one coupon per order."))
    d, out = tmp_path / "in", tmp_path / "out"
    d.mkdir()
    table = {}
    for n, t in [("question_001.wav", "can i use two coupon codes"), ("PC901_t1.wav", "Hello, mujhe order ke baare mein help chahiye."),
                 ("PC901_t2.wav", "Order number hai 1859 9253."), ("PC901_t3.wav", "Last four digits 4265 hain."),
                 ("bad.wav", "x")]:
        silent_wav(d / n)
        table[n] = {"en-IN": t}
    stt = FakeSTT(table, fail_on={"bad.wav"})
    rc = batch.main(["--audio_dir", str(d), "--out_dir", str(out)], stt=stt, tts=FakeTTS())
    assert rc == 0                                                    # even the file whose STT failed got a spoken reply
    for stem in ("question_001", "PC901_t1", "PC901_t3", "bad"):
        assert (out / f"{stem}.json").exists() and (out / f"{stem}.txt").exists() and (out / f"{stem}.wav").exists()
    rec = json.loads((out / "question_001.json").read_text(encoding="utf-8"))
    assert rec["used_chunks"] and rec["reply"] == "Only one coupon per order."
    t3 = json.loads((out / "PC901_t3.json").read_text(encoding="utf-8"))
    assert t3["tool_calls"] == ["lookup_order(order_id=18599253, verification_value=4265)"] and t3["mode"] == "conversation"
    assert "STT failed" in json.loads((out / "bad.json").read_text(encoding="utf-8"))["error"]
    assert (out / "results.csv").exists() and len((out / "results.jsonl").read_text(encoding="utf-8").splitlines()) == 5


def test_batch_exit_code_flags_missing_audio(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    d = tmp_path / "in"; d.mkdir(); silent_wav(d / "q.wav")
    rc = batch.main(["--audio_dir", str(d), "--out_dir", str(tmp_path / "o")], stt=FakeSTT({"q.wav": {"en-IN": "refund policy"}}), tts=FakeTTS(fail=True))
    assert rc == 2


# ---------------- Gnani wrappers ----------------
def test_long_wav_is_split_below_the_60s_limit(tmp_path):
    p = silent_wav(tmp_path / "long.wav", seconds=130, rate=8000)
    parts = gnani_io.split_wav(p, tmp_path)
    assert len(parts) == 3
    for q in parts:
        with wave.open(str(q), "rb") as w:
            assert w.getnframes() / w.getframerate() <= 55.01


def test_stt_passes_format_only_when_the_sdk_accepts_it(tmp_path):
    p = silent_wav(tmp_path / "a.wav")

    class WithFormat:
        def transcribe(self, path, language_code, format="verbatim"):
            self.seen = (language_code, format)
            return {"transcript": "hello"}

    class WithoutFormat:
        def transcribe(self, path, language_code):
            return {"transcript": "hello"}

    c1 = WithFormat()
    assert gnani_io.GnaniSTT(c1, fmt="transcribe").transcribe(p, "en-IN") == "hello" and c1.seen == ("en-IN", "transcribe")
    assert gnani_io.GnaniSTT(WithoutFormat(), fmt="transcribe").transcribe(p, "en-IN") == "hello"      # no crash on older SDKs


def test_wer_ignores_case_punctuation_and_spoken_digit_formatting():
    from src.eval.wer import wer
    assert wer("My refund started on 17 October 2026.", "my refund started on 17 october two zero two six") == 0.0
    assert wer("one two three four", "one two three five") == 1.0                # a spoken digit run is ONE token: one wrong digit = wrong ID
    assert 0 < wer("refund to upi", "refund to ups") < 1 and wer("", "") == 0.0


def test_streaming_wav_header_cannot_cause_a_flood_of_stt_requests(tmp_path):        # the bug seen in voice_smoke.py: 1,627 requests
    import struct
    p = silent_wav(tmp_path / "stream.wav", seconds=1, rate=24000)
    b = bytearray(p.read_bytes())
    b[4:8] = struct.pack("<I", 0xFFFFFFFF)          # RIFF length unknown
    b[40:44] = struct.pack("<I", 0xFFFFFFFF)        # data length unknown: header now claims ~24 hours
    p.write_bytes(bytes(b))
    assert gnani_io.split_wav(p, tmp_path) == [p]


def test_even_a_truly_long_recording_is_capped(tmp_path):
    p = silent_wav(tmp_path / "huge.wav", seconds=1200, rate=8000)               # 20 minutes
    assert len(gnani_io.split_wav(p, tmp_path)) <= gnani_io.MAX_PARTS


# ---------------- TTS: behaviour measured on a real SDK (timbre-v3.0 and 'hi-en' are rejected by older versions) ----------------
class StrictClient:
    """Mimics the installed SDK: validates model and language locally and raises ValueError, like the real one."""
    models, langs = {"timbre-v2.0", "timbre-v2.5"}, {"auto", "en-IN", "hi-IN", "bn-IN", "gu-IN", "kn-IN", "ml-IN", "mr-IN", "pa-IN", "ta-IN", "te-IN"}

    def __init__(self):
        self.calls = []

    def synthesize(self, text, voice, model, language, output_file=None):
        if model not in self.models:
            raise ValueError(f"Unsupported model '{model}'. Choose from: timbre-v2.0, timbre-v2.5")
        if language not in self.langs:
            raise ValueError(f"Unsupported language '{language}'. Choose from: auto, bn-IN, en-IN")
        self.calls.append((model, voice, language))
        Path(output_file).write_bytes(b"RIFF")


def test_hinglish_reply_still_gets_audio_on_an_sdk_without_hi_en(tmp_path):               # the silent-reply bug found by voice_smoke.py
    c = StrictClient()
    t = gnani_io.GnaniTTS(c)
    out = t.synthesize("Aapka refund aa jayega.", "hi-en", tmp_path / "o.wav")
    assert out.exists() and t.last_used == ("timbre-v2.5", "Poorvi", "auto") and c.calls == [("timbre-v2.5", "Poorvi", "auto")]


def test_unsupported_options_are_remembered_so_they_are_not_retried(tmp_path, capsys):
    t = gnani_io.GnaniTTS(StrictClient(), model="timbre-v3.0", voice="Shivani")
    t.synthesize("hello", "en", tmp_path / "a.wav")
    t.synthesize("hello again", "en", tmp_path / "b.wav")
    assert t.last_used == ("timbre-v2.5", "Kaveri", "en-IN") and ("timbre-v3.0", "Shivani", "en-IN") in t.errors
    assert capsys.readouterr().err.count("timbre-v3.0/Shivani/en-IN failed") == 1                  # warned once, not on every reply


def test_customer_and_agent_voices_differ_in_every_language():
    for lang in ("en", "hi", "hi-en"):
        assert gnani_io.VOICES["agent"][lang] != gnani_io.VOICES["customer"][lang]


def test_every_tts_option_failing_raises_instead_of_hiding_it(tmp_path):
    class Dead:
        def synthesize(self, *a, **k):
            raise ValueError("Unsupported everything")
    with pytest.raises(ValueError):
        gnani_io.GnaniTTS(Dead()).synthesize("hi", "en", tmp_path / "x.wav")


def test_network_errors_are_retried_but_configuration_errors_are_not():
    assert gnani_io._transient(ConnectionError("Connection aborted"))
    assert gnani_io._transient(RuntimeError("HTTPSConnectionPool: Max retries exceeded ... getaddrinfo failed"))
    assert gnani_io._transient(RuntimeError("429 Too Many Requests"))
    assert not gnani_io._transient(ValueError("Unsupported model 'timbre-v3.0'"))


def test_stt_survives_a_dropped_connection_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(gnani_io.time, "sleep", lambda s: None)
    p = silent_wav(tmp_path / "a.wav")

    class Flaky:
        n = 0

        def transcribe(self, path, language_code, format="verbatim"):
            Flaky.n += 1
            if Flaky.n < 3:
                raise ConnectionError("('Connection aborted.', ConnectionResetError(10054))")
            return {"transcript": "hello"}

    assert gnani_io.GnaniSTT(Flaky(), fmt="transcribe").transcribe(p, "en-IN") == "hello" and Flaky.n == 3


# ---------------- regression tests built from REAL Gnani transcripts (voice_smoke.py output) ----------------
REAL_EN = {"en-IN": "my order number is 18599253 and my refund to up i started on 17th October 2026",
           "hi-IN": "माइ ऑर्डर नंबर इज़ 18599253 एन्ड माइ रिफंड टू यूपी आई स्टार्टेड ऑन सेवेनतींठ ऑक्टोबर 20 26"}
REAL_HI = {"en-IN": "mera order number hai ek hazaar aath sau unsath nau hazaar do sau tirpan aur mujhe refund kab milega",
           "hi-IN": "मेरा ऑर्डर नंबर है 1,859 9,253 और मुझे रिफंड कब मिलेगा"}


def test_language_choice_on_real_transcripts_with_itn():
    assert choose(REAL_EN["en-IN"], REAL_EN["hi-IN"])[0] == "en-IN"           # English speech stays en-IN
    assert choose(REAL_HI["en-IN"], REAL_HI["hi-IN"])[0] == "hi-IN"           # Hindi speech: en-IN only produced Hindi number WORDS


def test_itn_digits_with_thousands_separators_still_give_the_order_id():
    from src.agent.conversation import extract_order_id
    from src.agent.normalize import normalize_asr
    assert extract_order_id(normalize_asr(REAL_HI["hi-IN"]))[0] == "18599253"                 # "1,859 9,253"
    assert extract_order_id(normalize_asr(REAL_EN["en-IN"]))[0] == "18599253"
    assert "upi" in normalize_asr(REAL_EN["en-IN"]) and "up i" not in normalize_asr(REAL_EN["en-IN"])


def test_a_hindi_caller_can_complete_verification_from_itn_transcripts(monkeypatch):
    from src.agent import conversation as C
    monkeypatch.setattr(llm, "chat", make_fake())
    c = C.Conversation()
    c.respond("नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।")
    t2 = c.respond(REAL_HI["hi-IN"])
    assert t2["tool_calls"] == ["lookup_order(order_id=18599253)"]
    t3 = c.respond("आख़िरी चार अंक 4,265 हैं")
    assert t3["tool_calls"] == ["lookup_order(order_id=18599253, verification_value=4265)"] and t3["verified"]


REAL_HINGLISH = {"en-IN": "mera order number hai 18599253 aur refund kab tak aayega",
                 "hi-IN": "मेरा ऑर्डर नंबर है 18599253 और रिफंड कब तक आएगा?"}


def test_fast_path_thresholds_match_the_real_transcripts():
    assert plausibility(REAL_EN["en-IN"]) >= 0.75 and plausibility(REAL_HINGLISH["en-IN"]) >= 0.75     # 0.94 / 0.80 measured
    assert plausibility(REAL_HI["en-IN"]) < 0.75                                                         # 0.47: Hindi speech needs hi-IN
    assert choose(REAL_HINGLISH["en-IN"], REAL_HINGLISH["hi-IN"])[0] == "en-IN"


def test_english_and_hinglish_use_one_stt_request_but_hindi_uses_two(tmp_path):
    ens, his = silent_wav(tmp_path / "e.wav"), silent_wav(tmp_path / "h.wav")
    stt = FakeSTT({"e.wav": REAL_HINGLISH, "h.wav": REAL_HI})
    ag = VoiceAgent(stt, FakeTTS())
    assert ag.listen(ens)[0] == "en-IN" and [c for c in stt.calls if c[0] == "e.wav"] == [("e.wav", "en-IN")]
    code, text = ag.listen(his)
    assert code == "hi-IN" and "1,859 9,253" in text and [c[1] for c in stt.calls if c[0] == "h.wav"] == ["en-IN", "hi-IN"]
