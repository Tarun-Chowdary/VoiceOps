import os, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

import pytest
from fakes import make_fake
from src.agent import llm
from src.ui.session import DemoSession, MODES


def silent_wav(path: Path, seconds=0.2, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class FakeSTT:
    def __init__(self, table):
        self.table = table

    def transcribe(self, path, language_code="en-IN"):
        return self.table.get(Path(path).name, {}).get(language_code, "")


class FakeTTS:
    def synthesize(self, text, lang, out_path):
        return silent_wav(Path(out_path))


@pytest.fixture(autouse=True)
def _cleanup_tickets():
    yield
    from src import config
    f = config.DATASET_DIR / "tools" / "tickets_created.csv"
    if f.exists():
        f.unlink()


def session(tmp_path, table=None):
    return DemoSession(stt=FakeSTT(table or {}), tts=FakeTTS(), out_dir=tmp_path)


def test_a_typed_conversation_shows_identity_tools_and_a_spoken_reply(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    s = session(tmp_path)
    v1 = s.handle_text("Hello, mujhe order ke baare mein help chahiye.")
    assert v1["audio"] and Path(v1["audio"]).exists() and "not verified" in v1["status"] and "No tools" in v1["tools"]
    v2 = s.handle_text("Order number hai 1859 9253.")
    assert "lookup_order(order_id=18599253)" in v2["tools"] and "not verified" in v2["status"]
    v3 = s.handle_text("Last four digits 4265 hain.")
    assert "verified" in v3["status"] and "not verified" not in v3["status"] and "verification_value=4265" in v3["tools"]
    assert v3["chat"].count("You:") == 3 and "Agent:" in v3["chat"]


def test_handover_shows_the_ticket_in_the_trust_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    s = session(tmp_path)
    for m in ["Hi, I need some help with an order.", "The order number is 3711 3676.", "The last four digits are 1581."]:
        s.handle_text(m)
    v = s.handle_text("Actually, can I just speak to a person?")
    assert "ESCALATE" in v["status"] and "ticket `TKT" in v["status"] and "create_ticket()" in v["tools"]


def test_single_question_mode_shows_the_document_chunks_used(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake(generate_reply="Only one coupon per order."))
    v = session(tmp_path).handle_text("can i use two coupon codes on one order", mode="single")
    assert v["reply"] == "Only one coupon per order." and "**[" in v["sources"] and "> " in v["sources"]


def test_spoken_audio_goes_through_stt_and_reset_starts_over(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    wav = silent_wav(tmp_path / "a.wav")
    s = session(tmp_path, {"a.wav": {"en-IN": "Hello, mujhe order ke baare mein help chahiye."}})
    v = s.handle_audio(str(wav), "conversation", "auto")
    assert v["transcript"].startswith("Hello") and "heard via `en-IN`" in v["status"]
    s.reset()
    assert s.log == [] and s.agent.conv is None


def test_unclear_audio_is_handled_not_crashed(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    wav = silent_wav(tmp_path / "n.wav")
    v = session(tmp_path, {}).handle_audio(str(wav))
    assert v["audio"] and "unclear_audio" in v["status"]


def test_the_gradio_app_builds_with_all_its_endpoints(monkeypatch, tmp_path):
    gr = pytest.importorskip("gradio")
    import app
    demo = app.build(lambda: session(tmp_path))
    fns = demo.fns.values() if isinstance(demo.fns, dict) else demo.fns
    names = {getattr(f, "api_name", None) for f in fns}
    if not names - {None}:
        pytest.skip("this Gradio version does not expose api_name on fns")
    assert {"speak", "upload", "type", "send", "reset"} <= names
    assert [m[1] for m in MODES] == ["conversation", "single"]


def test_warranty_prompts_ask_for_the_claim_path():
    from src.agent import rag, conversation
    assert "how to raise a claim" in rag.SYSTEM and "how to raise a claim" in conversation.CONV_SYSTEM
