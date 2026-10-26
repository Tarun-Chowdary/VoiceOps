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
    assert v3["chat"].count("vo-bubble vo-user") == 3 and "vo-bubble vo-agent" in v3["chat"]


def test_handover_shows_the_ticket_in_the_trust_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    s = session(tmp_path)
    for m in ["Hi, I need some help with an order.", "The order number is 3711 3676.", "The last four digits are 1581."]:
        s.handle_text(m)
    v = s.handle_text("Actually, can I just speak to a person?")
    assert "ESCALATE" in v["status"] and "🎫 TKT" in v["status"] and "create_ticket()" in v["tools"]


def test_single_question_mode_shows_the_document_chunks_used(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake(generate_reply="Only one coupon per order."))
    v = session(tmp_path).handle_text("can i use two coupon codes on one order", mode="single")
    assert v["reply"] == "Only one coupon per order." and "vo-pill" in v["sources"] and "vo-src-body" in v["sources"] and "##" not in v["sources"]


def test_spoken_audio_goes_through_stt_and_reset_starts_over(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    wav = silent_wav(tmp_path / "a.wav")
    s = session(tmp_path, {"a.wav": {"en-IN": "Hello, mujhe order ke baare mein help chahiye."}})
    v = s.handle_audio(str(wav), "conversation", "auto")
    assert v["transcript"].startswith("Hello") and "heard via en-IN" in v["status"]
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


# ---------------- rendering is safe and tidy ----------------
def test_untrusted_text_is_escaped_in_the_chat():
    from src.ui import render
    html = render.chat_html([("<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "answer")])
    assert "<script>" not in html and "<img" not in html and "&lt;script&gt;" in html


def test_source_cards_show_plain_text_not_a_markdown_heading():
    from src.ui import render
    rec = {"used_chunks": [{"section_id": "STY-1", "doc_id": "AGT-STY", "text": "## [STY-1] Opening\nIntroduce yourself."}],
           "retrieved_sections": ["STY-1", "RET-2"], "used_sections": ["STY-1"]}
    html = render.sources_html(rec)
    assert "[STY-1] Opening Introduce yourself." in html and "##" not in html and "RET-2" in html


# ---------------- inbound limits on the demo ----------------
def test_a_visitor_cannot_flood_the_demo(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    s = DemoSession(stt=FakeSTT({}), tts=FakeTTS(), out_dir=tmp_path, max_turns_per_min=2)
    assert s.handle_text("hello there")["rec"] is not None and s.handle_text("hello again")["rec"] is not None
    blocked = s.handle_text("a third message")
    assert blocked["rec"] is None and "going fast" in blocked["status"] and blocked["audio"] is None


def test_oversized_text_and_audio_are_refused_without_spending_quota(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "chat", make_fake())
    s = DemoSession(stt=FakeSTT({}), tts=FakeTTS(), out_dir=tmp_path, max_turns_per_min=100)
    assert "limit is 500" in s.handle_text("x" * 501)["status"]
    big = tmp_path / "big.wav"
    big.write_bytes(b"\0" * (11 * 1024 * 1024))
    assert "larger than 10 MB" in s.handle_audio(str(big))["status"]
    assert s.limiter.try_acquire()[0]                      # the refused requests did not use up the visitor's allowance


def test_app_passes_css_and_theme_wherever_gradio_accepts_them():
    gr = pytest.importorskip("gradio")
    import app
    kw = app.style_kwargs(gr.Blocks.launch) or app.style_kwargs(gr.Blocks.__init__)
    assert "css" in kw and ".vo-hero" in kw["css"]
