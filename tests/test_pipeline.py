import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

from src.agent import understand as U
from src.data.catalog import Catalog
from src.kb.loader import load_chunks
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.pipeline import retrieve
from src.retrieval.routes import pinned_sections


def _fake_chat(payload):
    return lambda system, messages, **k: payload


def test_understand_parses_fenced_json(monkeypatch):
    monkeypatch.setattr(U.llm, "chat", _fake_chat('```json\n{"language":"hi","intent":"return_window","english_query":"return window books","payment_method":null}\n```'))
    u = U.understand("x")
    assert u.intent == "return_window" and u.language == "hi" and u.ok


def test_understand_survives_garbage_and_bad_intent(monkeypatch):
    monkeypatch.setattr(U.llm, "chat", _fake_chat("sorry, I cannot"))
    assert U.understand("hello").ok is False
    monkeypatch.setattr(U.llm, "chat", _fake_chat('{"intent":"made_up"}'))
    assert U.understand("hello").intent == "other"


def test_pins_conditional():
    assert "REF-3" in pinned_sections("refund_timeline", "cod")
    assert "RETv3-2" in pinned_sections("return_window", None, "electronics")
    assert "RETv3-2" not in pinned_sections("return_window", None, "fashion")
    assert pinned_sections("unsupported") == []


def test_pipeline_pins_first_and_demotes_superseded():
    chunks = load_chunks()
    r = HybridRetriever(chunks)
    und = U.Understanding(intent="refund_timeline", english_query="refund upi how long")
    got, _ = retrieve(r, Catalog(), "refund upi kab tak", und, max_total=8)
    ids = [g.chunk.section_id for g in got]
    assert ids[:3] == ["REF-2", "KB-00.3", "KB-00.1"] and len(ids) <= 8
    sup = [i for i, g in enumerate(got) if g.chunk.status == "superseded" and g.reason != "pinned"]
    assert all(i >= len(got) - len(sup) for i in sup)
