import os, sys, json
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

from src.kb.loader import load_chunks
from src.policy.calc import Calendar, build_facts, parse_return_windows, parse_festive, parse_cutoff
from src.agent.understand import Understanding
from src.data.catalog import Catalog

CAL = Calendar.load()
BY = {c.section_id: c for c in load_chunks()}


def test_business_days_match_dataset_examples():
    # Mon-Sat only, skipping the 19 Oct warehouse closure (KB-00.3)
    assert CAL.add_business_days(date(2026, 10, 17), 7) == date(2026, 10, 27)
    assert CAL.add_business_days(date(2026, 10, 12), 3) == date(2026, 10, 15)
    assert CAL.add_business_days(date(2026, 10, 14), 5) == date(2026, 10, 21)


def test_tables_are_parsed_from_the_kb_not_hardcoded():
    w = parse_return_windows(BY["RET-2"].text)
    assert w["electronics"][1] == 7 and w["fashion"][1] == 15 and w["grocery"][1] is None
    assert parse_cutoff(BY["RET-3"].text) == date(2026, 8, 1)
    assert parse_festive(BY["RET-8"].text) == (date(2026, 10, 1), date(2026, 10, 15), 30)


def test_refund_fact_gives_exact_date():
    und = Understanding(intent="refund_timeline", payment_method="credit_card",
                        events=[{"event": "refund_initiated", "date": "2026-10-17"}])
    facts = " ".join(build_facts(und, None, BY, CAL))
    assert "27 October 2026" in facts


def test_electronics_before_cutoff_keeps_10_days():
    prod = Catalog().resolve("sku ELE-0019")
    und = Understanding(intent="return_window", events=[{"event": "delivered", "date": "2026-07-20"}])
    facts = " ".join(build_facts(und, Catalog().resolve("Orbix Fit 3 Pro Smartwatch"), BY, CAL))
    assert "10 calendar days" in facts and "closed" in facts


def test_festive_sale_fashion_gets_30_days():
    und = Understanding(intent="return_window", events=[{"event": "order_placed", "date": "2026-10-05"}])
    facts = " ".join(build_facts(und, Catalog().resolve("Rangoli Cotton Straight Kurta"), BY, CAL))
    assert "30 calendar days" in facts


def _dispatch_fake(understand_json: dict, answer_json: dict):
    """One fake llm.chat that answers the 'understand' prompt and the 'generate' prompt differently."""
    def fake(system, messages, **k):
        return json.dumps(understand_json if "language-understanding module" in system else answer_json)
    return fake


def test_rag_filters_citations_to_retrieved_sections(monkeypatch):
    from src.agent import rag
    monkeypatch.setattr(rag.llm, "chat", _dispatch_fake(
        {"language": "en", "intent": "refund_timeline", "english_query": "refund upi"},
        {"decision": "answer", "reply": "ok", "used_sections": ["REF-2", "NOT-REAL-9"]}))
    out = rag.answer("refund upi kab tak")
    assert out["cited"] == ["REF-2"] and "NOT-REAL-9" not in out["used_sections"]
    assert out["intent"] == "refund_timeline"


def test_used_sections_include_what_the_code_consulted(monkeypatch):
    from src.agent import rag
    monkeypatch.setattr(rag.llm, "chat", _dispatch_fake(
        {"language": "en", "intent": "refund_timeline", "english_query": "refund credit card",
         "payment_method": "credit_card", "dated_events": [{"event": "refund_initiated", "date": "2026-10-17"}]},
        {"decision": "answer", "reply": "27 October 2026", "used_sections": ["REF-2"]}))
    out = rag.answer("credit card refund started 17 October 2026")
    assert out["cited"] == ["REF-2"]
    assert "KB-00.3" in out["used_sections"]           # the business-day rule the calculator applied
    assert any("27 October 2026" in f for f in out["facts"])


def test_silence_rule_and_clarify_rule_are_in_the_prompt():
    from src.agent import rag
    assert "silence in the documents is not a policy" in rag.SYSTEM
    assert "general procedure" in rag.SYSTEM
