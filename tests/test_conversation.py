import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("EMBED_BACKEND", "hash")

import pytest
from fakes import make_fake
from src.agent import conversation as C
from src.agent import llm


@pytest.fixture(autouse=True)
def _tickets_cleanup():
    yield
    from src import config
    f = config.DATASET_DIR / "tools" / "tickets_created.csv"
    if f.exists():
        f.unlink()


def _conv(monkeypatch, **kw):
    prompts = []
    monkeypatch.setattr(llm, "chat", make_fake(prompts=prompts, **kw))
    return C.Conversation(), prompts


def test_protocol_happy_path_hinglish(monkeypatch):                     # mirrors practice PC003 T1-T3
    c, _ = _conv(monkeypatch)
    t1 = c.respond("Hello, mujhe order ke baare mein help chahiye.")
    assert "AI assistant" in t1["reply"] and t1["decision"] == "clarify" and t1["tool_calls"] == []
    t2 = c.respond("Order number hai 1859 9253.")
    assert t2["tool_calls"] == ["lookup_order(order_id=18599253)"] and "aakhri 4" in t2["reply"]
    t3 = c.respond("Last four digits 4265 hain.")
    assert t3["tool_calls"] == ["lookup_order(order_id=18599253, verification_value=4265)"] and t3["verified"] and t3["decision"] == "answer"


def test_pending_question_answered_after_verification_and_no_sensitive_fields(monkeypatch):   # mirrors PC011
    c, prompts = _conv(monkeypatch, generate_reply="It ships soon.")
    t1 = c.respond("Hi, my order hasn't arrived yet. Where is it?")
    assert t1["decision"] == "clarify" and "order ID" in t1["reply"]
    c.respond("The order number is 2629 5015.")
    t3 = c.respond("The last four digits are 8752.")
    assert t3["reply"] == "It ships soon."                              # the earlier question was answered, not forgotten
    prompt = prompts[-1]
    block = prompt.split("ORDER (caller is verified):", 1)[1].split("VERIFIED FACTS", 1)[0]
    assert "status: shipped" in block and "promised_date: 2026-10-22" in block
    assert "delivery_pincode" not in block and "customer_id" not in block        # sensitive fields are never put in the order block
    assert "641004" not in prompt and "C100079" not in prompt                    # and their VALUES appear nowhere in the prompt (IDV-4)


def test_failed_verification_twice_shares_nothing(monkeypatch):         # mirrors PC021
    c, prompts = _conv(monkeypatch)
    c.respond("नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।")
    c.respond("ऑर्डर नंबर है 7746 1439।")
    t3 = c.respond("आख़िरी चार अंक 1266 हैं।")
    assert t3["decision"] == "clarify" and "दोबारा" in t3["reply"] and not t3["verified"]
    t4 = c.respond("आख़िरी चार अंक 1266 हैं।")
    assert t4["decision"] == "refuse" and "रजिस्टर्ड" in t4["reply"]
    assert t4["tool_calls"] == ["lookup_order(order_id=77461439, verification_value=1266)"]
    assert prompts == []                                                # the LLM never saw anything about this order
    t5 = c.respond("Where is my order? Is it shipped?")
    assert t5["decision"] == "refuse" and t5["tool_calls"] == []        # locked: no more attempts, no data


def test_human_request_creates_one_ticket_and_only_claims_it_when_created(monkeypatch):   # mirrors PC005
    c, _ = _conv(monkeypatch)
    for m in ["नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।", "ऑर्डर नंबर है 3711 3676।", "आख़िरी चार अंक 1581 हैं।"]:
        c.respond(m)
    t4 = c.respond("क्या मैं किसी इंसान से बात कर सकती हूँ?")
    assert t4["decision"] == "escalate" and t4["tool_calls"] == ["create_ticket()"] and "टिकट" in t4["reply"] and t4["ticket_id"]
    again = c.respond("please connect me to a human agent")
    assert again["tool_calls"] == []                                    # no duplicate ticket


def test_ticket_failure_is_never_claimed_as_success(monkeypatch):
    c, _ = _conv(monkeypatch)
    monkeypatch.setattr(C, "create_ticket", lambda *a, **k: {"created": False, "error": "x"})
    t = c.respond("I want to talk to a human agent please")
    assert t["decision"] == "escalate" and "couldn't raise a ticket" in t["reply"] and "I've raised a ticket" not in t["reply"]


def test_repeat_contact_detected_from_order_data(monkeypatch):          # mirrors PC007 (contacts_last_7_days = 2)
    c, _ = _conv(monkeypatch)
    for m in ["Hello, mujhe order ke baare mein help chahiye.", "Order number hai 4501 1817.", "Last four digits 1267 hain."]:
        c.respond(m)
    t4 = c.respond("Main is hafte teesri baar call kar rahi hoon isi baare mein.")
    assert t4["decision"] == "escalate" and t4["tool_calls"] == ["create_ticket()"]


def test_generator_cannot_claim_actions_it_did_not_do(monkeypatch):
    c, _ = _conv(monkeypatch, generate_reply="Sorry about that. I have cancelled your order for you. You can also use My Orders.")
    for m in ["Hi, I need some help with an order.", "The order number is 3506 1733.", "The last four digits are 2743."]:
        c.respond(m)
    t4 = c.respond("Please cancel it.")
    assert "cancelled your order" not in t4["reply"] and "My Orders" in t4["reply"]


def test_general_policy_question_needs_no_verification(monkeypatch):
    c, prompts = _conv(monkeypatch, generate_reply="Only one coupon.")
    t = c.respond("Can I use two coupon codes on one order?")
    assert t["reply"] == "Only one coupon." and t["tool_calls"] == [] and "ORDER: none" in prompts[-1]


def test_order_view_hides_sensitive_fields():
    from tools import lookup_order
    v = C.order_view(lookup_order("26295015", "8752"))
    assert "641004" not in v and "C100079" not in v and "contacts_last_7_days" not in v and "status: shipped" in v


def test_refund_facts_include_rrn_and_overdue_escalation():                # mirrors PC010 (overdue UPI refund)
    from datetime import date
    from src.kb.loader import load_chunks
    from src.policy.calc import Calendar, build_facts_with_sources, escalation_reasons
    from src.agent.understand import Understanding
    from tools import lookup_order
    import re, pandas as pd
    from src import config
    c = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str).fillna("")
    g = c[c.conversation_id == "PC010"]
    oid = g.order_id.iloc[0]
    val = re.search(r"verification_value=(\w+)", " ".join(g.expected_tool_calls)).group(1)
    order = lookup_order(oid, val)
    by, cal = {x.section_id: x for x in load_chunks()}, Calendar.load()
    und = Understanding(intent="refund_timeline")
    facts, src = build_facts_with_sources(und, None, by, cal, order=order)
    assert any("315633825001" in f for f in facts) and "REF-6" in src
    assert [r[0] for r in escalation_reasons(order, und, None, by, cal)] == ["refund_overdue"]


def test_cancel_fact_picks_the_rule_by_order_status():
    from src.kb.loader import load_chunks
    from src.policy.calc import Calendar, build_facts_with_sources
    from src.agent.understand import Understanding
    by, cal = {x.section_id: x for x in load_chunks()}, Calendar.load()
    und = Understanding(intent="cancel_order")
    f1, _ = build_facts_with_sources(und, None, by, cal, order={"status": "packed"})
    f2, _ = build_facts_with_sources(und, None, by, cal, order={"status": "shipped"})
    assert "CAN-1" in f1[0] and "CAN-2" in f2[0]


# ---------------- fixes from the first real conversation run ----------------
def _facts(order, intent, prod_text=None, **und_kw):
    from src.agent.understand import Understanding
    from src.data.catalog import Catalog
    from src.kb.loader import load_chunks
    from src.policy.calc import Calendar, build_facts_with_sources
    by, cal = {x.section_id: x for x in load_chunks()}, Calendar.load()
    prod = Catalog().resolve(prod_text) if prod_text else None
    return build_facts_with_sources(Understanding(intent=intent, **und_kw), prod, by, cal, order=order)


def test_guard_keeps_a_credit_recorded_in_the_order_but_strips_an_invented_voucher():     # PC023 vs PC015
    from src.agent.guard import strip_unsupported_claims as g
    kept, _ = g("The ₹79 express fee was credited to your wallet.", allowed_amounts={"79"})
    assert "79" in kept
    kept2, removed = g("Sorry. A ₹300 voucher has been added to your wallet.", allowed_amounts={"79"})
    assert "300" not in kept2 and removed


def test_cancel_after_shipping_mentions_refund_timing_only_for_prepaid():                 # PC019/PC020: the '7 to 10 days'
    f, src = _facts({"status": "shipped", "payment_method": "upi"}, "cancel_order")
    assert any("7 to 10 calendar days" in x for x in f) and "REF-1" in src
    f2, _ = _facts({"status": "shipped", "payment_method": "cod"}, "cancel_order")
    assert any("nothing is charged" in x for x in f2) and not any("7 to 10" in x for x in f2)


def test_return_in_qc_gets_a_stage_fact_with_a_computed_date():                           # PC009
    order = {"return_status": "picked_up", "return_pickup_on": "2026-10-20", "is_plus": False}
    f, src = _facts(order, "refund_timeline")
    assert any("quality check (QC)" in x and "22 October 2026" in x for x in f) and "RET-7" in src


def test_reverse_fee_fact_depends_on_plus_status():                                       # PC004 expects the ₹49
    f, _ = _facts({"is_plus": False, "delivered_on": "2026-10-14", "placed_at": "2026-10-06T10:00:00"}, "return_eligibility", "UrbanThread Slim Fit Jeans")
    assert any("₹49" in x for x in f)
    f2, _ = _facts({"is_plus": True}, "return_eligibility", "UrbanThread Slim Fit Jeans")
    assert any("Plus member" in x and "none" in x for x in f2) and not any("₹49 applies" in x for x in f2)


def test_high_value_damage_found_from_the_order_damage_record_even_if_intent_is_wrong():  # PC002 T4
    from src.agent.understand import Understanding
    from src.data.catalog import Catalog
    from src.kb.loader import load_chunks
    from src.policy.calc import Calendar, escalation_reasons
    by, cal = {x.section_id: x for x in load_chunks()}, Calendar.load()
    order = {"damage_sku": "ELE-0021", "items": []}
    r = escalation_reasons(order, Understanding(intent="policy_override"), Catalog().resolve("Zentro Vista 43-inch 4K Smart TV"), by, cal)
    assert [x[0] for x in r] == ["high_value_damage"]
    assert escalation_reasons({"damage_sku": None}, Understanding(intent="policy_override"), None, by, cal) == []


def test_vague_help_is_defined_as_other_in_the_understanding_prompt():                    # PC007 T1
    from src.agent.understand import SYSTEM, INTENTS
    assert "vague request for help" in INTENTS["other"] and 'use intent "other" and needs_order false' in SYSTEM


def test_new_pins():
    from src.retrieval.routes import pinned_sections
    assert "RET-7" in pinned_sections("refund_timeline", None, None, False, True)
    assert "REF-1" in pinned_sections("cancel_order") and "RET-6" in pinned_sections("return_eligibility")


def test_internal_section_ids_never_reach_the_customer():                                  # PC019 leaked "REF-1 ke tahat"
    from src.agent.rag import clean_reply
    assert clean_reply("parcel wapas pahunchne par REF-1 ke tahat refund milta hai.") == "parcel wapas pahunchne par refund milta hai."
    assert clean_reply("Refunds take 1 to 3 days (REF-2, KB-00.3). See RET-2.") == "Refunds take 1 to 3 days."
    assert clean_reply("Is the RETv3-2 window 10 days? Per KB-00.1, policy wins.") == "Is the window 10 days? Policy wins."
    assert clean_reply("Plain sentence. e.g. this stays.") == "Plain sentence. e.g. this stays."   # untouched when no ID was present
    assert "ELE-0012" in clean_reply("SKU ELE-0012 is covered.")                            # SKUs are not section IDs


def test_hindi_spoken_digits_become_order_ids_and_pins():
    from src.agent.normalize import normalize_asr as n
    assert "79363550" in n("ऑर्डर नंबर है सात नौ तीन छह तीन पाँच पाँच शून्य")
    assert "4265" in n("char do chhe paanch hain")
    assert "1234" in n("१२३४")
    assert n("do teen din mein aa jayega") == "do teen din mein aa jayega"                  # a 2-word run stays words
    assert "ek aur sawaal" in n("ek aur sawaal hai")
