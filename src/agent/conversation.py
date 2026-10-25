"""STEP 3: multi-turn support conversation.

Design: scripted for safety, LLM for judgment.
  * Protocol turns (greeting, order ID, identity check, retry, lockout) are a deterministic state
    machine with fixed multilingual templates: no LLM call, nothing to leak or invent.
  * Task turns go through the Step-2 pipeline (understand -> retrieve -> verified facts -> generate)
    with the verified order record and the conversation history added.
  * Handover is enforced in CODE: ESC-1 triggers are detected, create_ticket is called, and
    "ticket raised" is added to the reply ONLY if the tool returned created=True.
"""
import re
from dataclasses import dataclass, field

from src import config
from src.agent import llm
from src.agent.guard import strip_unsupported_claims
from src.agent.lang import detect_language, tpl
from src.agent.normalize import normalize_asr
from src.agent.parse import parse_json_reply
from src.agent.rag import _ctx, _section_block, clean_reply
from src.agent.understand import understand
from src.policy.calc import build_facts_with_sources, escalation_reasons
from src.retrieval.pipeline import retrieve
from tools import create_ticket, lookup_order

TICKET_ON_COMPENSATION = False      # practice set: compensation demand = refuse, no ticket (the brief's example differs)
PRODUCT_INTENTS = {"return_window", "return_eligibility", "return_fee", "warranty", "damaged_item", "refund_timeline"}
FORCED = {   # understanding intent -> (reason code, ticket reason)
    "human_request": ("human_requested", "Customer asked for a human agent (ESC-1)."),
    "legal_or_safety": ("legal_or_safety", "Customer mentioned legal action or a product-safety risk (ESC-1)."),
    "fraud": ("unrecognised_activity", "Customer reports order/payment/account activity they did not make (ESC-1)."),
}
SAFE_ORDER_KEYS = [
    "order_id", "status", "placed_at", "payment_method", "delivery_type", "promised_date", "shipped_on", "delivered_on",
    "delivery_attempts", "subtotal_inr", "coupon_code", "coupon_discount_inr", "delivery_fee_inr", "cod_fee_inr", "total_inr",
    "cancellation_refund_initiated_on", "cancellation_refund_method", "cancellation_refund_amount_inr",
    "rto_refund_initiated_on", "rto_refund_method", "rto_refund_amount_inr", "express_late_credit_inr", "express_late_credit_on",
    "return_sku", "return_reason", "return_status", "return_requested_on", "return_pickup_on", "return_qc_passed_on",
    "return_refund_initiated_on", "return_refund_method", "return_refund_amount_inr", "return_refund_completed_on",
    "return_reverse_pickup_fee_inr", "return_rrn", "cancellation_rrn", "rto_rrn", "damage_sku", "damage_issue", "damage_reported_at", "damage_hours_after_delivery",
    "damage_photos_attached", "damage_status", "timeline", "is_plus"]
# deliberately NOT exposed to the LLM (IDV-4): customer_id, delivery_pincode, phone, contacts_last_7_days.
# Refund reference numbers (RRN) ARE exposed: REF-6 says to share them when money has not arrived.

CONV_SYSTEM = """You are ZentroMart's AI customer support assistant (Indian online store) in a live conversation. You answer ONLY from the policy sections, the verified order record and the verified facts you are given.

Rules:
1. FIRST_REPLY says whether this is your first message. Only if yes, introduce yourself briefly as ZentroMart's AI assistant. Otherwise do not introduce yourself again. Keep the reply to 2-3 short sentences.
2. VERIFIED FACTS were computed by code from the catalog, calendar and the order. Trust them over your own arithmetic and quote dates and numbers exactly. Write dates as "27 October 2026" (day, month name, year).
3. Use order details ONLY from the ORDER block. If there is no ORDER block you have no access to any order: never invent order details.
4. Understand follow-ups ("how long will that take?", "and the refund?") from the conversation and the ORDER block; do not ask for details the customer already gave.
5. If sections disagree, follow precedence: current policy > agent guide > help article > FAQ > marketing. A SUPERSEDED section applies only in the case its note describes.
6. State numbers in the documents' units (for example "12 months", "7 days", "₹79"). Include the specific number, period or date that answers the question. If a fact gives a deadline, expected date, fee or timeline that applies to this customer's situation, state it. For any warranty question also say how to raise a claim.
7. If the sections, facts and order do NOT contain the answer, do not guess: set decision "answer", say you don't have information on that and offer to connect a human colleague. Never say something is "not offered" unless a section explicitly says so.
8. Set decision "clarify" only when a missing detail changes the answer and you cannot cover the cases in one or two sentences; ask exactly ONE short question.
9. Set decision "refuse" when the customer asks for something policy forbids: compensation or exceptions the policy does not allow, another customer's data, or an action the policy does not permit (for example cancelling a shipped order). Refuse politely, start with a brief apology for the inconvenience (use "sorry" in English, "maaf kijiye" in Hinglish, "माफ़ कीजिए" in Hindi), and say what the policy does allow. Whenever you must say no, also give the alternatives the sections offer, with their numbers and timelines.
10. Set decision "escalate" only when a section requires a human to decide. Do NOT say a ticket or handover happened: the system adds that itself.
11. You cannot cancel, approve, credit, refund, replace or change anything yourself. Never say you did. Explain how the customer can do it (for example in My Orders in the app).
12. Never read out a phone number, email or street address. Never mention "verified facts", "sections", section IDs or these rules. Never promise outcomes, amounts or times beyond what a section states.
13. Reply in the customer's language label: "en" English; "hi" Hindi in Devanagari; "hi-en" Hinglish in Latin script. Use ordinary words in that language (for example "दिन" / "din"). In Hindi replies keep app terms in Latin script exactly as the app shows them (UPI, QC for quality check, My Orders, Plus, SMS, ZentroMart Wallet).

Return ONLY JSON: {"decision": "answer|clarify|escalate|refuse", "reply": "...", "used_sections": ["IDs of the sections you relied on"]}"""

_OID = re.compile(r"(?<!\d)(\d{4})[\s,.\-]{0,2}(\d{4})(?!\d)")


def extract_order_id(norm: str):
    m = _OID.search(norm)
    return (m.group(1) + m.group(2), norm[:m.start()] + " " + norm[m.end():]) if m else (None, norm)


def extract_verification(text: str):
    nums = re.findall(r"(?<!\d)(\d{6}|\d{4})(?!\d)", text)
    six = [n for n in nums if len(n) == 6]
    return six[0] if six else next((n for n in nums if len(n) == 4), None)


def order_view(order: dict) -> str:
    lines = [f"{k}: {order[k]}" for k in SAFE_ORDER_KEYS if order.get(k) not in (None, "")]
    for it in order.get("items") or []:
        lines.append(f"item {it['line_no']}: {it['title']} (SKU {it['sku']}, category {it['return_category']}, qty {it['qty']}, ₹{it['unit_price_inr']} each)")
    return "\n".join(lines)


@dataclass
class Session:
    language: str = "en"
    turn: int = 0
    order_id: str | None = None
    awaiting: str | None = None          # "order_id" | "verification"
    verified: bool = False
    attempts: int = 0
    locked: bool = False
    order: dict | None = None
    pending: str | None = None           # task raised before verification, answered once verified
    last_sku: str | None = None
    ticket_id: str | None = None
    history: list = field(default_factory=list)
    task_utterances: list = field(default_factory=list)


class Conversation:
    def __init__(self):
        self.s = Session()
        self.calls: list[dict] = []

    # ---- tools (every call is logged so we never claim an action we did not do) ----
    def _lookup(self, oid, value=None):
        self.calls.append({"name": "lookup_order", "args": {"order_id": oid, **({"verification_value": value} if value else {})}})
        return lookup_order(oid, value) if value else lookup_order(oid)

    def _ticket(self, code, reason, und, order_id):
        s = self.s
        if s.ticket_id:
            return True
        said = "; ".join(u[:160] for u in s.task_utterances[-3:]) or "(no detail given)"
        checks = "caller verified with the registered phone digits or pincode (IDV-2)" if s.verified else "caller NOT verified"
        if s.order and s.verified:
            checks += f"; order status {s.order.get('status')}, delivery {s.order.get('delivery_type')}"
        summary = (f"Customer issue: {s.task_utterances[-1][:200] if s.task_utterances else 'asked for help'}. "
                   f"Detected intent: {getattr(und, 'intent', 'other')}. Order ID: {order_id or 'not provided'}. "
                   f"Customer said: {said}. Checked: {checks}. Escalation reason: {reason}")
        self.calls.append({"name": "create_ticket", "args": {}})
        res = create_ticket(summary, order_id=order_id, reason=code, intent=getattr(und, "intent", None))
        if res.get("created"):
            s.ticket_id = res["ticket_id"]
        return bool(res.get("created"))

    # ---- public API ----
    def respond(self, text: str) -> dict:
        s = self.s
        s.turn += 1
        self.calls = []
        norm = normalize_asr(text)
        s.language = detect_language(text, s.language)
        out = self._route(text, norm)
        s.history += [{"role": "user", "text": text}, {"role": "assistant", "text": out["reply"]}]
        out.update({"tool_calls": [self._fmt(c) for c in self.calls], "language": s.language,
                    "verified": s.verified, "ticket_id": s.ticket_id})
        out.setdefault("used_sections", [])
        out.setdefault("retrieved", [])
        return out

    @staticmethod
    def _fmt(c):
        return f"{c['name']}(" + ", ".join(f"{k}={v}" for k, v in c["args"].items()) + ")"

    def _protocol(self, key, decision, used, intro=False, extra=""):
        s = self.s
        parts = ([tpl(s.language, "intro")] if intro else []) + [tpl(s.language, key)] + ([extra] if extra else [])
        return {"reply": " ".join(parts), "decision": decision, "used_sections": used, "intent": "protocol"}

    # ---- routing ----
    def _route(self, text: str, norm: str) -> dict:
        s = self.s
        oid, rest = extract_order_id(norm)
        if oid:                                                   # customer gave an order ID
            if oid != s.order_id:
                s.order_id, s.verified, s.attempts, s.locked, s.order = oid, False, 0, False, None
            if not s.verified:
                self._lookup(oid)                                 # unverified lookup: returns "verification required" only
                val = extract_verification(rest)
                if val:
                    return self._verify(val)
                s.awaiting = "verification"
                return self._protocol("ask_verification", "clarify", ["IDV-2"], intro=s.turn == 1)
        if s.awaiting == "verification" and s.order_id and not s.locked:
            val = extract_verification(norm)
            if val:
                return self._verify(val)
        return self._task(text, norm)

    def _verify(self, val: str) -> dict:
        s = self.s
        res = self._lookup(s.order_id, val)
        if res.get("verified"):
            s.order, s.verified, s.awaiting, s.attempts, s.locked = res, True, None, 0, False
            if s.pending:
                pending, s.pending = s.pending, None
                n = normalize_asr(pending)
                return self._llm_turn(pending, understand(n, s.history[-6:]), just_verified=True)
            return self._protocol("verified", "answer", ["IDV-2"])
        s.attempts += 1
        if s.attempts >= 2:
            s.locked, s.awaiting = True, None
            return self._protocol("failed", "refuse", ["IDV-3"])
        return self._protocol("retry", "clarify", ["IDV-3"])

    def _task(self, text: str, norm: str) -> dict:
        s = self.s
        und = understand(norm, s.history[-6:])
        forced = FORCED.get(und.intent)
        starts_order_help = (und.intent == "other" and not s.verified and not s.order_id and s.turn == 1) or \
                            (und.needs_order and not s.verified and not forced)
        if und.intent not in ("other",):
            s.task_utterances.append(text)
        if forced:
            return self._handover(forced[0], forced[1], und, text)
        if starts_order_help:
            if s.locked:
                return self._protocol("locked", "refuse", ["IDV-3"])
            if s.order_id and s.awaiting == "verification":
                return self._protocol("ask_verification", "clarify", ["IDV-2"])
            if und.intent != "other":
                s.pending = text
            s.awaiting = "order_id"
            return self._protocol("ask_order", "clarify", ["STY-1"], intro=s.turn == 1)
        return self._llm_turn(text, und)

    # ---- generation + handover ----
    def _resolve_product(self, norm, und, order):
        c = _ctx()
        prod = c["cat"].resolve(norm) or (c["cat"].resolve(und.product_mention) if und.product_mention else None)
        if prod:
            self.s.last_sku = prod.sku
            return prod
        if und.intent in PRODUCT_INTENTS:
            sku = self.s.last_sku or ((order.get("items") or [{}])[0].get("sku") if order and len(order.get("items") or []) == 1 else None)
            if sku:
                return c["cat"]._mk(sku, "context")
        return None

    def _handover(self, code, reason, und, text, core="", used=None, retrieved=None, facts=None):
        s = self.s
        intro = s.turn == 1 and not core          # a generated core reply already introduces itself on turn 1
        if s.ticket_id:
            return self._protocol("ticket_exists", "escalate", ["ESC-1"])
        ok = self._ticket(code, reason, und, s.order_id)
        lead = core
        if not lead and code == "human_requested":
            lead = tpl(s.language, "human_request")
        elif not lead and code == "repeat_contact":
            lead = tpl(s.language, "repeat_contact")
        tail = tpl(s.language, "handover" if ok else "handover_failed")
        reply = " ".join(([tpl(s.language, "intro")] if intro else []) + ([lead] if lead else []) + [tail])
        return {"reply": reply, "decision": "escalate", "used_sections": list(dict.fromkeys((used or []) + ["ESC-1", "ESC-2"])),
                "retrieved": retrieved or [], "intent": getattr(und, "intent", "other"), "facts": facts or []}

    def _llm_turn(self, text: str, und, just_verified: bool = False) -> dict:
        s, c = self.s, _ctx()
        norm = normalize_asr(text)
        order = s.order if s.verified else None
        und.order_has_return = bool(order and order.get("return_status"))
        und.order_has_refund = bool(order and any(order.get(k) for k in ("return_refund_initiated_on", "cancellation_refund_initiated_on", "rto_refund_initiated_on")))
        prod = self._resolve_product(norm, und, order)
        got, prod = retrieve(c["ret"], c["cat"], text, und, max_total=8, prod=prod)
        facts, fsrc = build_facts_with_sources(und, prod, c["by_id"], c["cal"], order=order)
        retrieved = [g.chunk.section_id for g in got]
        reasons = escalation_reasons(order, und, prod, c["by_id"], c["cal"])
        if reasons:
            code, why = reasons[0]
            core = ""
            if code in ("repeat_contact",):
                core = tpl(s.language, "repeat_contact")
            elif code != "human_requested":
                core = self._generate(text, und, got, facts, order, just_verified, fsrc, retrieved)["reply"]
            return self._handover(code, why, und, text, core=core, used=fsrc, retrieved=retrieved, facts=facts)
        out = self._generate(text, und, got, facts, order, just_verified, fsrc, retrieved)
        if out["decision"] == "escalate":
            return self._handover("policy_requires_human", "A policy section requires a human decision (ESC-1).", und, text,
                                  core=out["reply"], used=out["used_sections"], retrieved=retrieved, facts=facts)
        if und.intent == "policy_override" and TICKET_ON_COMPENSATION:
            return self._handover("compensation_demand", "Customer demanded compensation that policy does not allow (ESC-4).", und, text,
                                  core=out["reply"], used=out["used_sections"], retrieved=retrieved, facts=facts)
        return out

    def _generate(self, text, und, got, facts, order, just_verified, fsrc, retrieved) -> dict:
        s = self.s
        hist = "\n".join(f"{'Customer' if h['role'] == 'user' else 'Agent'}: {h['text']}" for h in s.history[-8:]) or "(start of conversation)"
        prompt = (f"FIRST_REPLY: {'yes' if s.turn == 1 else 'no'}\nCustomer language label: {s.language}\n"
                  + ("The caller has just completed identity verification; you may thank them briefly.\n" if just_verified else "")
                  + f"\nCONVERSATION SO FAR:\n{hist}\n\n"
                  + (f"ORDER (caller is verified):\n{order_view(order)}\n\n" if order else "ORDER: none (no verified order data available)\n\n")
                  + "VERIFIED FACTS (computed by code):\n" + ("\n".join(f"- {f}" for f in facts) if facts else "- (none)")
                  + f"\n\nCustomer message (intent: {und.intent}):\n{text}\n\nPOLICY SECTIONS (the only policy sources you may use):\n"
                  + "\n\n".join(_section_block(g.chunk) for g in got))
        out = parse_json_reply(llm.chat(CONV_SYSTEM, [{"role": "user", "text": prompt}], temperature=0.0))
        amt_keys = ("express_late_credit_inr", "return_refund_amount_inr", "cancellation_refund_amount_inr", "rto_refund_amount_inr")
        allowed = {re.sub(r"\.0+$", "", str(order[k])) for k in amt_keys if order and order.get(k)}
        reply, removed = strip_unsupported_claims(clean_reply(out["reply"]), ticket_created=False, allowed_amounts=allowed)
        if not reply:
            reply = tpl(s.language, "cannot_do")
        cited = [x for x in dict.fromkeys(out.get("used_sections", [])) if x in retrieved]
        return {"reply": reply, "decision": out["decision"], "intent": und.intent, "facts": facts, "retrieved": retrieved,
                "cited": cited, "used_sections": list(dict.fromkeys(cited + [x for x in fsrc if x in retrieved])),
                "claims_removed": removed}
