"""One LLM call per turn that turns raw (possibly Hindi / Hinglish / ASR-noisy) customer text
into a small structured object: intent, an English search query, and entities.

Why a single call: it feeds routing, retrieval AND (later) the conversation state, so we pay
for one request per turn, which matters under a free-tier daily quota.
"""
import json, re
from dataclasses import dataclass, field
from src import config
from src.agent import llm

INTENTS = {
    "refund_timeline": "when or by which date a refund will arrive; refund speed by payment method",
    "return_window": "how many days the customer has to return an item",
    "return_eligibility": "whether an item can be returned (opened, used, wrong size, category rules)",
    "return_fee": "reverse-pickup or return fees, Plus-member waiver, free collection",
    "cancel_order": "cancelling an order, before or after it ships",
    "damaged_item": "item arrived damaged, defective, wrong or missing, or a follow-up about a damage report / replacement request",
    "late_delivery": "order arrived late, compensation for delay",
    "delivery_attempts": "courier could not deliver, customer not home, failed delivery attempts",
    "delivery_area": "whether ZentroMart delivers to a place/pincode, COD or Express availability",
    "warranty": "warranty, ZentroMart Assured, warranty claims",
    "troubleshooting": "a device does not work and the customer wants fix steps",
    "order_status": "where is my order, tracking",
    "fraud": "activity the customer did not do: unknown order, payment or account use",
    "human_request": "the customer asks to speak to a human agent / person",
    "legal_or_safety": "mentions legal action, consumer court, police complaint, or a product safety risk (swelling battery, smoke, injury)",
    "policy_override": "asks for something policy forbids: other customers' data, exceptions, extra compensation",
    "unsupported": "a topic the support documents do not cover (stores, discounts, bulk invoices, etc.)",
    "other": "greeting, thanks, small talk, unclear, or a vague request for help that does not say what the problem is (for example 'I need help with an order')",
}

SYSTEM = f"""You are the language-understanding module of ZentroMart's customer support agent (Indian online store: groceries, electronics, fashion, home, beauty).
Today's date is {config.TODAY.isoformat()}.
Read the customer's message (English, Hindi, or Hinglish; it may contain speech-recognition errors) and return ONLY one JSON object with these keys:
- "language": "en", "hi" (Devanagari), or "hi-en" (Hinglish in Latin script)
- "intent": exactly one of: {", ".join(INTENTS)}
- "english_query": a short English search query (max 20 words) restating the customer's real question in plain support vocabulary (for example: "return window days for books and stationery"). Keep product names, SKUs and payment methods.
- "payment_method": "upi", "credit_card", "debit_card", "net_banking", "wallet", "cod", or null
- "product_mention": the product name or SKU mentioned, or null. Speech recognition garbles spoken codes (letters split or replaced by similar-sounding words, zeros dropped); SKUs look like three letters + four digits, for example ELE-0012: reconstruct it in that form if you can
- "dated_events": list of {{"event": "refund_initiated" | "order_placed" | "delivered" | "other", "date": "YYYY-MM-DD"}} for dates the customer mentions (resolve "yesterday", "parso" = the day before yesterday, etc. from today's date), or []
- "needs_order": true if answering requires the customer's OWN specific order (its status, its refund, cancelling it, returning something they bought, a damage report); false for general policy questions and greetings
- "needs_clarification": true only if the question cannot be answered without one missing detail
Intent guide:
""" + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items()) + "\nIf the customer only says they need help (with an order) but not what the problem or question is, use intent \"other\" and needs_order false.\nDo not answer the customer. Output JSON only."


@dataclass
class Understanding:
    language: str = "en"
    intent: str = "other"
    english_query: str = ""
    payment_method: str | None = None
    product_mention: str | None = None
    events: list = field(default_factory=list)
    needs_clarification: bool = False
    needs_order: bool = False
    order_has_refund: bool = False       # set by the conversation from the verified order, not by the LLM
    order_has_return: bool = False
    ok: bool = True


def _parse(text: str) -> dict | None:
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def understand(text: str, history: list[dict] | None = None) -> Understanding:
    """history: recent [{"role": "user"|"assistant", "text": ...}] so follow-ups like 'how long will that take?' resolve."""
    msg = text
    if history:
        ctx = "\n".join(f"{'Customer' if h['role'] == 'user' else 'Agent'}: {h['text']}" for h in history[-6:])
        msg = f"Conversation so far:\n{ctx}\n\nLatest customer message (classify THIS one, using the conversation to resolve references like 'that' or 'it'):\n{text}"
    d = _parse(llm.chat(SYSTEM, [{"role": "user", "text": msg}], temperature=0.0))
    if not d:                               # never crash a turn: fall back to the raw text
        return Understanding(english_query=text, ok=False)
    intent = str(d.get("intent", "other")).strip().lower()
    return Understanding(
        language=str(d.get("language", "en")),
        intent=intent if intent in INTENTS else "other",
        english_query=str(d.get("english_query") or text),
        payment_method=d.get("payment_method") or None,
        product_mention=d.get("product_mention") or None,
        events=[e for e in (d.get("dated_events") or []) if isinstance(e, dict) and e.get("date")],
        needs_clarification=bool(d.get("needs_clarification", False)),
        needs_order=bool(d.get("needs_order", False)))
