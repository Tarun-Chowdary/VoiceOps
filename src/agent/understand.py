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
    "damaged_item": "item arrived damaged, defective, wrong or missing",
    "late_delivery": "order arrived late, compensation for delay",
    "delivery_attempts": "courier could not deliver, customer not home, failed delivery attempts",
    "delivery_area": "whether ZentroMart delivers to a place/pincode, COD or Express availability",
    "warranty": "warranty, ZentroMart Assured, warranty claims",
    "troubleshooting": "a device does not work and the customer wants fix steps",
    "order_status": "where is my order, tracking",
    "fraud": "activity the customer did not do: unknown order, payment or account use",
    "policy_override": "asks for something policy forbids: other customers' data, exceptions, extra compensation",
    "unsupported": "a topic the support documents do not cover (stores, discounts, bulk invoices, etc.)",
    "other": "greeting, thanks, small talk, unclear",
}

SYSTEM = f"""You are the language-understanding module of ZentroMart's customer support agent (Indian online store: groceries, electronics, fashion, home, beauty).
Today's date is {config.TODAY.isoformat()}.
Read the customer's message (English, Hindi, or Hinglish; it may contain speech-recognition errors) and return ONLY one JSON object with these keys:
- "language": "en", "hi" (Devanagari), or "hi-en" (Hinglish in Latin script)
- "intent": exactly one of: {", ".join(INTENTS)}
- "english_query": a short English search query (max 20 words) restating the customer's real question in plain support vocabulary (for example: "return window days for books and stationery"). Keep product names, SKUs and payment methods.
- "payment_method": "upi", "credit_card", "debit_card", "net_banking", "wallet", "cod", or null
- "product_mention": the product name or SKU mentioned, or null
- "mentioned_dates": list of dates in YYYY-MM-DD resolved from the message ("yesterday" = the day before today), or []
- "needs_clarification": true only if the question cannot be answered without one missing detail
Intent guide:
""" + "\n".join(f"- {k}: {v}" for k, v in INTENTS.items()) + "\nDo not answer the customer. Output JSON only."


@dataclass
class Understanding:
    language: str = "en"
    intent: str = "other"
    english_query: str = ""
    payment_method: str | None = None
    product_mention: str | None = None
    mentioned_dates: list = field(default_factory=list)
    needs_clarification: bool = False
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


def understand(text: str) -> Understanding:
    d = _parse(llm.chat(SYSTEM, [{"role": "user", "text": text}], temperature=0.0))
    if not d:                               # never crash a turn: fall back to the raw text
        return Understanding(english_query=text, ok=False)
    intent = str(d.get("intent", "other")).strip().lower()
    return Understanding(
        language=str(d.get("language", "en")),
        intent=intent if intent in INTENTS else "other",
        english_query=str(d.get("english_query") or text),
        payment_method=d.get("payment_method") or None,
        product_mention=d.get("product_mention") or None,
        mentioned_dates=[x for x in (d.get("mentioned_dates") or []) if isinstance(x, str)],
        needs_clarification=bool(d.get("needs_clarification", False)))
