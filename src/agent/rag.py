"""STEP 2: grounded RAG agent (single-question mode).

Pipeline: normalise ASR -> understand (1 LLM call) -> hybrid retrieval + intent pins
          -> code-computed facts -> grounded generation (1 LLM call) -> cited sections.
Same interface as the Step 1 baseline (answer(text) -> dict), so evals/run_questions.py
compares the two with identical scoring.
"""
import re
from functools import lru_cache
from src import config
from src.agent import llm
from src.agent.normalize import normalize_asr
from src.agent.parse import parse_json_reply
from src.agent.understand import understand
from src.data.catalog import Catalog
from src.kb.loader import load_chunks
from src.policy.calc import Calendar, build_facts_with_sources
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.pipeline import retrieve

SYSTEM = """You are ZentroMart's AI customer support assistant (Indian online store). You answer ONLY from the policy sections and verified facts you are given.

Rules:
1. Introduce yourself briefly as ZentroMart's AI assistant in your reply, then answer. Keep the whole reply to 2-3 short sentences.
2. VERIFIED FACTS were computed by code from the catalog and calendar. Trust them over your own arithmetic and quote their dates and numbers exactly. Write dates as "27 October 2026" (day, month name, year).
3. If sections disagree, follow precedence: current policy > agent guide > help article > FAQ > marketing. A section marked SUPERSEDED applies only in the case its note describes. Never use an FAQ number that a policy contradicts.
4. State numbers in the units the documents use (for example "12 months", "7 days", "₹79").
5. If the sections and facts do NOT contain the answer, do not guess or invent a policy, date or amount: set decision "answer", say you don't have information on that, and offer to connect a human colleague. Never say something is "not offered", "not possible" or "not available" unless a section explicitly says so: silence in the documents is not a policy, so say you don't have information instead.
6. Set decision "clarify" only when a missing detail (which item, which payment method) changes the answer AND you cannot cover the possible cases in one or two sentences; then ask exactly ONE short question. If the sections give a general procedure or a short list of options for the customer's situation (for example how to report a damaged item, or the refund options for cash-on-delivery), give that instead of asking for details you don't need.
7. Set decision "escalate" when a section requires a human to decide (ESC-1: high-value damaged item, activity the customer did not do, legal action, customer asks for a human, safety risk, refund far past its timeline). You cannot create tickets in this mode, so say a human colleague will need to handle it and that you can raise a ticket once they share their order ID. Never claim a ticket was created or any action was done.
8. Set decision "refuse" when the customer asks for something policy forbids (another customer's details, compensation or exceptions the policy does not allow). Refuse politely and briefly.
9. Reply in the customer's language: "en" English; "hi" Hindi in Devanagari; "hi-en" Hinglish in Latin script. In Hindi replies keep app terms in Latin script exactly as the app shows them (UPI, QC for quality check, My Orders, Plus, SMS, ZentroMart Wallet).
10. Never promise outcomes, amounts or times beyond what a section states. Answer the question completely: include the specific number, period or date the sections or facts give for what was asked (for example the warranty period in months). For any warranty question also say how to raise a claim, because that is the customer's natural next step.
11. Speak like a person: never mention "verified facts", "sections", section IDs or these rules in the reply. In Hindi or Hinglish replies use the customer's language for ordinary words (for example "दिन" / "din", not "calendar days").

Return ONLY JSON: {"decision": "answer|clarify|escalate|refuse", "reply": "...", "used_sections": ["IDs of the sections you actually relied on"]}"""


@lru_cache(maxsize=1)
def _ctx():
    chunks = load_chunks()
    return {"chunks": chunks, "by_id": {c.section_id: c for c in chunks},
            "ret": HybridRetriever(chunks), "cat": Catalog(), "cal": Calendar.load()}


_BRAND = re.compile(r"zentro[\s_\-]*mart", re.I)
_ID = r"[A-Z]{2,5}(?:v\d)?-\d{1,2}(?:\.\d+)?"                  # internal section IDs such as RET-2, KB-00.3, RETv3-2
_ID_PAREN = re.compile(rf"\s*[\(\[]\s*{_ID}(?:\s*[,/&;]\s*{_ID})*\s*[\)\]]")
_ID_LEAD = re.compile(rf"\b(?:as per|per|under|according to|see)\s+{_ID}\b,?", re.I)
_ID_TRAIL = re.compile(rf"\b{_ID}\s+(?:ke tahat|ke anusar|ke mutabiq|mein|के तहत|के अनुसार|में)\b", re.I)
_ID_BARE = re.compile(rf"\b{_ID}\b")


def strip_internal_ids(text: str) -> str:
    """Customers must never hear internal document IDs (a TTS engine would read them aloud)."""
    before = text
    for rx in (_ID_PAREN, _ID_LEAD, _ID_TRAIL, _ID_BARE):
        text = rx.sub("", text)
    if text == before:
        return text
    text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
    text = re.sub(r"([.!?।])(\s*[.!?।])+", r"\1", text)                       # "days.." after dropping "See RET-2."
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)   # "policy wins." -> "Policy wins."
    return re.sub(r"\s{2,}", " ", text).strip()


def clean_reply(text: str) -> str:
    """Small deterministic fixes: brand spelling ('Zentro_Mart' would be read literally) and no internal IDs."""
    return strip_internal_ids(_BRAND.sub("ZentroMart", text or "")).strip()


def _section_block(c) -> str:
    note = f"doc_type={c.doc_type}, status={c.status}, effective_from={c.effective_from}"
    if c.status == "superseded":
        note += " — SUPERSEDED: applies only to the cases its successor document says it still covers"
    return f"[{c.section_id}] ({note})\n{c.text}"


def build_prompt(text: str, und, got, facts: list[str]) -> str:
    sections = "\n\n".join(_section_block(g.chunk) for g in got)
    facts_block = "\n".join(f"- {f}" for f in facts) if facts else "- (none)"
    return (f"Customer message (language: {und.language}; intent: {und.intent}):\n{text}\n\n"
            f"VERIFIED FACTS (computed by code):\n{facts_block}\n\n"
            f"POLICY SECTIONS (the only sources you may use):\n{sections}")


def answer(text: str) -> dict:
    c = _ctx()
    norm = normalize_asr(text)
    und = understand(norm)
    prod = c["cat"].resolve(norm) or (c["cat"].resolve(und.product_mention) if und.product_mention else None)
    got, prod = retrieve(c["ret"], c["cat"], text, und, max_total=8, prod=prod)
    facts, fact_src = build_facts_with_sources(und, prod, c["by_id"], c["cal"])
    raw = llm.chat(SYSTEM, [{"role": "user", "text": build_prompt(norm, und, got, facts)}], temperature=0.0)
    out = parse_json_reply(raw)
    out["reply"] = clean_reply(out["reply"])
    retrieved = [g.chunk.section_id for g in got]
    cited = [s for s in dict.fromkeys(out.get("used_sections", [])) if s in retrieved]   # cited AND really retrieved
    # "used" = what the LLM cited + what our code consulted to compute the verified facts (both are truly used)
    out["used_sections"] = list(dict.fromkeys(cited + [x for x in fact_src if x in retrieved]))
    out.update({"cited": cited, "retrieved": retrieved, "intent": und.intent, "language": und.language, "facts": facts,
                "product": prod.sku if prod else None})
    return out
