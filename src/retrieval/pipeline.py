"""retrieve(): understanding + product facts + hybrid search + pinned sections + precedence.

Order of the final context (what the LLM reads, and what we log as 'retrieved chunks'):
  1. pinned sections for the intent (rule sections first)   reason = "pinned"
  2. hybrid hits, policy > agent guide > help > FAQ > marketing (KB-00.1)  reason = "hybrid#rank"
Superseded documents are demoted to the end unless explicitly pinned (RET-3 says when v3 applies).
"""
from dataclasses import dataclass
from src.agent.normalize import normalize_asr
from src.retrieval.routes import pinned_sections


@dataclass
class Item:
    chunk: object
    reason: str


def retrieve(retriever, catalog, text: str, und=None, max_total: int = 8, n_hybrid: int = 8):
    """text: raw customer text (ASR or typed). und: an Understanding (or None for plain hybrid)."""
    norm = normalize_asr(text)
    prod = catalog.resolve(norm)
    cat = prod.return_category if prod and prod.return_category else None
    bm_q = norm + (" " + cat if cat else "")
    dn_q = ((und.english_query + " ") if und else "") + (cat or "") if und else bm_q
    if und:
        bm_q = (und.english_query + " " + bm_q).strip()
    hits = retriever.search(bm_q, k=n_hybrid, dense_query=dn_q.strip() or bm_q)

    by_id = {c.section_id: c for c in retriever.chunks}
    pins = pinned_sections(und.intent, und.payment_method, cat) if und else []
    out, seen = [], set()
    for sid in pins:
        if sid in by_id and sid not in seen:
            out.append(Item(by_id[sid], "pinned"))
            seen.add(sid)
    rest = []
    for rank, h in enumerate(hits, 1):
        if h.chunk.section_id not in seen:
            rest.append((h.chunk.status == "superseded", h.chunk.priority, rank, h))
    rest.sort(key=lambda t: (t[0], t[1], t[2]))      # superseded last, then precedence, then retrieval rank
    for _, _, rank, h in rest:
        out.append(Item(h.chunk, f"hybrid#{rank}"))
    return out[:max_total], prod
