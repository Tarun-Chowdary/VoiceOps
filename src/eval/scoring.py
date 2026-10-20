"""Scoring for practice/questions.csv, following the dataset README conventions:
 - ' ; ' separates required items (ALL must be present)
 - ' / ' separates alternatives inside an item (ANY one is enough)
"""
import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or "")).lower()
    s = s.replace("\u20b9", "rs ").replace(",", "")   # rupee sign, thousands commas
    return re.sub(r"\s+", " ", s)


def _items(cell):
    if cell is None or str(cell).strip() in ("", "nan"):
        return []
    return [i.strip() for i in str(cell).split(" ; ") if i.strip()]


def _alts(item):
    return [a.strip() for a in item.split(" / ") if a.strip()]


def check_must_include(reply: str, cell) -> tuple[bool, list]:
    r = _norm(reply)
    missing = [it for it in _items(cell)
               if not any(_norm(a) in r for a in _alts(it))]
    return (not missing), missing


def check_must_not_include(reply: str, cell) -> tuple[bool, list]:
    r = _norm(reply)
    hits = [a for it in _items(cell) for a in _alts(it) if _norm(a) in r]
    return (not hits), hits


def decision_ok(decision: str, expected, acceptable) -> bool:
    ok = set(_items(acceptable)) | ({str(expected).strip()} if expected else set())
    return str(decision).strip().lower() in {o.lower() for o in ok}


def score_row(row, reply: str, decision: str) -> dict:
    inc_ok, missing = check_must_include(reply, row.get("must_include"))
    exc_ok, hits = check_must_not_include(reply, row.get("must_not_include"))
    dec_ok = decision_ok(decision, row.get("expected_decision"), row.get("acceptable_decisions"))
    return {"decision_ok": dec_ok, "include_ok": inc_ok, "exclude_ok": exc_ok,
            "pass": dec_ok and inc_ok and exc_ok,
            "missing": missing, "forbidden_hits": hits}


def retrieval_hit(row, used_ids) -> bool | None:
    """Did retrieval return the gold section(s)? Gold items like 'WAR-2' (KB sections)
    or 'products.csv:ELE-0012' (a data row, not a KB chunk)."""
    gold = [g for g in _items(row.get("gold_evidence")) if ":" not in g]
    if not gold:
        return None
    used = {u.strip() for u in used_ids}
    return all(g in used for g in gold)
