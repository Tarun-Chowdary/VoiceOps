"""Product resolver: find which catalog product a customer is talking about, even through speech-recognition noise.

Order of evidence (most to least specific):
  1. an explicit SKU written like  ELE-0012 / ele 12 / ELE0012   (prefix must be a real catalog prefix)
  2. the product title, tolerant of typos and ASR damage: the colour suffix is ignored, spoken numbers and units are
     expanded ("seven" -> 7, "gigabytes" -> gb) and near-spellings match ("curta" ~ "Kurta")
  3. a spoken SKU cue ("sku", "skew", "s q", "sk u" ...) followed by a number. SKU numbers are UNIQUE across
     prefixes (1-70), so the number alone identifies the product even when "ELE" was heard as "early" or "e le".
Facts (category, warranty, price, Assured) always come from data/products.csv, never from the LLM.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import pandas as pd
from src import config

_TOK = re.compile(r"[a-z0-9]+")
_CUE = re.compile(r"\b(?:sku|skus|skew|skue|skoo|scu|sk|sq|sqe|sque|s\s?k\s?u|sk\s?u|s\s?q\s?e?)\b")
_ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
         "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70}
_SYN = {"gigabytes": "gb", "gigabyte": "gb", "gigs": "gb", "inches": "inch", "litres": "litre", "liters": "litre"}


@dataclass
class ProductMatch:
    sku: str
    title: str
    return_category: str
    warranty_months: str
    assured: str
    price_inr: str
    method: str                      # 'sku' | 'title' | 'sku_cue' | 'context'
    score: float = 1.0
    candidates: list = field(default_factory=list)   # other SKUs with equal score (e.g. colour variants)


def _skeleton(w: str) -> str:
    """Consonant skeleton for ASR-damaged names: 'castrel' and 'Kestrel' -> 'kstrl'; 'curta' and 'Kurta' -> 'krt'."""
    w = re.sub(r"ph", "f", w.lower())
    w = re.sub(r"ck|c|q", "k", w)
    w = re.sub(r"z", "s", w)
    w = re.sub(r"[aeiouyhw]", "", w)
    return re.sub(r"(.)\1+", r"\1", w)


def _similar(a: str, b: str) -> bool:
    """Same word allowing for speech-recognition spelling: close string match, or the same consonant skeleton (>= 4 consonants)."""
    if a == b:
        return True
    if len(a) <= 3 or len(b) <= 3:
        return False
    if SequenceMatcher(None, a, b).ratio() >= 0.78:
        return True
    sa = _skeleton(a)
    return len(sa) >= 4 and sa == _skeleton(b)


def _number_words(tokens: list[str]) -> int | None:
    """First number in a token list: digits ('12'), or words ('two', 'twenty two')."""
    for i, t in enumerate(tokens):
        if t.isdigit() and len(t) <= 4:
            return int(t)
        if t in _TENS:
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            return _TENS[t] + (_ONES[nxt] if nxt in _ONES and _ONES[nxt] < 10 else 0)
        if t in _ONES:
            return _ONES[t]
    return None


class Catalog:
    def __init__(self):
        self.df = pd.read_csv(config.DATA_DIR / "products.csv", dtype=str).fillna("")
        self.by_sku = {r.sku: r for r in self.df.itertuples()}
        self.prefixes = {s[:3].lower() for s in self.by_sku}
        self.by_number = {}
        for s in self.by_sku:
            self.by_number.setdefault(int(s.split("-")[1]), []).append(s)
        self.core = {r.sku: _TOK.findall(r.title.split(",")[0].lower()) for r in self.df.itertuples()}
        self._sku_rx = re.compile(r"\b(" + "|".join(sorted(self.prefixes)) + r")[-\s]?(\d{1,4})\b")

    def _mk(self, sku, method, score=1.0, cands=()):
        r = self.by_sku[sku]
        return ProductMatch(sku, r.title, r.return_category, r.warranty_months, r.assured,
                            r.price_inr, method, score, list(cands))

    # ---- 1. explicit SKU ----
    def _explicit(self, t: str):
        for pre, num in self._sku_rx.findall(t):
            sku = f"{pre.upper()}-{int(num):04d}"
            if sku in self.by_sku:
                return sku
        return None

    # ---- 3. spoken cue + number ----
    def _cue_number(self, t: str):
        for m in _CUE.finditer(t):
            toks = _TOK.findall(t[m.end(): m.end() + 60])[:6]
            n = _number_words(toks)
            if n in self.by_number and len(self.by_number[n]) == 1:
                return self.by_number[n][0]
        return None

    # ---- 2. title with typo / ASR tolerance ----
    def _title_scores(self, t: str):
        q = set(_TOK.findall(t))
        q |= {_SYN[w] for w in q if w in _SYN}
        q |= {str(_ONES[w]) for w in list(q) if w in _ONES}
        scores = {}
        for sku, toks in self.core.items():
            hit = sum(1 for w in toks if w in q or any(_similar(w, x) for x in q))
            scores[sku] = (hit / len(toks), toks[0] in q or any(_similar(toks[0], x) for x in q))
        return scores

    def _title(self, t: str):
        scores = self._title_scores(t)
        best = max(v[0] for v in scores.values())
        if best <= 0:
            return None
        top = [s for s, v in scores.items() if abs(v[0] - best) < 1e-9]
        brand_ok = any(scores[s][1] for s in top)
        other = max((v[0] for s, v in scores.items() if self.core[s] != self.core[top[0]]), default=0.0)
        if best >= 0.75 or (best >= 0.6 and brand_ok and best - other >= 0.15):
            return best, top
        return None

    def resolve(self, text: str) -> ProductMatch | None:
        t = (text or "").lower()
        sku = self._explicit(t)
        if sku:
            return self._mk(sku, "sku")
        found = self._title(t)
        cue = self._cue_number(t)
        if found:
            score, tops = found
            best = cue if cue in tops else max(tops, key=lambda s: len(self.core[s]))   # a spoken SKU settles colour/size variants
            cats = {self.by_sku[s].return_category for s in tops}
            m = self._mk(best, "title", score, [s for s in tops if s != best])
            if len(cats) > 1:
                m.return_category = ""            # ambiguous category: do not guess
            return m
        if cue:
            return self._mk(cue, "sku_cue", 0.8)
        return None
