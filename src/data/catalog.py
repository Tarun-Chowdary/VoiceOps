"""Product resolver: find which catalog product a customer is talking about.

Matches (1) an explicit SKU like ELE-0012, else (2) the product title with typo
tolerance (works for Hindi/Hinglish questions that contain a Latin-script product name).
Returns category / warranty / Assured facts straight from data/products.csv, so those
facts never depend on the LLM remembering them.
"""
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import pandas as pd
from src import config

_SKU = re.compile(r"\b([a-z]{3})-?(\d{4})\b", re.I)
_TOK = re.compile(r"[a-z0-9]+")


@dataclass
class ProductMatch:
    sku: str
    title: str
    return_category: str
    warranty_months: str
    assured: str
    price_inr: str
    method: str                      # 'sku' | 'title'
    score: float = 1.0
    candidates: list = field(default_factory=list)   # other SKUs with equal score (e.g. colour variants)


class Catalog:
    def __init__(self):
        self.df = pd.read_csv(config.DATA_DIR / "products.csv", dtype=str).fillna("")
        self.by_sku = {r.sku: r for r in self.df.itertuples()}
        self.title_tokens = {r.sku: _TOK.findall(r.title.lower()) for r in self.df.itertuples()}

    def _mk(self, sku, method, score=1.0, cands=()):
        r = self.by_sku[sku]
        return ProductMatch(sku, r.title, r.return_category, r.warranty_months, r.assured,
                            r.price_inr, method, score, list(cands))

    def resolve(self, text: str) -> ProductMatch | None:
        t = text.lower()
        for pre, num in _SKU.findall(t):
            sku = f"{pre.upper()}-{num}"
            if sku in self.by_sku:
                return self._mk(sku, "sku")
        q = set(_TOK.findall(t))
        if not q:
            return None
        best, best_score = [], 0.0
        for sku, toks in self.title_tokens.items():
            hit = sum(1 for w in toks if w in q or any(
                len(w) > 3 and SequenceMatcher(None, w, x).ratio() >= 0.85 for x in q))
            score = hit / len(toks)
            if score > best_score + 1e-9:
                best, best_score = [sku], score
            elif abs(score - best_score) < 1e-9:
                best.append(sku)
        if best_score < 0.75:
            return None
        # prefer the most specific title when several tie on coverage
        top = max(best, key=lambda s: len(self.title_tokens[s]))
        cats = {self.by_sku[s].return_category for s in best}
        m = self._mk(top, "title", best_score, [s for s in best if s != top])
        if len(cats) > 1:
            m.return_category = ""   # ambiguous category: do not guess
        return m
