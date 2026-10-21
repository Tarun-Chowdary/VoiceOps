"""A small, dependency-free BM25 (Okapi) implementation.

Written by hand (not rank_bm25) because the corpus is tiny and so that every
step is explainable: term frequency saturation (k1), length normalisation (b),
and inverse document frequency.
"""
import math, re, unicodedata
from collections import Counter

_TOKEN = re.compile(r"[\w\u0900-\u097F]+", re.U)          # latin, digits, Devanagari
STOP = set("""a an the is are was were be been of to in on for and or if it its this that
with as at by from my me i you your can do does did will would so uh um haan hai kya
ka ki ke ko se mein me pe par hoon hu""".split())


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text).lower()
    return [w for w in _TOKEN.findall(t) if w not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(t) for t in self.toks) / max(self.N, 1)
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tf):
            dl, s = len(self.toks[i]), 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out
