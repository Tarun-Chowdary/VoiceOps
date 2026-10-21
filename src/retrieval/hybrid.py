"""Hybrid retrieval: BM25 (exact words, SKUs, numbers) + dense (meaning, Hindi) fused with
Reciprocal Rank Fusion:  score(d) = sum_over_rankers  w / (k + rank_of_d).
RRF needs no score calibration, which matters because BM25 and cosine scores are on
completely different scales.
"""
from dataclasses import dataclass
from src.retrieval.bm25 import BM25


def rrf(rank_lists: list[list[int]], weights: list[float] | None = None, k: int = 60) -> dict[int, float]:
    weights = weights or [1.0] * len(rank_lists)
    fused: dict[int, float] = {}
    for ranks, w in zip(rank_lists, weights):
        for pos, idx in enumerate(ranks, 1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + pos)
    return fused


@dataclass
class Hit:
    chunk: object
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class HybridRetriever:
    def __init__(self, chunks, use_dense: bool = True, w_bm25: float = 1.0, w_dense: float = 1.0):
        self.chunks = chunks
        self.bm = BM25([c.search_text for c in chunks])
        self.w = (w_bm25, w_dense)
        self.dense = None
        if use_dense:
            try:
                from src.retrieval.dense import DenseIndex
                self.dense = DenseIndex(chunks)
            except Exception as e:      # graceful degradation: never crash a batch run
                print(f"[retrieval] dense index unavailable ({type(e).__name__}: {e}); using BM25 only")

    @staticmethod
    def _ranking(scores, only_positive=False):
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [i for i in order if scores[i] > 0] if only_positive else order

    def search(self, query: str, k: int = 5, mode: str = "hybrid", dense_query: str | None = None) -> list[Hit]:
        bm_rank = self._ranking(self.bm.scores(query), only_positive=True)   # no noise from zero-score docs
        dn_rank = self._ranking(self.dense.scores(dense_query or query)) if self.dense else []
        if mode == "bm25" or not dn_rank:
            lists, ws = [bm_rank], [1.0]
        elif mode == "dense":
            lists, ws = [dn_rank], [1.0]
        else:
            lists, ws = [bm_rank, dn_rank], list(self.w)
        fused = rrf(lists, ws)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        bpos = {i: p for p, i in enumerate(bm_rank, 1)}
        dpos = {i: p for p, i in enumerate(dn_rank, 1)}
        return [Hit(self.chunks[i], fused[i], bpos.get(i), dpos.get(i)) for i in top]
