"""Retrieval ablation table: each row adds one component. No LLM calls, no quota.
Usage: python evals/run_ablation.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config
from src.agent.normalize import normalize_asr
from src.kb.loader import load_chunks
from src.retrieval.hybrid import HybridRetriever
from src.data.catalog import Catalog

chunks = load_chunks()
ret = HybridRetriever(chunks)
cat = Catalog()
q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
items = lambda s: [x.strip() for x in s.split(" ; ") if x.strip()]


def with_category(t: str) -> str:
    t = normalize_asr(t)
    m = cat.resolve(t)
    return t + (" " + m.return_category if m and m.return_category else "")


def run(name, fn, col, k=5, mode="bm25"):
    hits = rec = n = 0
    by = {}
    for r in q.itertuples():
        gold = [g for g in items(r.gold_evidence) if ":" not in g]
        if not gold:
            continue
        top = [h.chunk.section_id for h in ret.search(fn(getattr(r, col)), k=k, mode=mode)]
        ok = all(g in top for g in gold)
        hits += ok
        rec += sum(g in top for g in gold) / len(gold)
        n += 1
        by.setdefault(r.language, [0, 0])
        by[r.language][0] += ok
        by[r.language][1] += 1
    langs = "  ".join(f"{l}:{a}/{b}" for l, (a, b) in sorted(by.items()))
    print(f"{name:40s} | {hits / n:6.1%} | {rec / n:6.1%} | {langs}")


print(f"{'stage':40s} | all-gold@5 | recall@5 | by language (hit/total)")
run("1. BM25 on raw ASR text", lambda t: t, "asr_text")
run("2. + ASR normaliser", normalize_asr, "asr_text")
run("3. + product category from catalog", with_category, "asr_text")
if ret.dense is not None:
    run("4. dense only (e5) on the same query", with_category, "asr_text", mode="dense")
    run("5. HYBRID: BM25 + dense (RRF)", with_category, "asr_text", mode="hybrid")
    run("5b. hybrid, k=8", with_category, "asr_text", k=8, mode="hybrid")
else:
    print("(dense index unavailable: stages 4-5 skipped)")
print()
run("(reference) BM25 on clean text", lambda t: t, "question")
run("(reference) + product category", with_category, "question")
