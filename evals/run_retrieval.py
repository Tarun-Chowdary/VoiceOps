"""Measure retrieval quality against practice/questions.csv gold_evidence (no LLM needed).
Usage: python evals/run_retrieval.py --input question --k 5
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config
from src.kb.loader import load_chunks
from src.retrieval.bm25 import BM25

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="question", choices=["question", "asr_text"])
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()

chunks = load_chunks()
bm = BM25([c.search_text for c in chunks])
q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
items = lambda s: [x.strip() for x in s.split(" ; ") if x.strip()]

rows = []
for _, r in q.iterrows():
    gold = [g for g in items(r["gold_evidence"]) if ":" not in g]
    if not gold:
        continue
    sc = bm.scores(r[a.input])
    order = sorted(range(len(chunks)), key=lambda i: -sc[i])
    top = [chunks[i].section_id for i in order[:a.k]]
    neg = items(r["hard_negatives"])
    rows.append({"id": r["id"], "lang": r["language"], "gold": gold, "top": top,
                 "all_hit": all(g in top for g in gold),
                 "recall": sum(g in top for g in gold) / len(gold),
                 "top1_ok": top[0] in gold,
                 "neg_in_top": any(n in top for n in neg)})
d = pd.DataFrame(rows)
print(f"chunks indexed: {len(chunks)} | questions with KB gold: {len(d)} | k={a.k} | input={a.input}")
print(f"all-gold hit@{a.k}: {d.all_hit.mean():.1%}   avg recall@{a.k}: {d.recall.mean():.1%}   top1 in gold: {d.top1_ok.mean():.1%}")
print(f"hard-negative appears in top-{a.k}: {d.neg_in_top.mean():.1%}")
print("\nBy language (all-gold hit):\n", d.groupby('lang').all_hit.agg(['sum', 'count']))
print("\nMisses:")
for _, x in d[~d.all_hit].iterrows():
    print(f"  {x['id']} [{x['lang']}] gold={x['gold']} top={x['top']}")
