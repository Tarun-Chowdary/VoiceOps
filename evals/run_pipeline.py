"""End-to-end retrieval evaluation of the full pipeline.
  python evals/run_pipeline.py --intent none            # hybrid only (no LLM)
  python evals/run_pipeline.py --intent llm --sleep 4   # uses 1 LLM call per question (cached)
  python evals/run_pipeline.py --intent oracle          # practice labels: plumbing/upper bound ONLY (circular!)
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config
from src.kb.loader import load_chunks
from src.data.catalog import Catalog
from src.agent.normalize import normalize_asr
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.pipeline import retrieve

ap = argparse.ArgumentParser()
ap.add_argument("--intent", default="none", choices=["none", "llm", "oracle"])
ap.add_argument("--input", default="asr_text", choices=["question", "asr_text"])
ap.add_argument("--max_total", type=int, default=8)
ap.add_argument("--no-dense", action="store_true")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.0)
a = ap.parse_args()

chunks = load_chunks()
ret = HybridRetriever(chunks, use_dense=not a.no_dense)
cat = Catalog()
q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
if a.limit:
    q = q.head(a.limit)
items = lambda s: [x.strip() for x in s.split(" ; ") if x.strip()]

if a.intent == "llm":
    from src.agent import llm
    from src.agent.understand import understand
if a.intent == "oracle":
    from src.agent.understand import Understanding

rows, correct_intent = [], []
for r in q.itertuples():
    text = getattr(r, a.input)
    und = None
    if a.intent == "llm":
        und = understand(normalize_asr(text))
        correct_intent.append((r.id, r.intent, und.intent))
        if a.sleep and not llm.LAST_CACHED:
            time.sleep(a.sleep)
    elif a.intent == "oracle":
        und = Understanding(intent=r.intent, english_query=normalize_asr(text))
    got, _ = retrieve(ret, cat, text, und, max_total=a.max_total)
    ids = [g.chunk.section_id for g in got]
    gold = [g for g in items(r.gold_evidence) if ":" not in g]
    if not gold:
        continue
    top5 = ids[:5]
    rows.append({"id": r.id, "lang": r.language, "gold": gold, "ids": ids,
                 "hit": all(g in ids for g in gold), "hit5": all(g in top5 for g in gold),
                 "recall": sum(g in ids for g in gold) / len(gold),
                 "prec": sum(i in gold for i in ids) / max(len(ids), 1), "n": len(ids)})
d = pd.DataFrame(rows)
print(f"intent={a.intent} input={a.input} dense={'off' if a.no_dense else 'on'} | questions with KB gold: {len(d)}")
print(f"all-gold in final context (<= {a.max_total} chunks): {d.hit.mean():.1%} | in first 5: {d.hit5.mean():.1%}")
print(f"avg recall: {d.recall.mean():.1%} | avg precision: {d.prec.mean():.1%} | avg chunks: {d.n.mean():.1f}")
print("by language:", {l: f"{int(g.hit.sum())}/{len(g)}" for l, g in d.groupby('lang')})
if correct_intent:
    ok = sum(1 for _, t, p in correct_intent if t == p)
    print(f"\nLLM intent accuracy vs practice labels: {ok}/{len(correct_intent)} = {ok / len(correct_intent):.1%}")
    for i, t, p in correct_intent:
        if t != p:
            print(f"  {i}: labelled {t:18s} predicted {p}")
print("\nMisses:")
for _, x in d[~d.hit].iterrows():
    print(f"  {x['id']} [{x['lang']}] gold={x['gold']} got={x['ids']}")
