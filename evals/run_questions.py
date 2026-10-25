"""Run practice/questions.csv through an agent and score it.
Usage:  python evals/run_questions.py --agent baseline --input question --sleep 4
        python evals/run_questions.py --agent rag --input question --sleep 4
"""
import argparse, importlib, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.agent import llm
from src.eval.scoring import score_row, retrieval_hit

ap = argparse.ArgumentParser()
ap.add_argument("--agent", default="baseline")          # module name in src/agent/
ap.add_argument("--input", default="question", choices=["question", "asr_text"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ids", default="")                    # e.g. PQ0005,PQ0032 (targeted rerun)
ap.add_argument("--sleep", type=float, default=0.0)     # seconds between calls (free-tier RPM)
a = ap.parse_args()

agent = importlib.import_module(f"src.agent.{a.agent}")
q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
if a.ids:
    q = q[q["id"].isin([x.strip() for x in a.ids.split(",")])]
if a.limit:
    q = q.head(a.limit)
res = config.ROOT / "evals" / "results"
res.mkdir(parents=True, exist_ok=True)

rows = []
for i, (_, r) in enumerate(q.iterrows(), 1):
    t0 = time.time()
    out = agent.answer(r[a.input])
    s = score_row(r, out["reply"], out["decision"])
    rows.append({"id": r["id"], "intent": r["intent"], "language": r["language"],
                 "answerable": r["answerable"], "input": r[a.input],
                 "decision": out["decision"], "expected": r["expected_decision"],
                 "reply": out["reply"], "used_sections": " ; ".join(out.get("used_sections", [])),
                 "retrieved": " ; ".join(out.get("retrieved", [])),
                 "pred_intent": out.get("intent", ""),
                 "retrieval_hit": retrieval_hit(r, out.get("used_sections", [])),
                 "retrieved_hit": retrieval_hit(r, out.get("retrieved", [])), **s})
    print(f"[{i}/{len(q)}] {r['id']} {'PASS' if s['pass'] else 'FAIL'} "
          f"({time.time() - t0:.1f}s{', cached' if llm.LAST_CACHED else ''})", flush=True)
    if a.sleep and not llm.LAST_CACHED:
        time.sleep(a.sleep)
    pd.DataFrame(rows).to_csv(res / f"{a.agent}_{a.input}_partial.csv", index=False, encoding="utf-8-sig")

df = pd.DataFrame(rows)
path = res / f"{a.agent}_{a.input}.csv"
df.to_csv(path, index=False, encoding="utf-8-sig")
n = len(df)
print(f"\n== {a.agent} | input={a.input} | model={llm.MODEL} | n={n}")
print(f"Overall pass      : {df['pass'].sum()}/{n} = {df['pass'].mean():.1%}")
print(f"Decision correct  : {df['decision_ok'].mean():.1%}")
print(f"must_include ok   : {df['include_ok'].mean():.1%}")
print(f"must_not ok       : {df['exclude_ok'].mean():.1%}")
for col, name in (("retrieval_hit", "Cited sections hit "), ("retrieved_hit", "Retrieved hit      ")):
    v = df[col].dropna()
    if len(v):
        print(f"{name}: {v.astype(bool).mean():.1%} (n={len(v)})")
print("\nBy answerable:\n", df.groupby("answerable")["pass"].agg(["sum", "count"]))
print("\nBy language:\n", df.groupby("language")["pass"].agg(["sum", "count"]))
print("\nBy intent:\n", df.groupby("intent")["pass"].agg(["sum", "count"]))
print(f"\nSaved -> {path}")
