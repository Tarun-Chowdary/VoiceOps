"""Run practice/questions.csv through an agent and score it.
Usage:  python evals/run_questions.py --agent baseline --input question
        python evals/run_questions.py --agent baseline --input asr_text
"""
import argparse, importlib, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config
from src.eval.scoring import score_row, retrieval_hit

ap = argparse.ArgumentParser()
ap.add_argument("--agent", default="baseline")          # module name in src/agent/
ap.add_argument("--input", default="question", choices=["question", "asr_text"])
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.0)     # seconds between calls (free-tier RPM)
a = ap.parse_args()

agent = importlib.import_module(f"src.agent.{a.agent}")
q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
if a.limit:
    q = q.head(a.limit)

rows = []
for _, r in q.iterrows():
    out = agent.answer(r[a.input])
    s = score_row(r, out["reply"], out["decision"])
    hit = retrieval_hit(r, out.get("used_sections", []))
    rows.append({"id": r["id"], "intent": r["intent"], "language": r["language"],
                 "answerable": r["answerable"], "input": r[a.input],
                 "decision": out["decision"], "expected": r["expected_decision"],
                 "reply": out["reply"], "used_sections": " ; ".join(out.get("used_sections", [])),
                 "retrieval_hit": hit, **s})
    if a.sleep:
        time.sleep(a.sleep)

df = pd.DataFrame(rows)
res = config.ROOT / "evals" / "results"
res.mkdir(parents=True, exist_ok=True)
path = res / f"{a.agent}_{a.input}.csv"
df.to_csv(path, index=False, encoding="utf-8-sig")

n = len(df)
print(f"\n== {a.agent} | input={a.input} | n={n}")
print(f"Overall pass      : {df['pass'].sum()}/{n} = {df['pass'].mean():.1%}")
print(f"Decision correct  : {df['decision_ok'].mean():.1%}")
print(f"must_include ok   : {df['include_ok'].mean():.1%}")
print(f"must_not ok       : {df['exclude_ok'].mean():.1%}")
rh = df["retrieval_hit"].dropna()
if len(rh):
    print(f"Retrieval hit     : {rh.astype(bool).mean():.1%} (n={len(rh)})")
print("\nBy answerable:\n", df.groupby("answerable")["pass"].agg(["sum", "count"]))
print("\nBy language:\n", df.groupby("language")["pass"].agg(["sum", "count"]))
print("\nBy intent:\n", df.groupby("intent")["pass"].agg(["sum", "count"]))
print(f"\nSaved -> {path}")
