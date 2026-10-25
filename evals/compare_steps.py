"""Step 1 vs Step 2 comparison, ready to paste into the report.
Usage: python evals/compare_steps.py            (reads evals/results/baseline_question.csv and rag_question.csv)
       python evals/compare_steps.py --input asr_text
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config

ap = argparse.ArgumentParser()
ap.add_argument("--input", default="question")
a = ap.parse_args()
res = config.ROOT / "evals" / "results"
b = pd.read_csv(res / f"baseline_{a.input}.csv")
r = pd.read_csv(res / f"rag_{a.input}.csv")
tf = lambda s: s.astype(str).str.lower().eq("true")
for d in (b, r):
    d["pass"] = tf(d["pass"])
    d["decision_ok"] = tf(d["decision_ok"])


def md(df):
    cols = [df.index.name or ""] + list(df.columns)
    rows = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        rows.append("| " + " | ".join([str(idx)] + [str(v) for v in r.values]) + " |")
    return "\n".join(rows)


def table(col):
    t = pd.DataFrame({"n": b.groupby(col).size(),
                      "Step 1": b.groupby(col)["pass"].sum(), "Step 2": r.groupby(col)["pass"].sum()})
    t["Step 1 %"] = (100 * t["Step 1"] / t["n"]).round(0).astype(int)
    t["Step 2 %"] = (100 * t["Step 2"] / t["n"]).round(0).astype(int)
    return t


lines = [f"# Step 1 vs Step 2 (input = {a.input}, n = {len(b)})", "",
         "| Metric | Step 1 prompt-only | Step 2 RAG |", "|---|---|---|",
         f"| Overall pass | {b['pass'].sum()}/{len(b)} = {b['pass'].mean():.1%} | {r['pass'].sum()}/{len(r)} = {r['pass'].mean():.1%} |",
         f"| Decision correct | {b['decision_ok'].mean():.1%} | {r['decision_ok'].mean():.1%} |"]
for col, name in (("retrieved_hit", "Retrieved sections hit gold"), ("retrieval_hit", "Used sections hit gold")):
    if col in r:
        v = tf(r[col].dropna().astype(str)) if r[col].notna().any() else pd.Series(dtype=bool)
        if len(v):
            lines.append(f"| {name} | n/a | {v.mean():.1%} (n={len(v)}) |")
for col in ("intent", "language", "answerable"):
    lines += ["", f"## By {col}", "", md(table(col))]
fixed = sorted(set(r.loc[r["pass"], "id"]) - set(b.loc[b["pass"], "id"]))
broke = sorted(set(b.loc[b["pass"], "id"]) - set(r.loc[r["pass"], "id"]))
lines += ["", f"Fixed by RAG: {len(fixed)} questions. Newly failing (regressions): {broke or 'none'}.",
          f"Still failing in Step 2: {sorted(r.loc[~r['pass'], 'id'])}"]
out = "\n".join(lines)
print(out)
(res / f"step1_vs_step2_{a.input}.md").write_text(out, encoding="utf-8")
print(f"\nSaved -> {res / f'step1_vs_step2_{a.input}.md'}")
