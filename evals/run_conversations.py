"""Replay practice/conversations.csv through the Conversation agent and score every turn.
Checks per turn: decision, required / forbidden phrases, tool calls (expected ones present, extras counted), evidence.
Usage: python evals/run_conversations.py [--input customer_message] [--limit N] [--sleep 8] [--ids PC003,PC007]
"""
import argparse, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config
from src.agent import llm
from src.agent.conversation import Conversation
from src.eval.scoring import check_must_include, check_must_not_include, decision_ok, retrieval_hit

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ids", default="")
ap.add_argument("--sleep", type=float, default=0.0)
a = ap.parse_args()

df = pd.read_csv(config.PRACTICE_DIR / "conversations.csv", dtype=str).fillna("")
if a.ids:
    df = df[df["conversation_id"].isin([x.strip() for x in a.ids.split(",")])]
ids = list(dict.fromkeys(df["conversation_id"]))
if a.limit:
    ids = ids[:a.limit]
TOOL = re.compile(r"(\w+)\(([^)]*)\)")


def parse_calls(s):
    out = []
    for name, args in TOOL.findall(s or ""):
        out.append((name, dict(x.strip().split("=", 1) for x in args.split(",") if "=" in x)))
    return out


def tools_ok(expected, actual):
    """Every expected call must be present (create_ticket args are free text, so only the name is compared)."""
    act = parse_calls(" ".join(actual))
    missing = []
    for name, args in expected:
        hit = any(n == name and (name == "create_ticket" or a_ == args) for n, a_ in act)
        if not hit:
            missing.append((name, args))
    extra = [n for n, _ in act if n not in {e[0] for e in expected}]
    return not missing, missing, extra


rows = []
for cid in ids:
    conv, g = Conversation(), df[df["conversation_id"] == cid].sort_values("turn", key=lambda s: s.astype(int))
    for r in g.itertuples():
        out = conv.respond(r.customer_message)
        exp_calls = parse_calls(r.expected_tool_calls)
        t_ok, missing, extra = tools_ok(exp_calls, out["tool_calls"])
        inc, miss = check_must_include(out["reply"], r.must_include)
        exc, hits = check_must_not_include(out["reply"], r.must_not_include)
        dec = decision_ok(out["decision"], r.expected_decision, getattr(r, "acceptable_decisions", ""))
        ev = retrieval_hit({"gold_evidence": r.gold_evidence}, out.get("used_sections", [])) if hasattr(r, "gold_evidence") else None
        ok = dec and inc and exc and t_ok
        rows.append({"conversation_id": cid, "turn": r.turn, "scenario": r.scenario, "customer": r.customer_message,
                     "reply": out["reply"], "decision": out["decision"], "expected": r.expected_decision,
                     "tools": " ; ".join(out["tool_calls"]), "expected_tools": r.expected_tool_calls,
                     "decision_ok": dec, "include_ok": inc, "exclude_ok": exc, "tools_ok": t_ok, "extra_calls": " ".join(extra),
                     "evidence_hit": ev, "pass": ok, "missing_phrases": miss, "forbidden_hits": hits, "missing_tools": missing})
        print(f"{cid} T{r.turn} {'PASS' if ok else 'FAIL'}  [{out['decision']}/{r.expected_decision}] {r.scenario}", flush=True)
        if a.sleep and not llm.LAST_CACHED:
            time.sleep(a.sleep)

d = pd.DataFrame(rows)
res = config.ROOT / "evals" / "results"
res.mkdir(parents=True, exist_ok=True)
d.to_csv(res / "conversations.csv", index=False, encoding="utf-8-sig")
conv_ok = d.groupby("conversation_id")["pass"].all()
print(f"\nTurns passed        : {int(d['pass'].sum())}/{len(d)} = {d['pass'].mean():.1%}")
print(f"Conversations passed: {int(conv_ok.sum())}/{len(conv_ok)} (every turn correct)")
print(f"Decision correct    : {d['decision_ok'].mean():.1%}   Phrases ok: {(d['include_ok'] & d['exclude_ok']).mean():.1%}   Tool calls ok: {d['tools_ok'].mean():.1%}")
ev = d["evidence_hit"].dropna()
print(f"Evidence (used sections) hit: {ev.astype(bool).mean():.1%} (n={len(ev)})")
print(f"Turns with unexpected extra tool calls: {(d['extra_calls'] != '').sum()}")
print("\nBy scenario (turns passed):\n", d.groupby("scenario")["pass"].agg(["sum", "count"]).to_string())
print("\nFailures:")
for _, x in d[~d["pass"]].iterrows():
    print(f"  {x['conversation_id']} T{x['turn']} [{x['scenario']}] decision={x['decision']}/{x['expected']} "
          f"missing={x['missing_phrases']} forbidden={x['forbidden_hits']} tools_missing={x['missing_tools']}\n     USER: {x['customer'][:90]}\n     BOT : {x['reply'][:200]}")
print(f"\nSaved -> {res / 'conversations.csv'}")
