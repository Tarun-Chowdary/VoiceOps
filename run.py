"""Batch runner used by the organizers:  python run.py --audio_dir <folder of customer audio> --out_dir <folder>

For every audio file: Gnani STT -> agent -> Gnani TTS. Saves per file: <stem>.json (transcript, reply, decision,
retrieved + used document chunks, tool calls, timings), <stem>.txt (reply text) and <stem>.wav (reply audio),
plus results.jsonl / results.csv for the whole folder.

Files are independent single questions by default. Files named like  <conversation>_t1.wav, <conversation>_t2.wav
(or ..._turn2) are replayed in order as ONE conversation (identity check, order lookup, handover ...).
Use --mode conversation to group  <name>_<number>  files, or --manifest manifest.csv (columns: file,conversation_id,turn).
"""
import argparse, csv, json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.voice.gnani_io import AUDIO_EXT

TURN_MARK = re.compile(r"^(?P<conv>.+?)[_\-\s](?:t|turn)[_\-\s]?(?P<turn>\d{1,3})$", re.I)
NUM_SUFFIX = re.compile(r"^(?P<conv>.+?)[_\-\s](?P<turn>\d{1,3})$")
CONV_WORDS = re.compile(r"(conv|conversation|call|dialog|dialogue|session|^pc\d)", re.I)


def plan(files: list[Path], mode: str = "auto", manifest: Path | None = None):
    """Returns [("single"|"conversation", conversation_id, [(turn, path), ...]), ...]"""
    jobs, groups = [], {}
    if manifest:
        by_name = {f.name: f for f in files}
        for r in csv.DictReader(open(manifest, encoding="utf-8-sig")):
            f = by_name.get(r["file"])
            if f:
                groups.setdefault(r["conversation_id"], []).append((int(r.get("turn") or 1), f))
        return [("conversation", cid, sorted(items)) for cid, items in groups.items()]
    singles = []
    for f in files:
        m = TURN_MARK.match(f.stem)
        if not m and mode == "conversation":
            m = NUM_SUFFIX.match(f.stem)
        if not m and mode == "auto":
            n = NUM_SUFFIX.match(f.stem)
            m = n if n and CONV_WORDS.search(n.group("conv")) else None
        if m and mode != "single":
            groups.setdefault(m.group("conv"), []).append((int(m.group("turn")), f))
        else:
            singles.append(f)
    for cid, items in groups.items():
        if len(items) >= 2 or mode == "conversation":
            jobs.append(("conversation", cid, sorted(items)))
        else:
            singles.append(items[0][1])
    jobs += [("single", f.stem, [(1, f)]) for f in sorted(singles)]
    return sorted(jobs, key=lambda j: j[1])


def main(argv=None, stt=None, tts=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--mode", default="auto", choices=["auto", "single", "conversation"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--stt-lang", default="auto", choices=["auto", "en-IN", "hi-IN"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait after each uncached LLM call (free-tier limits)")
    a = ap.parse_args(argv)

    from src.voice.pipeline import VoiceAgent
    files = sorted(p for p in Path(a.audio_dir).rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        print(f"No audio files found in {a.audio_dir} (accepted: {sorted(AUDIO_EXT)})")
        return 1
    jobs = plan(files, a.mode, Path(a.manifest) if a.manifest else None)
    if a.limit:
        jobs = jobs[:a.limit]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agent = VoiceAgent(stt=stt, tts=tts, stt_lang=a.stt_lang)
    rows, jl = [], open(out / "results.jsonl", "w", encoding="utf-8")
    n = sum(len(j[2]) for j in jobs)
    done = 0
    for kind, cid, items in jobs:
        agent.new_conversation()
        for turn, f in items:
            done += 1
            t0 = time.time()
            stem = f.stem
            rec = agent.turn(f, out / f"{stem}.wav", kind=kind)
            rec.update(file=f.name, conversation_id=cid if kind == "conversation" else "", turn=turn if kind == "conversation" else "",
                       mode=kind, total_seconds=round(time.time() - t0, 2))
            (out / f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            (out / f"{stem}.txt").write_text(rec["reply"], encoding="utf-8")
            jl.write(json.dumps(rec, ensure_ascii=False) + "\n"); jl.flush()
            rows.append(rec)
            print(f"[{done}/{n}] {f.name} -> {rec['decision'] or '-'} | heard: {rec['transcript'][:60]!r}"
                  + (f" | ERROR: {rec['error'][:80]}" if rec["error"] else ""), flush=True)
            if a.sleep:
                try:
                    from src.agent import llm
                    if not llm.LAST_CACHED and rec["intent"] not in ("unclear_audio", "protocol"):
                        time.sleep(a.sleep)
                except Exception:
                    pass
    jl.close()
    cols = ["file", "conversation_id", "turn", "mode", "transcript", "stt_language", "reply", "decision", "intent", "language",
            "used_sections", "retrieved_sections", "tool_calls", "ticket_id", "audio_out", "error", "stt_seconds", "agent_seconds", "tts_seconds", "total_seconds"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: " ; ".join(r[k]) for k in ("used_sections", "retrieved_sections", "tool_calls")}})
    ok = sum(1 for r in rows if r["audio_out"])
    print(f"\nProcessed {len(rows)} files | spoken replies produced: {ok}/{len(rows)} | errors: {sum(1 for r in rows if r['error'])} | output: {out.resolve()}")
    return 0 if ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
