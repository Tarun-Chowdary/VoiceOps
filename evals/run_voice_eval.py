"""End-to-end VOICE evaluation on the practice questions, without needing any recordings.

For each practice question: Gnani TTS speaks the question (a different voice from the agent's) -> Gnani STT ->
the same agent -> reply scored with the same rules as the text evaluation. Reports speech-recognition quality
(WER, SKU recovery), the STT language the agent chose, and the end-to-end pass rate vs the text-only run.
NOTE: synthesized speech is cleaner than real calls. Also record yourself on a few questions and run run.py on them.

  python evals/run_voice_eval.py --limit 5 --sleep 8
  python evals/run_voice_eval.py --stt-lang auto --stt-format verbatim
  python evals/run_voice_eval.py --stt-lang en-IN --stt-format transcribe
"""
import argparse, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from src import config
from src.agent import llm
from src.agent.normalize import normalize_asr
from src.eval.scoring import score_row, retrieval_hit
from src.eval.wer import wer
from src.data.catalog import Catalog
from src.voice.gnani_io import GnaniSTT, GnaniTTS
from src.voice.pipeline import VoiceAgent

ap = argparse.ArgumentParser()
ap.add_argument("--stt-lang", default="auto", choices=["auto", "en-IN", "hi-IN"])
ap.add_argument("--stt-format", default="transcribe", choices=["verbatim", "transcribe"])
ap.add_argument("--voice", default="", help="override the voice that plays the CUSTOMER (default: a different voice from the agent)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ids", default="")
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--no-reply-audio", action="store_true", help="skip speaking the agent's replies (faster, fewer TTS calls)")
a = ap.parse_args()


class NullTTS:
    def synthesize(self, text, lang, out_path):
        Path(out_path).write_bytes(b"")
        return Path(out_path)


q = pd.read_csv(config.PRACTICE_DIR / "questions.csv", dtype=str).fillna("")
if a.ids:
    q = q[q["id"].isin([x.strip() for x in a.ids.split(",")])]
if a.limit:
    q = q.head(a.limit)
cache = config.ROOT / ".cache" / "voice_eval"
cache.mkdir(parents=True, exist_ok=True)
tag = f"{a.stt_lang}_{a.stt_format}"
outdir = config.ROOT / "evals" / "results" / f"voice_{tag}"
outdir.mkdir(parents=True, exist_ok=True)

customer = GnaniTTS(voice=a.voice or None, role="customer")
agent = VoiceAgent(stt=GnaniSTT(fmt=a.stt_format), tts=NullTTS() if a.no_reply_audio else GnaniTTS(), stt_lang=a.stt_lang)
CAT = Catalog()


def script(t):
    return "dev" if len(re.findall(r"[\u0900-\u097F]", t)) > len(re.findall(r"[A-Za-z]", t)) else "lat"


rows = []
for i, (_, r) in enumerate(q.iterrows(), 1):
    wav = cache / f"{r['id']}_{a.voice or 'customer'}.wav"
    if not wav.exists():
        try:
            customer.synthesize(r["question"], r["language"], wav)       # en / hi / hi-en -> en-IN / hi-IN / hi-en (or auto)
        except Exception as e:
            print(f"[{i}/{len(q)}] {r['id']} SKIPPED: could not synthesize the customer audio ({type(e).__name__}: {str(e)[:100]})", flush=True)
            continue
    t0 = time.time()
    rec = agent.turn(wav, outdir / f"{r['id']}.wav", kind="single")
    s = score_row(r, rec["reply"], rec["decision"])
    sku = re.search(r"sku=([A-Z]{3}-\d{4})", r["entities"])
    rows.append({"id": r["id"], "language": r["language"], "intent": r["intent"], "question": r["question"],
                 "heard": rec["transcript"], "stt_language": rec["stt_language"], "wer": wer(r["question"], rec["transcript"]) if script(r["question"]) == script(rec["transcript"]) else float("nan"),
                 "sku_ok": (lambda m: bool(m) and (m.sku == sku.group(1) or sku.group(1) in m.candidates))(CAT.resolve(normalize_asr(rec["transcript"]))) if sku else None,
                 "reply": rec["reply"], "decision": rec["decision"], "expected": r["expected_decision"],
                 "retrieval_hit": retrieval_hit(r, rec["used_sections"]), "audio_out": bool(rec["audio_out"]) or a.no_reply_audio,
                 "error": rec["error"], **s})
    print(f"[{i}/{len(q)}] {r['id']} {'PASS' if s['pass'] else 'FAIL'} wer={rows[-1]['wer']:.2f} [{rec['stt_language']}] heard: {rec['transcript'][:60]!r}", flush=True)
    if a.sleep and not llm.LAST_CACHED:
        time.sleep(a.sleep)

d = pd.DataFrame(rows)
path = config.ROOT / "evals" / "results" / f"voice_eval_{tag}.csv"
d.to_csv(path, index=False, encoding="utf-8-sig")
n = len(d)
print(f"\n== VOICE end-to-end | stt-lang={a.stt_lang} stt-format={a.stt_format} | customer voice={a.voice or 'default'} | n={n}")
print(f"End-to-end pass     : {int(d['pass'].sum())}/{n} = {d['pass'].mean():.1%}")
print(f"Spoken replies made : {int(d['audio_out'].sum())}/{n}      errors: {int((d['error'] != '').sum())}")
print(f"Mean WER            : {d['wer'].mean():.1%} (only rows where question and transcript use the same script; {int(d['wer'].isna().sum())} rows excluded)   by language: " + str({k: f"{v:.1%}" for k, v in d.groupby('language')['wer'].mean().items()}))
sk = d["sku_ok"].dropna()
if len(sk):
    print(f"Product recovered   : {int(sk.sum())}/{len(sk)}  (catalog match from the heard text vs the SKU in the question)")
print("STT language chosen by label:\n", pd.crosstab(d["language"], d["stt_language"]).to_string())
base = config.ROOT / "evals" / "results" / "rag_question.csv"
if base.exists():
    b = pd.read_csv(base)
    b["pass"] = b["pass"].astype(str).str.lower().eq("true")
    m = d.merge(b[["id", "pass"]].rename(columns={"pass": "text_pass"}), on="id")
    print(f"\nText-only RAG on the same rows: {int(m['text_pass'].sum())}/{len(m)}  -> voice lost {int((m['text_pass'] & ~m['pass']).sum())}, gained {int((~m['text_pass'] & m['pass']).sum())}")
print("\nBy language (pass):\n", d.groupby("language")["pass"].agg(["sum", "count"]).to_string())
print("\nFailures:")
for _, x in d[~d["pass"]].iterrows():
    print(f"  {x['id']} [{x['language']}] wer={x['wer']:.2f} missing={x['missing']}\n     SAID : {x['question'][:90]}\n     HEARD: {x['heard'][:90]}\n     BOT  : {x['reply'][:150]}")
print(f"\nSaved -> {path}")
