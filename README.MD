# VoiceOps: a voice-first customer-support agent for ZentroMart

A support agent people can **call**: it answers from company documents (RAG), holds a verified multi-turn
conversation with real order lookups and human handover, and speaks with **Gnani STT/TTS** in English, Hindi and Hinglish.
Built for the VoiceOps 2026 capstone (IIT Guwahati C&A × Gnani.ai).

```
customer audio ─► Gnani STT ─► ASR normaliser ─► understand (1 LLM call) ─► hybrid retrieval ─► verified facts ─► grounded answer ─► Gnani TTS ─► audio
                  (en-IN/hi-IN)  (digits, SKUs)   intent, entities,        BM25 + e5 + intent     (calendar,        (cites sections,      (speech-
                                                  English query            pins + precedence      order record)      abstains)             friendly)
                                         protocol turns (verification, lockout) and ESC-1 handovers are CODE, not LLM
```

## What makes it trustworthy
* **Answers only from the documents.** Policy sections are retrieved by section ID, ranked by the KB's own precedence
  (policy > guide > help > FAQ > marketing); superseded documents apply only where the newer policy says so.
* **The LLM never does arithmetic or table lookups.** Business-day deadlines (Mon-Sat minus holidays), return windows,
  the pickup fee and the 1 Aug 2026 electronics rule are computed by code from numbers *parsed from the KB text*.
* **Safety in code.** Identity check before any order detail (phone digits or pincode, one retry, then lockout);
  phone/pincode/customer ID never reach the LLM; every ESC-1 handover trigger is detected in code; a ticket is
  claimed **only if `create_ticket` returned `created=True`**; sentences claiming an action that was not done are removed.
* **Never silent.** Unclear audio gets a spoken "please repeat" (and does not burn a verification attempt); an internal
  error gets a spoken apology.

## Results (practice set; see the report for the caveats on tuning)
| Stage | Result |
|---|---|
| Step 1, prompt-only LLM | 12/52 = 23.1% |
| Step 2, RAG + calculator | 50/52 = 96.2% |
| Step 3, conversations (23 convs, 96 turns) | 95/96 turns = 99.0%; 22/23 conversations fully correct; tool calls 100% |
| Retrieval ablation (all gold sections found) | BM25 on ASR 23.4% → normaliser 36.2% → +catalog 44.7% → +embeddings (hybrid) 63.8% → +intent routing 95.7% |
| LLM intent accuracy | 50/52 = 96.2% |
| Voice end-to-end (synthesized audio) | *fill in from `python evals/run_voice_eval.py`* |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                                       # then fill in GNANI_API_KEY and LLM_API_KEY
```
* Get a Gnani key at app.gnani.ai/voice (one key covers STT and TTS). Use any OpenAI-compatible LLM endpoint
  (default: Gemini through its OpenAI-compatible URL).
* `DATASET_DIR` must point to the extracted organizer dataset (it is not copied into this repo).
* First run downloads a ~0.5 GB multilingual embedding model (`intfloat/multilingual-e5-small`). If it cannot be
  loaded the agent degrades to BM25-only retrieval instead of crashing.

## Voice notes (measured on real Gnani output)
* **STT:** `format="transcribe"` (inverse text normalization) is the default: spoken numbers and dates arrive as digits
  (`18599253`, `17th October 2026`) instead of words; Hindi speech needs `hi-IN` because `en-IN` returns Hindi number words.
  The agent transcribes with both `en-IN` and `hi-IN` and keeps the more plausible transcript (default `en-IN`).
* **TTS:** `timbre-v2.5` by default (accepted by every SDK version; `timbre-v3.0` exists only in newer ones). Hinglish uses
  language `hi-en` where the SDK accepts it and `auto` otherwise. Unsupported options are detected, remembered and skipped.
* Network errors and rate limits are retried with backoff; a turn always ends with something spoken.

## Run (batch, what the organizers use)
```bash
python run.py --audio_dir ./customer_audio --out_dir ./out
```
For every audio file (wav/mp3/ogg/flac/aac/m4a) it saves `<name>.wav` (spoken reply), `<name>.txt` (reply text) and
`<name>.json` (transcript, decision, **retrieved and used document chunks with their text**, tool calls, timings), plus
`results.csv` / `results.jsonl`. Files are independent questions by default; files named `<conv>_t1.wav, <conv>_t2.wav, ...`
are replayed as one conversation (use `--mode conversation` or `--manifest manifest.csv` for other naming).
Add `--sleep 8` on a free LLM tier. Exit code 0 means every file got a spoken reply.

## Evaluate and test
```bash
python -m pytest tests -q                                   # 50+ offline tests (fake LLM / STT / TTS, no API calls)
python evals/run_questions.py --agent baseline --sleep 8    # Step 1
python evals/run_questions.py --agent rag --sleep 8         # Step 2
python evals/compare_steps.py                               # Step 1 vs Step 2 report table
python evals/run_conversations.py --sleep 8                 # Step 3 replay of the practice conversations
python evals/run_ablation.py                                # retrieval ablation (no LLM)
python scripts/voice_smoke.py                               # checks your Gnani setup (needs a key)
python evals/run_voice_eval.py --sleep 8                    # end-to-end voice evaluation (needs a key)
```

## Layout
`src/kb` chunking by section ID · `src/retrieval` BM25, dense, RRF fusion, intent pins, precedence · `src/policy` calendar and
KB-parsed rules · `src/agent` normaliser, understanding, RAG, conversation state machine, guard · `src/voice` Gnani I/O,
speech rewrite, language choice, voice turn · `run.py` batch runner · `evals/` measurement scripts · `tests/` unit tests.

## Honest limitations
* Practice-set scores are optimistic: the intent→section routing table and several prompt rules were refined after error analysis on it.
* The en-IN vs hi-IN choice is a heuristic (the REST API needs an explicit language); Hindi and Hinglish speech are acoustically similar, so Hindi callers usually get Hinglish (Latin-script) replies.
* Only synthesized audio was used for automated voice evaluation; real noisy calls will be harder.
