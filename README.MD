# 🎙️ VoiceOps: a voice-first customer-support agent for ZentroMart

**Customers call. The agent listens, answers from the company's own documents, verifies who it is talking to before it
touches an order, and hands over to a human when policy says a person must decide.**
It speaks English, Hindi and Hinglish, and every speech step uses **Gnani STT/TTS**.
Built for the VoiceOps 2026 capstone (IIT Guwahati Consulting & Analytics Club × Gnani.ai).

| | |
|---|---|
| **Grounded** | Answers only from the policy documents; says "I don't have that information" instead of inventing a policy, date or amount |
| **Safe by construction** | Identity check, escalation rules and "ticket created" are enforced in code, not left to the LLM |
| **Correct numbers** | Business-day deadlines, return windows and fees are *calculated by code from rules parsed out of the documents* |
| **Voice-robust** | Survives speech-recognition noise: spoken digits, garbled SKUs, misheard product names, glued words |
| **Never silent** | Unclear audio, an LLM error or a TTS failure still ends in a spoken reply |

## Try it
```bash
pip install -r requirements-demo.txt
python app.py            # opens http://127.0.0.1:7860, allow the microphone
```
Speak, upload a clip, or type. The **trust panel** beside the conversation shows what a buyer of this product would want to audit:
the decision and intent, the **document sections the answer relied on**, the **facts the code calculated**, every **tool call**,
and whether the caller is **verified**. A ready-made 2-minute demo script is in the app's "Suggested demo script" accordion.

## How it works
```
 caller audio ─► Gnani STT ─► ASR normaliser ─► understand ─► hybrid retrieval ─► verified facts ─► grounded answer ─► Gnani TTS ─► audio
                 en-IN/hi-IN   digits, SKUs,     1 LLM call:   BM25 + e5 dense     calendar, fees,   cites sections,      speech-
                 (picks best)  glued words       intent,       + intent routing    order record      abstains when        friendly
                                                 entities      + KB precedence                         documents are silent
                       protocol turns (greeting, identity check, lockout) and ESC-1 handovers are CODE, not LLM
```
1. **Listen.** `en-IN` transcript first; `hi-IN` is requested too only if the first looks wrong (fast path saves a request). `format="transcribe"` returns numbers and dates as digits.
2. **Normalise.** Spoken digit runs ("one eight five nine…", "सात नौ तीन…") become digits; "up i" → UPI; glued words ("deliveryat") are split; a garbled spoken SKU is recovered by number, because SKU numbers are unique.
3. **Understand** (one LLM call): intent, English search query, payment method, dated events, whether the question needs a verified order.
4. **Retrieve.** Section-level chunks (`[RET-2]`…) → BM25 + multilingual embeddings fused with Reciprocal Rank Fusion → intent-pinned sections → ordered by the KB's own precedence (policy > guide > help > FAQ > marketing; superseded documents last).
5. **Calculate.** Dates, windows and fees are computed in code from numbers parsed from the KB text (so the agent follows the documents if they change) and from the verified order.
6. **Answer.** The LLM sees only the retrieved sections, the verified facts and (when verified) a sanitised order record, and cites what it used.
7. **Guard.** Internal section IDs are stripped, sentences claiming an action that was not performed are removed, and a ticket is claimed only if `create_ticket` returned `created=True`.
8. **Speak.** A light rewrite for the ear (spelled-out SKUs, grouped IDs) on top of Gnani's own text normalisation.

### Conversation (Step 3)
Protocol turns are a deterministic state machine with fixed English/Hindi/Hinglish templates: ask for the order ID, call `lookup_order` (no data revealed), ask for the registered phone's last 4 digits **or** the delivery pincode, allow one retry, then lock out. The caller's phone, pincode and customer ID never reach the LLM. A question asked before verification is remembered and answered afterwards. ESC-1 triggers (human requested, legal/safety, unrecognised activity, repeat contact, high-value damage, refund more than 3 business days overdue) are detected in code and create a ticket with a short summary.

### Two product features beyond the brief
* **Trust panel** (`app.py`): live citations, calculated facts, tool trace and verification state for every turn.
* **One-command evaluation and regression harness** (`evals/`): prompt-only vs RAG comparison, retrieval ablation, conversation replay and end-to-end voice evaluation, each reproducible and scored with the same rules.

## Results (practice data; read the caveats below)
| Measure | Result |
|---|---|
| Step 1, prompt-only LLM (52 questions) | 12/52 = **23.1%** (17.3% on ASR-style text) |
| Step 2, RAG + calculator (same 52) | **52/52** on the latest run (49/52 on the first run, before prompt fixes) |
| Retrieval, all gold sections found | BM25 on ASR text 23.4% → +normaliser 36.2% → +catalog 44.7% → +embeddings 63.8% → +LLM intent routing **95.7%** |
| LLM intent accuracy vs practice labels | 50/52 = 96.2% |
| Step 3, 23 conversations / 96 turns | **95/96 turns = 99.0%**, 22/23 conversations fully correct, tool calls 100% |
| Step 4, voice end-to-end (Gnani TTS → STT → agent → TTS) | **50/52 = 96.2%**, 52/52 spoken replies, 0 errors, English WER 6.9% |

Regenerate everything with the commands below; `evals/results/` holds the raw per-question files.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt                          # + requirements-demo.txt for the app, requirements-dev.txt for tests
cp .env.example .env                                     # fill in GNANI_API_KEY and LLM_API_KEY
```
| Variable | Meaning |
|---|---|
| `GNANI_API_KEY` | One key covers STT and TTS (app.gnani.ai/voice). Never commit it. |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Any OpenAI-compatible endpoint. Default setup: Gemini through its OpenAI-compatible URL, `gemini-3.5-flash-lite`. |
| `DATASET_DIR` | The extracted organizer dataset (not copied into this repo). |
| `GNANI_STT_FORMAT` (`transcribe`), `GNANI_TTS_MODEL` (`timbre-v2.5`), `GNANI_TTS_VOICE_*`, `GNANI_TTS_LANG_HINGLISH` | Optional voice settings, see `.env.example`. |

The first run downloads a ~0.5 GB multilingual embedding model (`intfloat/multilingual-e5-small`). If it cannot be loaded the agent degrades to BM25-only retrieval instead of crashing.
Run `pip freeze > requirements.lock.txt` to record exact versions of your environment.

## Run: the batch command
```bash
python run.py --audio_dir ./customer_audio --out_dir ./out
```
For every audio file (`wav mp3 ogg flac aac m4a`) it saves, in `out/`:
* `<name>.wav`: the spoken reply · `<name>.txt`: the reply text
* `<name>.json`: transcript, decision, intent, **retrieved and used document chunks (with their text)**, calculated facts, tool calls, ticket ID, timings, any error
* `results.csv` / `results.jsonl` for the whole folder

Files are independent questions by default. Files named `<conversation>_t1.wav`, `<conversation>_t2.wav`, … (or `…_turn2`) are replayed in order as **one conversation**; use `--mode conversation` to group `<name>_<number>` files, or `--manifest manifest.csv` (columns `file,conversation_id,turn`). `--sleep 8` throttles LLM calls on a free tier. Exit code 0 means every file got a spoken reply.
`python scripts/make_sample_outputs.py` generates `sample_audio/` and `sample_outputs/` by running this command on a few spoken examples.

## Evaluate and test
```bash
python -m pytest tests -q                                  # 118 offline tests (fake LLM / STT / TTS: no keys, no cost)
python evals/run_questions.py --agent baseline --sleep 8   # Step 1
python evals/run_questions.py --agent rag --sleep 8        # Step 2
python evals/compare_steps.py                              # Step 1 vs Step 2 table for the report
python evals/run_ablation.py                               # retrieval ablation (no LLM)
python evals/run_pipeline.py --intent llm --sleep 8        # retrieval with LLM intent routing + intent accuracy
python evals/run_conversations.py --sleep 8                # Step 3: replay the practice conversations
python scripts/voice_smoke.py                              # checks your Gnani setup (needs a key)
python evals/run_voice_eval.py --sleep 8                   # Step 4: end-to-end voice evaluation (needs a key)
```

## Rules compliance
* **Speech:** only Gnani STT/TTS. No other speech service, not even as a fallback.
* **No hardcoded answers.** Replies are generated from retrieved document text plus code-calculated facts. Calculation rules (windows, fees, timelines, cut-off dates) are *parsed from the documents*, and test answers come from the agent, never typed or edited by hand.
* **Data:** only the synthetic dataset; the caller's phone, pincode and customer ID are never sent to the LLM.
* **Secrets:** `.env`, API keys, virtual environments and the dataset are git-ignored.

## Limitations (honest)
* **Practice-set scores are optimistic.** The intent→section routing table was written from the documents and cross-checked against the practice labels, and several prompt rules were refined after error analysis on it. The hidden test will be harder. LLM intent accuracy is the generalisable number; a held-out question set is the right next step.
* **Hindi vs Hinglish speech is acoustically similar,** and the REST API needs an explicit language, so the agent chooses `en-IN` unless it looks wrong. Hindi callers often get Latin-script (Hinglish) replies.
* **Automated voice testing used synthesised speech,** which is cleaner than phone audio; the speech-recognition front end is the most likely place to lose accuracy on real calls (a known case: the recogniser can drop the second half of a long question).
* Free-tier LLM quotas shaped the design (≈2 LLM calls per turn, disk cache, code for everything deterministic).
* Not built: insights from the historical support logs (intents, escalation reasons, deflection rate).

## Layout
```
run.py                 batch command            app.py                demo + trust panel
src/kb/                chunking by [SECTION-ID] src/retrieval/        BM25, dense, RRF, intent pins, precedence
src/policy/            calendar + KB-parsed rules  src/agent/         normaliser, understand, RAG, conversation, guard
src/voice/             Gnani I/O, speech rewrite, language choice, voice turn      src/ui/   demo logic
evals/                 measurement scripts      tests/                offline tests      scripts/   setup checks, sample outputs
```
