# 🎙️ VoiceOps

[🎬 Watch the 2-minute demo](demo.mp4) · [📄 Read the report](report.pdf)

VoiceOps is a voice-first customer-support agent for **ZentroMart**, a fictional Indian online store. Customers simply call and ask. The agent listens (Gnani speech-to-text), answers **only from the company's own policy documents**, verifies who it is talking to before it touches an order, and hands over to a human when policy says a person must decide, then speaks back (Gnani text-to-speech) in English, Hindi or Hinglish.

Built for the VoiceOps 2026 capstone (IIT Guwahati Consulting & Analytics Club × Gnani.ai).

---

## 🚀 Features

- 🗣️ Voice in, voice out in **English, Hindi and Hinglish** with Gnani STT/TTS
- 📚 Answers **only from the documents**, and says "I don't have that information" instead of inventing a policy, date or amount
- 🧮 **Dates, windows and fees are calculated by code**, from rules parsed out of the documents (the LLM never does the arithmetic)
- 🔐 **Identity check** before any order detail: registered phone's last 4 digits or delivery pincode, one retry, then lockout
- 🎫 **Honest handover**: escalation rules are enforced in code, and "ticket raised" is said only if the ticket was really created
- 🧠 **Remembers the conversation**: follow-ups ("and the refund?"), topic switches, and a question asked before verification
- 🛡️ **Output guard**: internal document IDs are stripped and any claim of an action that did not happen is removed
- 🔍 **Trust panel** in the demo: the document sections used, calculated facts, tool calls and verification status for every turn
- 🚦 **Rate limiting**: outbound pacing of LLM calls to stay inside provider quotas, plus per-visitor limits and size caps on the demo
- 🔁 **Never silent**: retries, fallbacks and a spoken apology if anything fails
- 🧪 **128 offline tests** and a one-command evaluation harness

---

## 🛠 Tech Stack

### Voice
- Gnani STT & TTS REST (`gnani-vachana` SDK): `en-IN` / `hi-IN` transcription, `timbre-v2.5` voices

### AI & Retrieval
- Any OpenAI-compatible LLM (default: Gemini 3.5 Flash-Lite)
- `intfloat/multilingual-e5-small` embeddings (local)
- BM25 written from scratch, fused with Reciprocal Rank Fusion
- A small deterministic policy calculator (business days, return windows, fees)

### Application
- Python 3.10+, pandas
- Gradio (demo app)
- pytest (offline tests with fake LLM / STT / TTS)

### Data
- Synthetic ZentroMart dataset: 9 CSV tables and 21 Markdown policy documents (not included in this repo)

---

## 🧠 How It Works

1. The caller speaks. Gnani STT transcribes it (`en-IN` first; `hi-IN` is asked too only when the first transcript looks wrong).
2. A normaliser repairs speech damage: spoken digits become digits, "up i" becomes UPI, glued words are split, and garbled spoken SKUs are recovered by number.
3. One LLM call extracts the intent, an English search query, payment method, dates and whether the question needs a verified order.
4. Hybrid retrieval (BM25 + multilingual embeddings) finds document sections, intent routing pins the ones that must be present, and the knowledge base's own precedence rules order them (policy > guide > help > FAQ > marketing; superseded documents last).
5. Code calculates the facts: refund dates in business days, return deadlines, fees, and the rules for the verified order.
6. The LLM writes a short answer using only those sections, facts and a sanitised order record, and cites what it used.
7. The guard removes internal IDs and unsupported claims; handovers create a real ticket first.
8. Gnani TTS speaks the reply, rewritten lightly for the ear (spelled-out SKUs, grouped IDs).

```
 audio ─► STT ─► normalise ─► understand ─► retrieve ─► calculate ─► answer ─► guard ─► TTS ─► audio
                                 (LLM)        (hybrid)     (code)      (LLM)    (code)
                  identity check, lockout and ESC-1 handovers are code, not LLM
```

---

## 📊 Results

| Measure (practice data) | Result |
|---|---|
| Step 1: prompt-only LLM | 12/52 = **23.1%** |
| Step 2: RAG + calculator | **52/52** on the latest run (49/52 on the first) |
| Retrieval: all gold sections found | 23.4% → 36.2% → 44.7% → 63.8% → **95.7%** (normaliser → catalog → embeddings → intent routing) |
| LLM intent accuracy | 50/52 = 96.2% |
| Step 3: 23 conversations / 96 turns | **95/96 turns = 99.0%**, 22/23 conversations fully correct, tool calls 100% |
| Step 4: voice end-to-end (Gnani TTS → STT → agent → TTS) | **51/52 = 98.1%**, 52/52 spoken replies, 0 errors, English WER 6.9% |

> Practice-set numbers are optimistic (see Limitations). Raw per-question results are in `evals/results/`.

---

## 🔒 Privacy & Safety

- The caller's phone number, pincode and customer ID are **never sent to the LLM**
- No order detail is revealed until the identity check passes; two failed attempts lock the order
- Handover and "ticket created" are decided by code, never claimed by the LLM
- Replies never contain internal document IDs or claims of actions that did not happen
- Only the synthetic dataset is used; no real customer data or recordings
- API keys live in `.env` (git-ignored); `.env.example` has no secrets
- Demo app: per-visitor rate limit, 500-character and 10 MB input caps, and a bounded request queue

---

## 📦 Installation (Local Setup)

### 1. Clone and create an environment

```bash
git clone https://github.com/your-username/voiceops.git
cd voiceops
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
```

### 2. Install

```bash
pip install -r requirements.txt           # batch command
pip install -r requirements-demo.txt      # + the demo app
```

### 3. Configure

```bash
cp .env.example .env
```

Fill in `.env`:

```
GNANI_API_KEY=your_gnani_key          # app.gnani.ai/voice (one key covers STT and TTS)
LLM_API_KEY=your_llm_key
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
LLM_MODEL=gemini-3.5-flash-lite
LLM_MAX_RPM=12                        # stay under your provider's per-minute quota (0 = off)
DATASET_DIR=./dataset/Capstone_Project_Dataset_FTTT_2026
```

Extract the organizer dataset to `DATASET_DIR` (it is intentionally not part of this repo). The first run downloads a ~0.5 GB embedding model; if it cannot be loaded the agent falls back to BM25-only retrieval instead of crashing.

---

## ▶️ Usage

### Talk to it in the browser

```bash
python app.py
```

Allow the microphone, speak (or type), and watch the **trust panel** on the right. A 2-minute demo script is in the app's "Suggested demo script" accordion and in `docs/DEMO_SCRIPT.md`.

### Batch command (what the organizers run)

```bash
python run.py --audio_dir ./customer_audio --out_dir ./out
```

For every audio file (`wav mp3 ogg flac aac m4a`) it saves in `out/`:

- `<name>.wav`: the spoken reply
- `<name>.txt`: the reply text
- `<name>.json`: transcript, decision, intent, the **document chunks retrieved and used (with their text)**, calculated facts, tool calls, ticket ID, timings and any error
- `results.csv` / `results.jsonl` for the whole folder

Files are independent questions by default. Files named `<conversation>_t1.wav`, `<conversation>_t2.wav`, … are replayed as **one conversation** (use `--mode conversation` or `--manifest manifest.csv` for other naming). Exit code 0 means every file got a spoken reply.

`python scripts/make_sample_outputs.py` creates `sample_audio/` and `sample_outputs/` by running this command on a few spoken examples.

---

## 🧪 Evaluate & Test

```bash
python -m pytest tests -q                                  # 128 offline tests, no keys needed
python evals/run_questions.py --agent baseline             # Step 1
python evals/run_questions.py --agent rag                  # Step 2
python evals/compare_steps.py                              # Step 1 vs Step 2 table
python evals/run_ablation.py                               # retrieval ablation (no LLM)
python evals/run_pipeline.py --intent llm                  # retrieval with LLM intent routing
python evals/run_conversations.py                          # Step 3 replay
python scripts/voice_smoke.py                              # checks your Gnani setup
python evals/run_voice_eval.py                             # Step 4 end-to-end voice evaluation
```

---

## 🛠️ Troubleshooting

- **`429 RESOURCE_EXHAUSTED` / slow batch runs:**
  - Your LLM quota is exhausted. Set `LLM_MAX_RPM` to your per-minute limit and use a model with a larger daily quota. Answers are cached on disk, so re-runs are free.
- **`Unsupported model 'timbre-v3.0'` or `Unsupported language 'hi-en'`:**
  - Older `gnani-vachana` versions reject them. The agent detects this and falls back (`timbre-v2.5`, language `auto` for Hinglish); run `pip install -U gnani-vachana` to use the newer options.
- **`getaddrinfo failed` / connection reset:**
  - A network or DNS problem. Calls are retried with backoff; check your connection or VPN.
- **`PermissionError` on Windows inside OneDrive:**
  - OneDrive locks folders it is syncing. Move the project outside OneDrive (for example `C:\dev\voiceops`).
- **`UnicodeEncodeError` / garbled Hindi in the terminal:**
  - Output is forced to UTF-8; if you pipe it through PowerShell, run `[Console]::OutputEncoding = [Text.Encoding]::UTF8` first.
- **No sound / no microphone in the demo:**
  - Allow microphone access for `127.0.0.1` in the browser, or type a message instead.
- **`HF Hub` unauthenticated warning:**
  - Harmless; set `HF_TOKEN` for faster model downloads.

---

## 🚧 Limitations

- Practice-set scores are optimistic: the intent-to-section routing table and several prompt rules were refined after error analysis on it. LLM intent accuracy is the generalisable number; a held-out question set is the next step.
- Hindi and Hinglish speech sound alike and the REST API needs an explicit language, so Hindi callers often get Latin-script (Hinglish) replies.
- Automated voice tests used synthesised speech, which is cleaner than phone audio. Speech recognition is the likeliest place to lose accuracy on real calls.
- Free-tier LLM quotas shaped the design: about two LLM calls per turn, a disk cache, and code for everything deterministic.
- Not built: insights from the historical support logs.

---

## 🗂️ Project Layout

```
run.py              batch command                 app.py            demo + trust panel
src/kb/             chunking by [SECTION-ID]      src/retrieval/    BM25, dense, RRF, intent pins, precedence
src/policy/         calendar + KB-parsed rules    src/agent/        normaliser, understanding, RAG, conversation, guard, rate limiter
src/voice/          Gnani I/O, speech rewrite     src/ui/           demo logic, rendering, theme
evals/              measurement scripts           tests/            offline tests
scripts/            setup checks, sample outputs  docs/             report slides, demo script
```

---

## 📫 Contact / Support

- For questions or suggestions, please open an issue on the [GitHub repository](https://github.com/your-username/voiceops/issues).
- You can also reach out via email: your-email@example.com

We welcome feedback and contributions!
