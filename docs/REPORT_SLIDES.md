# Report / presentation content: VoiceOps (12 slides → fits the 4-page report)

## Slide 1: Title

**VoiceOps: a voice-first support agent that answers from the documents**
ZentroMart capstone · IIT Guwahati C&A × Gnani.ai · [Tarun Chowdary Yegi] · [[GitHub URL](https://github.com/Tarun-Chowdary/VoiceOps/)]
_Visual:_ the app's hero banner.

## Slide 2: The problem

- Customers would rather **call** than type; support answers the same questions daily (late delivery, returns, refunds, cancellation)
- LLMs alone **invent** policies, dates and amounts, and may leak data or promise things they cannot do
- Goal: a support agent that is **grounded, safe and voice-first**
  _Note:_ evaluated in four steps: LLM → RAG → conversation → voice.

## Slide 3: Solution at a glance

- Pipeline: audio → STT → normalise → understand → retrieve → calculate → answer → guard → TTS
- **Design principle: the LLM handles language; code handles facts and safety** (dates, identity check, handover, "ticket created")
  _Visual:_ the pipeline diagram.

## Slide 4: Step 1: Prompt-only baseline

- **12/52 = 23.1%** correct (17.3% on speech-style text)
- Safe on common sense (fraud, privacy), wrong on anything ZentroMart-specific
- Failures: "7 days" for every product, agreed with a false premise ("still ten days?"), invented delivery hours, gave generic refund times
  _Visual:_ bar chart of pass rate by intent.

## Slide 5: Step 2: Grounded in the documents (RAG)

- Section-level chunks that keep their `[SECTION-ID]`; hybrid **BM25 + multilingual embeddings**; **intent routing**; KB **precedence** (policy > FAQ)
- A **calculator** parses windows, fees and timelines from the documents and computes business-day deadlines
- Retrieval found all gold sections: **23% → 36% → 45% → 64% → 96%** as each component was added
- Answers: **12/52 → 52/52**
  _Visual:_ retrieval ablation bar chart + Step 1 vs Step 2 table by intent.

## Slide 6: Step 3: A real conversation

- Code state machine: ask order ID → look up → verify (phone digits or pincode) → one retry → lockout; the LLM never sees phone, pincode or customer ID
- Remembers the order, follow-ups and a question asked before verification
- Handover rules detected in code; a ticket is claimed **only if it was created**; guard removes unsupported claims
- **95/96 turns (99.0%), 22/23 conversations, tool calls 100%**
  _Visual:_ conversation flow diagram.

## Slide 7: Step 4: Giving it a voice

- Gnani STT/TTS only; `transcribe` format turns spoken numbers into digits; `en-IN` first, `hi-IN` only when needed
- Repairs speech damage: spoken digits, "up i" → UPI, garbled SKUs recovered **by number**, misheard product names matched phonetically
- Never silent: unclear audio, LLM or TTS failures still end in a spoken reply
- **51/52 = 98.1%** end to end; English WER 6.9%
  _Visual:_ before/after of a garbled transcript ("s q e le two" → ELE-0002).

## Slide 8: Beyond the brief

- 🔍 **Trust panel**: sources, calculated facts, tool calls and verification for every turn
- 🚦 **Rate limiting**: outbound LLM pacing (stays inside quotas) and per-visitor limits plus size caps on the demo
- 🧪 **Evaluation harness**: ablation, conversation replay and voice evaluation reproducible with one command each; 128 offline tests
  _Visual:_ screenshot of the app.

## Slide 9: Results summary

| Step                      | Result                 |
| ------------------------- | ---------------------- |
| 1 Prompt-only             | 23.1% (12/52)          |
| 2 RAG                     | 52/52                  |
| Retrieval (gold sections) | 95.7%                  |
| 3 Conversations           | 99.0% of turns (95/96) |
| 4 Voice end-to-end        | 98.1% (51/52)          |

_Note:_ practice-set scores, optimistic (next slides).

## Slide 10: What went wrong, and what it taught us

- **Free-tier quota of 20 requests/day** → switched model, disk cache, rate limiter
- **A WAV header claimed 24 hours** and triggered 1,627 STT requests → measure real file size, cap the pieces
- **Docs ≠ installed SDK** (no `timbre-v3.0`, no `hi-en`) → probe options, fall back, warn once
- **Our own guard deleted a true sentence** ("₹79 credited") → allow amounts present in the order record
- **Speech recognition scrambled SKUs (0/13)** → number-based recovery, phonetic matching (12/13)
  _Visual:_ before/after numbers.

## Slide 11: Limitations and next steps

- Practice scores are optimistic (routing table written with labels visible) → build a held-out question set
- Hindi vs Hinglish speech is ambiguous for the STT language choice
- Tested with synthesised speech → test real, noisy phone audio
- Input validation (an order ID of the wrong length), support-log insights, lower latency

## Slide 12: Demo and thank you

- 2-minute demo: answer + follow-up, human handover, "I don't know"
- Repo: [[text](https://github.com/Tarun-Chowdary/VoiceOps/)] · Questions?
