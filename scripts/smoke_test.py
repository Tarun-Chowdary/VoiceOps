import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config

def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return ok

# 1. Data row counts (from the brief)
expected = {"customers": 300, "orders": 1000, "order_items": 1783,
            "products": 70, "pincodes": 36, "holidays": 4, "support_logs": 244}
for name, n in expected.items():
    df = pd.read_csv(config.DATA_DIR / f"{name}.csv", dtype=str)
    check(f"{name}.csv rows", len(df) == n, f"({len(df)})")

# 2. Tools
from tools import lookup_order, create_ticket
r = lookup_order("8978 2837")
check("unverified lookup leaks nothing",
      r.get("verification_required") and "status" not in r)
r = lookup_order("8978 2837", verification_value="4533")
check("verified lookup works", r.get("verified") is True, f"status={r.get('status')}")
t = create_ticket("Smoke test ticket: verifying handover tool works end to end.",
                  order_id="89782837", reason="smoke_test")
check("create_ticket", t.get("created") is True, t.get("ticket_id", ""))
tf = config.DATASET_DIR / "tools" / "tickets_created.csv"
if tf.exists():
    tf.unlink()  # don't leave test tickets around

# 3. Gnani TTS -> STT round trip (uses the SDK calls from the brief)
import os
if not os.getenv("GNANI_API_KEY"):
    check("GNANI_API_KEY set", False)
else:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    tts, stt = GnaniTTSClient(), GnaniSTTClient()
    out = config.ROOT / "scripts" / "smoke.wav"
    tts.synthesize("My order has not arrived yet.", voice="Kaveri",
                   model="timbre-v2.5", language="en-IN", output_file=str(out))
    heard = stt.transcribe(str(out), language_code="en-IN")["transcript"]
    check("Gnani TTS->STT", "order" in heard.lower(), f"heard: {heard!r}")

# 4. LLM: fill in once you've picked one
print("[TODO] LLM call: add after choosing a provider")