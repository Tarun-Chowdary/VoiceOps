import json, re

VALID = {"answer", "clarify", "escalate", "refuse"}

def parse_json_reply(text: str) -> dict:
    """Parse the model's JSON output; tolerate code fences and stray text."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    for candidate in (t, (re.search(r"\{.*\}", t, re.S) or [None])[0] if re.search(r"\{.*\}", t, re.S) else None):
        if not candidate:
            continue
        try:
            d = json.loads(candidate)
            dec = str(d.get("decision", "answer")).lower().strip()
            used = d.get("used_sections") or []
            return {"decision": dec if dec in VALID else "answer",
                    "reply": str(d.get("reply", "")).strip(),
                    "used_sections": [str(u).strip() for u in used if isinstance(u, str)]}
        except Exception:
            continue
    return {"decision": "answer", "reply": t, "used_sections": []}   # fall back to raw text
