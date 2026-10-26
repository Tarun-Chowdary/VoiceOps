"""HTML for the trust panel and the chat. Everything user- or model-generated is escaped (the reply text is untrusted)."""
import html
import re

DECISION_KIND = {"answer": "green", "clarify": "amber", "escalate": "red", "refuse": "purple"}
DECISION_ICON = {"answer": "✔", "clarify": "❓", "escalate": "🙋", "refuse": "⛔"}
esc = html.escape


def chip(text: str, kind: str = "slate") -> str:
    return f'<span class="vo-chip vo-{kind}">{esc(str(text))}</span>'


def clip(text: str, n: int = 360) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n].rstrip() + " …"


def strip_heading(text: str) -> str:
    """Document chunks start with a markdown heading ('## [STY-1] Opening ...'): show it as plain text, not giant bold."""
    return re.sub(r"^\s*#+\s*", "", text)


def empty(msg: str) -> str:
    return f'<div class="vo-empty">{esc(msg)}</div>'


def chat_html(log: list[tuple[str, str, str]]) -> str:
    if not log:
        return empty("The conversation will appear here.")
    rows = []
    for user, agent, decision in log[-8:]:
        kind = DECISION_KIND.get(decision, "slate")
        badge = f'<span class="vo-mini vo-{kind}">{DECISION_ICON.get(decision, "•")} {esc(decision or "-")}</span>'
        rows.append(f'<div class="vo-row vo-right"><div class="vo-bubble vo-user">{esc(user)}</div></div>'
                    f'<div class="vo-row"><div class="vo-bubble vo-agent">{esc(agent)}<div class="vo-badge">{badge}</div></div></div>')
    return '<div class="vo-chat">' + "".join(rows) + "</div>"


def status_html(rec: dict) -> str:
    chips = [chip(f"{DECISION_ICON.get(rec['decision'], '•')} {(rec['decision'] or '-').upper()}", DECISION_KIND.get(rec["decision"], "slate")),
             chip(f"intent · {rec['intent'] or '-'}", "indigo"), chip(f"language · {rec['language']}", "indigo")]
    chips.append(chip("✅ verified", "green") if rec["verified"] else chip("🔒 not verified", "slate"))
    if rec["ticket_id"]:
        chips.append(chip(f"🎫 {rec['ticket_id']}", "red"))
    if rec["stt_language"] and rec["stt_language"] != "typed":
        chips.append(chip(f"heard via {rec['stt_language']}", "teal"))
    timing = (f'<div class="vo-timing">⏱ speech-to-text {rec.get("stt_seconds", 0)}s · agent {rec.get("agent_seconds", 0)}s · '
              f'voice {rec.get("tts_seconds", 0)}s</div>')
    err = f'<div class="vo-error">⚠️ {esc(rec["error"])}</div>' if rec["error"] else ""
    return f'<div class="vo-chips">{"".join(chips)}</div>{timing}{err}'


def sources_html(rec: dict) -> str:
    cards = "".join(
        f'<div class="vo-src"><div class="vo-src-head"><span class="vo-pill">{esc(c["section_id"])}</span>'
        f'<span class="vo-doc">{esc(c["doc_id"])}</span></div><div class="vo-src-body">{esc(clip(strip_heading(c["text"])))}</div></div>'
        for c in rec["used_chunks"])
    if not cards:
        return empty("No document section was needed for this turn (greeting, identity check or handover wording).")
    extra = [s for s in rec["retrieved_sections"] if s not in rec["used_sections"]]
    more = f'<div class="vo-more">Also retrieved, not used: {" ".join(chip(s) for s in extra)}</div>' if extra else ""
    return cards + more


def facts_html(rec: dict) -> str:
    if not rec["facts"]:
        return empty("No calculated facts for this turn.")
    return '<ul class="vo-list">' + "".join(f"<li>{esc(f)}</li>" for f in rec["facts"]) + "</ul>"


def tools_html(rec: dict) -> str:
    if not rec["tool_calls"]:
        return empty("No tools called this turn.")
    return "".join(f'<div class="vo-tool"><code>{esc(t)}</code></div>' for t in rec["tool_calls"])


def notice_html(msg: str) -> str:
    return f'<div class="vo-notice">{esc(msg)}</div>'
