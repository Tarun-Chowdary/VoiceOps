"""Look and feel of the demo: an indigo-to-teal hero, colour-coded chips, chat bubbles and source cards.
Works in light and dark mode (translucent fills + Gradio's own colour variables)."""

HERO = """
<div class="vo-hero">
  <div class="vo-hero-title">🎙️ VoiceOps <span>· ZentroMart AI support agent</span></div>
  <div class="vo-hero-sub">Call, speak or type in <b>English, हिन्दी or Hinglish</b>. It answers <b>only from company documents</b>,
  verifies who you are before touching an order, and hands over to a human when policy says a person must decide.</div>
  <div class="vo-hero-chips">
    <span>📚 Grounded in documents</span><span>🔒 Identity-verified</span><span>🧮 Dates calculated by code</span>
    <span>🎫 Honest handover</span><span>🗣️ Gnani speech</span>
  </div>
</div>
"""

CSS = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.vo-hero { background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #0d9488 100%); color: #fff; border-radius: 18px;
           padding: 22px 26px; margin-bottom: 6px; box-shadow: 0 8px 28px rgba(79,70,229,.25); }
.vo-hero-title { font-size: 28px; font-weight: 800; letter-spacing: -.5px; }
.vo-hero-title span { font-weight: 500; opacity: .85; font-size: 20px; }
.vo-hero-sub { margin-top: 6px; font-size: 14.5px; opacity: .95; max-width: 880px; line-height: 1.5; }
.vo-hero-chips { margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; }
.vo-hero-chips span { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.28); padding: 4px 12px; border-radius: 999px; font-size: 12.5px; }

.vo-chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 8px; }
.vo-chip, .vo-mini { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12.5px; font-weight: 600; border: 1px solid transparent; }
.vo-mini { font-size: 11px; padding: 1px 8px; }
.vo-green  { background: rgba(16,185,129,.15); color: #10b981; border-color: rgba(16,185,129,.4); }
.vo-amber  { background: rgba(245,158,11,.15); color: #f59e0b; border-color: rgba(245,158,11,.4); }
.vo-red    { background: rgba(239,68,68,.15);  color: #ef4444; border-color: rgba(239,68,68,.4); }
.vo-purple { background: rgba(168,85,247,.15); color: #a855f7; border-color: rgba(168,85,247,.4); }
.vo-indigo { background: rgba(99,102,241,.15); color: #818cf8; border-color: rgba(99,102,241,.4); }
.vo-teal   { background: rgba(20,184,166,.15); color: #14b8a6; border-color: rgba(20,184,166,.4); }
.vo-slate  { background: rgba(148,163,184,.15); color: #94a3b8; border-color: rgba(148,163,184,.35); }
.vo-timing { font-size: 12.5px; opacity: .75; margin-top: 2px; }
.vo-error  { margin-top: 6px; padding: 8px 10px; border-radius: 10px; background: rgba(239,68,68,.12); color: #ef4444; font-size: 13px; }
.vo-notice { padding: 10px 12px; border-radius: 12px; background: rgba(245,158,11,.14); color: #f59e0b; font-weight: 600; }
.vo-empty  { padding: 12px; border: 1px dashed var(--border-color-primary); border-radius: 12px; opacity: .65; font-size: 13.5px; text-align: center; }

.vo-chat { display: flex; flex-direction: column; gap: 8px; padding: 6px 2px; }
.vo-row { display: flex; } .vo-right { justify-content: flex-end; }
.vo-bubble { max-width: 82%; padding: 10px 14px; border-radius: 16px; font-size: 14.5px; line-height: 1.45; white-space: pre-wrap; }
.vo-user  { background: linear-gradient(120deg,#4f46e5,#7c3aed); color: #fff; border-bottom-right-radius: 4px; }
.vo-agent { background: var(--block-background-fill); border: 1px solid var(--border-color-primary); border-bottom-left-radius: 4px; }
.vo-badge { margin-top: 6px; }

.vo-src { border: 1px solid var(--border-color-primary); border-left: 4px solid #6366f1; border-radius: 12px; padding: 9px 12px; margin-bottom: 8px; }
.vo-src-head { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.vo-pill { background: #6366f1; color: #fff; font-weight: 700; font-size: 12px; padding: 1px 9px; border-radius: 8px; font-family: ui-monospace, monospace; }
.vo-doc  { font-size: 12px; opacity: .6; font-family: ui-monospace, monospace; }
.vo-src-body { font-size: 13.5px; line-height: 1.5; opacity: .92; }
.vo-more { font-size: 12.5px; opacity: .75; margin-top: 4px; display: flex; flex-wrap: wrap; gap: 5px; align-items: center; }
.vo-list { margin: 0; padding-left: 18px; font-size: 13.5px; line-height: 1.55; }
.vo-tool { margin-bottom: 6px; } .vo-tool code { background: rgba(99,102,241,.12); color: #818cf8; padding: 4px 10px; border-radius: 8px; font-size: 13px; }
.vo-footer { text-align: center; font-size: 12.5px; opacity: .6; margin-top: 10px; }
"""

FOOTER = '<div class="vo-footer">Speech: Gnani STT/TTS · Retrieval: BM25 + multilingual e5 · Answers only from the ZentroMart documents · Synthetic data only</div>'


def theme():
    import gradio as gr
    return gr.themes.Soft(primary_hue="indigo", secondary_hue="teal", neutral_hue="slate",
                          font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
