"""Demo app: talk to the ZentroMart voice agent in the browser.   python app.py   (then allow the microphone)

Left: speak, upload a clip or type, and hear the reply. Right: the TRUST PANEL: what the agent decided, the document sections
it relied on, the facts the code calculated, the tools it called, and whether the caller is verified. Same pipeline as `run.py`.
"""
import argparse
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradio as gr
from src.ui import render, theme
from src.ui.session import DEMO_SCRIPT, MODES, DemoSession

EXAMPLES = ["Hello, mujhe order ke baare mein help chahiye.", "Hi, I need some help with an order.", "Order number hai 1859 9253.",
            "Last four digits 4265 hain.", "Kya main Kaafi Running Sneakers return kar sakti hoon?", "Aur refund kitne din mein aayega?",
            "Can I speak to a person?", "Is there a student discount?"]
IDLE_STATUS = render.empty("Speak or type a message to begin.")
IDLE = render.empty("Nothing yet.")


def style_kwargs(fn) -> dict:
    """Gradio 6 takes theme/css in launch(); older versions in Blocks(). Pass them wherever they are accepted."""
    params = inspect.signature(fn).parameters
    kw = {}
    if "theme" in params:
        kw["theme"] = theme.theme()
    if "css" in params:
        kw["css"] = theme.CSS
    return kw


def build(session_factory=DemoSession):
    def respond(view, session):
        return (session, view["transcript"], view["reply"], view["audio"], view["chat"], view["status"],
                view["sources"], view["facts"], view["tools"], None, "")

    def blank(session, msg):
        return (session, "", "", None, render.chat_html(session.log), render.notice_html(msg), IDLE, IDLE, IDLE, None, "")

    def on_audio(audio, mode, stt_lang, session):
        session = session or session_factory()
        return respond(session.handle_audio(audio, mode, stt_lang), session) if audio else blank(session, "Record or upload something first.")

    def on_text(text, mode, session):
        session = session or session_factory()
        return respond(session.handle_text(text, mode), session) if (text or "").strip() else blank(session, "Type a message first.")

    def on_reset(session):
        session = session or session_factory()
        session.reset()
        return (session, "", "", None, render.chat_html([]), render.notice_html("New conversation started. Say hello!"), IDLE, IDLE, IDLE, None, "")

    with gr.Blocks(title="VoiceOps · ZentroMart voice support agent", **style_kwargs(gr.Blocks.__init__)) as demo:
        gr.HTML(theme.HERO)
        session = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                mode = gr.Radio(choices=MODES, value="conversation", label="Mode")
                with gr.Group():
                    mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Speak (the reply plays automatically when you stop)")
                    with gr.Row():
                        box = gr.Textbox(placeholder="…or type a message and press Enter", show_label=False, scale=6, lines=1, max_lines=3)
                        send = gr.Button("Send ➤", variant="primary", scale=1)
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[box], label="Quick messages (click, then Send)")
                chat = gr.HTML(render.chat_html([]))
                voice = gr.Audio(label="🔊 Agent voice", type="filepath", autoplay=True, interactive=False)
                with gr.Row():
                    heard = gr.Textbox(label="Heard", interactive=False, scale=1)
                    reply = gr.Textbox(label="Agent reply", interactive=False, lines=2, scale=2)
                with gr.Row():
                    new = gr.Button("🔄 New conversation")
                    stt_lang = gr.Dropdown(choices=["auto", "en-IN", "hi-IN"], value="auto", label="Speech language", scale=2)
            with gr.Column(scale=5):
                gr.Markdown("## 🔍 Trust panel\n*Everything the agent used to answer, so you can audit it.*")
                status = gr.HTML(IDLE_STATUS)
                with gr.Accordion("📚 Document sections the answer relied on", open=True):
                    sources = gr.HTML(IDLE)
                with gr.Accordion("🧮 Facts calculated by code (dates, fees, windows)", open=True):
                    facts = gr.HTML(IDLE)
                with gr.Accordion("🔧 Tool calls", open=True):
                    tools = gr.HTML(IDLE)
                with gr.Accordion("🎬 Suggested demo script", open=False):
                    gr.Markdown(DEMO_SCRIPT)
        gr.HTML(theme.FOOTER)
        outs = [session, heard, reply, voice, chat, status, sources, facts, tools, mic, box]
        mic.stop_recording(on_audio, [mic, mode, stt_lang, session], outs, api_name="speak")
        mic.upload(on_audio, [mic, mode, stt_lang, session], outs, api_name="upload")
        box.submit(on_text, [box, mode, session], outs, api_name="type")
        send.click(on_text, [box, mode, session], outs, api_name="send")
        new.click(on_reset, [session], outs, api_name="reset")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link (needs internet)")
    ap.add_argument("--no-warmup", action="store_true")
    a = ap.parse_args()
    if not a.no_warmup:
        print("Loading documents and the embedding model (first run downloads ~0.5 GB)…", flush=True)
        from src.agent.rag import _ctx
        _ctx()
    demo = build()
    # queue: at most 2 turns processed at once and 16 waiting, so a burst of visitors cannot exhaust the API quotas
    demo.queue(default_concurrency_limit=2, max_size=16).launch(
        server_name="127.0.0.1", server_port=a.port, inbrowser=True, share=a.share, **style_kwargs(demo.launch))


if __name__ == "__main__":
    main()
