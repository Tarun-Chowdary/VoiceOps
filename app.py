"""Demo app: talk to the ZentroMart voice agent in the browser.   python app.py   (then allow the microphone)

Left: speak (or upload a clip, or type) and hear the reply.  Right: the TRUST PANEL, which shows what the agent
decided, the document chunks it relied on, the facts the code calculated, the tools it called and whether the
caller was verified. Everything runs through the same pipeline as the batch command `python run.py`.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gradio as gr
from src.ui.session import DEMO_SCRIPT, MODES, DemoSession

NOT_READY = "_Nothing yet: speak or type a message._"


def build(session_factory=DemoSession):
    def respond(view, session):
        return (session, view["transcript"], view["reply"], view["audio"], view["chat"], view["status"],
                view["sources"], view["facts"], view["tools"], None, "")

    def on_audio(audio, mode, stt_lang, session):
        session = session or session_factory()
        if not audio:
            return (session, "", "", None, "", "Record or upload something first.", NOT_READY, NOT_READY, NOT_READY, None, "")
        return respond(session.handle_audio(audio, mode, stt_lang), session)

    def on_text(text, mode, session):
        session = session or session_factory()
        if not (text or "").strip():
            return (session, "", "", None, "", "Type a message first.", NOT_READY, NOT_READY, NOT_READY, None, "")
        return respond(session.handle_text(text, mode), session)

    def on_reset(session):
        session = session or session_factory()
        session.reset()
        return (session, "", "", None, "", "New conversation started. Say hello!", NOT_READY, NOT_READY, NOT_READY, None, "")

    with gr.Blocks(title="VoiceOps: ZentroMart voice support agent") as demo:
        gr.Markdown("# 🎙️ VoiceOps · ZentroMart AI support agent\n"
                    "Speak in English, Hindi or Hinglish. The agent answers **only from company documents**, verifies who you are "
                    "before touching an order, and hands over to a human when policy says a person must decide.")
        session = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio(choices=MODES, value="conversation", label="Mode")
                stt_lang = gr.Dropdown(choices=["auto", "en-IN", "hi-IN"], value="auto", label="Speech language (auto picks en-IN or hi-IN)")
                mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Speak (the reply plays automatically after you stop)")
                with gr.Row():
                    box = gr.Textbox(placeholder="…or type a message and press Enter", show_label=False, scale=5)
                    send = gr.Button("Send", scale=1)
                new = gr.Button("🔄 New conversation")
                heard = gr.Textbox(label="Heard", interactive=False)
                reply = gr.Textbox(label="Agent reply", interactive=False, lines=3)
                voice = gr.Audio(label="🔊 Agent voice", type="filepath", autoplay=True, interactive=False)
                chat = gr.Markdown()
            with gr.Column(scale=5):
                gr.Markdown("## 🔍 Trust panel")
                status = gr.Markdown(NOT_READY)
                with gr.Accordion("📚 Document sections the answer relied on", open=True):
                    sources = gr.Markdown(NOT_READY)
                with gr.Accordion("🧮 Facts calculated by code (dates, fees, windows)", open=False):
                    facts = gr.Markdown(NOT_READY)
                with gr.Accordion("🔧 Tool calls", open=True):
                    tools = gr.Markdown(NOT_READY)
                with gr.Accordion("🎬 Suggested demo script", open=False):
                    gr.Markdown(DEMO_SCRIPT)
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
    build().queue().launch(server_name="127.0.0.1", server_port=a.port, inbrowser=True, share=a.share)


if __name__ == "__main__":
    main()
