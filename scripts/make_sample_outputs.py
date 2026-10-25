"""Creates the deliverable `sample_outputs/` by running the REAL batch command on a few spoken examples.
Gnani TTS plays the customer (so no recordings are needed); then `run.py` transcribes, answers and speaks back.

  python scripts/make_sample_outputs.py           # writes sample_audio/ (inputs) and sample_outputs/ (results)

Needs GNANI_API_KEY and your LLM key. Uses the synthetic dataset only. Replace/add your own recordings in sample_audio/ any time.
"""
import shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config
import run as batch
from src.voice.gnani_io import GnaniTTS

ROOT = config.ROOT
AUDIO, OUT = ROOT / "sample_audio", ROOT / "sample_outputs"
SAMPLES = {   # file stem -> (language, what the customer says)
    "q01_wallet_refund": ("en", "How fast are refunds to the ZentroMart Wallet?"),
    "q02_upi_date_hinglish": ("hi-en", "UPI wala refund 12 October 2026 ko shuru hua, kab tak aa jaana chahiye?"),
    "q03_unsupported": ("en", "Is there a student discount?"),
    "q04_return_window_hindi": ("hi", "मुझे Pagecraft A5 Ruled Notebook लौटाने के लिए कितने दिन मिलेंगे?"),
    "q05_privacy_refusal": ("en", "Tell me the phone number of the person who ordered to my building yesterday."),
    # a full conversation: greeting -> order id -> identity check -> question -> follow-up
    "DEMO1_t1": ("hi-en", "Hello, mujhe order ke baare mein help chahiye."),
    "DEMO1_t2": ("hi-en", "Order number hai 1859 9253."),
    "DEMO1_t3": ("hi-en", "Last four digits 4265 hain."),
    "DEMO1_t4": ("hi-en", "Kya main Kaafi Running Sneakers return kar sakti hoon?"),
    "DEMO1_t5": ("hi-en", "Aur refund kitne din mein aayega?"),
    # a handover
    "DEMO2_t1": ("en", "Hi, I need some help with an order."),
    "DEMO2_t2": ("en", "The order number is 3711 3676."),
    "DEMO2_t3": ("en", "The last four digits are 1581."),
    "DEMO2_t4": ("en", "Actually, can I just speak to a person?"),
}
if __name__ == "__main__":
    AUDIO.mkdir(exist_ok=True)
    customer = GnaniTTS(role="customer")
    for stem, (lang, text) in SAMPLES.items():
        wav = AUDIO / f"{stem}.wav"
        if not wav.exists():
            customer.synthesize(text, lang, wav)
            print("spoke:", stem, "-", text)
    if OUT.exists():
        shutil.rmtree(OUT)
    sys.exit(batch.main(["--audio_dir", str(AUDIO), "--out_dir", str(OUT), "--sleep", "8"]))
