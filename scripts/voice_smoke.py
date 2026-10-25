"""Answers the questions that cannot be answered without your Gnani key. Run:  python scripts/voice_smoke.py
1. Does your SDK's transcribe() accept the 'format' option?  2. Does Timbre v3.0 / Shivani work (else the fallback)?
3. How do en-IN vs hi-IN and verbatim vs transcribe behave on English, Hindi and Hinglish speech?"""
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config  # loads .env
from src.voice.gnani_io import GnaniSTT, GnaniTTS, _accepts
from src.voice.langid import choose, plausibility

stt_v, stt_t = GnaniSTT(fmt="verbatim"), GnaniSTT(fmt="transcribe")
c = stt_v._c()
print("transcribe signature:", inspect.signature(c.transcribe))
print("accepts format=      :", _accepts(c.transcribe, "format"))

out = Path(__file__).resolve().parent / "smoke_audio"
out.mkdir(exist_ok=True)

print("\n--- TTS probe: which model / voice / language options does YOUR SDK accept? ---")
probe = GnaniTTS()
combos = [("timbre-v3.0", "Shivani", "en-IN"), ("timbre-v2.5", "Kaveri", "en-IN"), ("timbre-v2.5", "Nalini", "hi-IN"),
          ("timbre-v2.5", "Poorvi", "hi-en"), ("timbre-v2.5", "Poorvi", "auto"), ("timbre-v2.5", "Deepak", "auto"),
          ("timbre-v2.5", "Nalini", "auto"), ("timbre-v2.5", "Kaveri", "auto")]
hinglish = "Namaste, main ZentroMart ka AI assistant hoon. Aapka refund do se teen din mein aa jayega."
for model, voice, code in combos:
    try:
        probe._synth(hinglish if code in ("hi-en", "auto") or voice in ("Poorvi", "Deepak", "Nalini") else "Hello, this is a voice test.",
                     code, out / f"probe_{model}_{voice}_{code}.wav", model, voice)
        print(f"  OK      {model:12s} {voice:8s} {code}")
    except Exception as e:
        print(f"  FAILED  {model:12s} {voice:8s} {code:6s} -> {type(e).__name__}: {str(e)[:170]}")
print("  -> LISTEN to scripts/smoke_audio/probe_timbre-v2.5_*_hi-en.wav / *_auto.wav (same Hinglish sentence) and pick the best-sounding voice.")

samples = [("en", "My order number is 1859 9253 and my refund to UPI started on 17 October 2026."),
           ("hi", "मेरा ऑर्डर नंबर है 1859 9253 और मुझे रिफंड कब मिलेगा?"),
           ("hi-en", "Mera order number hai 1859 9253, aur refund kab tak aayega?")]
for lang, text in samples:
    print("\n" + "=" * 70 + f"\n[{lang}] {text}")
    for label, tts in (("customer", GnaniTTS(role="customer")), ("agent", GnaniTTS())):
        try:
            p = tts.synthesize(text, lang, out / f"{lang}_{label}.wav")
            print(f"  TTS ok: {label:8s} used {tts.last_used} -> {p.name} ({p.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"  TTS FAILED for {label}: {type(e).__name__}: {e}")
    wav = out / f"{lang}_customer.wav"
    if not wav.exists():
        continue
    heard = {}
    print("  (STT: 2 languages x 2 formats = 4 requests for this sample)")
    for code in ("en-IN", "hi-IN"):
        for name, s in (("verbatim", stt_v), ("transcribe", stt_t)):
            try:
                heard[(code, name)] = s.transcribe(wav, code)
            except Exception as e:
                heard[(code, name)] = f"<error {type(e).__name__}: {e}>"
            print(f"  STT {code:5s} {name:10s} (plausibility {plausibility(heard[(code, name)]):.2f}): {heard[(code, name)]}")
    fast = plausibility(heard[("en-IN", "transcribe")]) >= 0.75
    code = "en-IN" if fast else choose(heard[("en-IN", "transcribe")], heard[("hi-IN", "transcribe")])[0]
    print(f"  agent would use: {code}" + (" (fast path: hi-IN request skipped)" if fast else "") + "   [format=transcribe, the default]")
print("\nDone. Listen to the Hinglish files in scripts/smoke_audio/ (probe_*_auto.wav, hi-en_agent.wav) and pick the voice you like best.")
