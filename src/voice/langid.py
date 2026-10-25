"""Choose between the en-IN and hi-IN transcripts of the same audio.

Gnani's REST STT needs an explicit language_code (auto-detect exists only for streaming). en-IN accepts
English mixed with Hindi and returns Latin script, so it is the default; hi-IN (Devanagari) is used only when
the en-IN transcript is empty or implausible while the hi-IN one looks like real Hindi.
Plausibility = share of tokens that are known words (KB vocabulary + common words); a heuristic that
evals/run_voice_eval.py measures on synthesized audio.
"""
import re
from functools import lru_cache
from src.agent.lang import HINGLISH
from src.retrieval.bm25 import STOP

COMMON_EN = set("""i my me the a an is are was it to of in on for and or not can do you your we what when how why where which
please want need help order orders number last four digits digit pincode phone return refund cancel delivery delivered
late item product money back days day week today yesterday tomorrow time check status track tracking""".split())
HI_STOP = set("""मैं मैंने है हैं था थी थे का की के को से में पर और या कि क्या कैसे कब कितने कितना कितनी मुझे मेरा मेरी मेरे हमारा
आप आपका आपकी आपके यह वह इस उस नहीं हाँ हां कर करना करें सकता सकती सकते चाहिए होगा होगी लेकिन तो भी अब ऑर्डर रिफंड
रिफ़ंड वापस कितने दिन डिलीवरी कैंसिल""".split())


@lru_cache(maxsize=1)
def _vocab():
    from src.kb.loader import load_chunks
    lat, dev = set(), set()
    for c in load_chunks():
        lat.update(re.findall(r"[a-z]+", c.search_text.lower()))
        dev.update(re.findall(r"[\u0900-\u097F]+", c.search_text))
    return lat | set(STOP) | HINGLISH | COMMON_EN, dev | HI_STOP


def plausibility(text: str) -> float:
    lat, dev = _vocab()
    toks = re.findall(r"[a-z]+|[\u0900-\u097F]+|\d+", text.lower())
    if not toks:
        return 0.0
    known = sum(1 for t in toks if t.isdigit() or t in lat or t in dev)
    return known / len(toks)


def choose(t_en: str, t_hi: str) -> tuple[str, str]:
    """Returns (language_code, transcript)."""
    t_en, t_hi = (t_en or "").strip(), (t_hi or "").strip()
    if not t_en and not t_hi:
        return "en-IN", ""
    if not t_hi:
        return "en-IN", t_en
    if not t_en:
        return "hi-IN", t_hi
    s_en, s_hi = plausibility(t_en), plausibility(t_hi)
    if s_en >= 0.5 or s_en >= s_hi - 0.05:
        return "en-IN", t_en
    return "hi-IN", t_hi
