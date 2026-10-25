"""Rewrite a reply for the ear, complementing (not replacing) Gnani's built-in text normalization.

Gnani's TTS already reads dates ("27 October 2026"), rupee amounts ("₹10,000") and ordinary numbers.
What it does NOT handle well, per its Input Formatting Guide, are identifiers: so we spell SKUs, group long
digit runs (order IDs, bank reference numbers) and expand abbreviations it does not know (QC, RRN, COD ...).
Internal section IDs are removed (they must never be spoken).
"""
import re
from src.agent.rag import clean_reply

_WEEKDAY = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
_PAREN_DAY = re.compile(rf"\s*\(\s*({_WEEKDAY})\s*\)")
_SKU = re.compile(r"\b([A-Z]{3})-(\d{4})\b")
_LONG_DIGITS = re.compile(r"(?<![₹\d,.\-])\d{7,}(?![\d,.])")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2705]")
_ABBR = {"QC": "Q C", "RRN": "R R N", "COD": "C O D", "SKU": "S K U", "ID": "I D", "OTP": "O T P"}
_ABBR_RX = re.compile(r"\b(" + "|".join(_ABBR) + r")\b")


def _group(m: re.Match) -> str:
    return " ".join(re.findall(r"\d{1,4}", m.group(0)))


def to_speech(text: str, lang: str = "en") -> str:
    t = clean_reply(text)
    t = re.sub(r"[*_`#>~]+", "", t)
    t = _EMOJI.sub("", t)
    t = _PAREN_DAY.sub(r", \1", t)
    t = _SKU.sub(lambda m: " ".join(m.group(1)) + " " + " ".join(m.group(2)), t)
    t = _LONG_DIGITS.sub(_group, t)
    t = _ABBR_RX.sub(lambda m: _ABBR[m.group(1)], t)
    return re.sub(r"\s{2,}", " ", t).strip()
