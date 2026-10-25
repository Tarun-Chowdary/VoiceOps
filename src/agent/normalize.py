"""ASR text normalizer: turn speech-recogniser output into clean, searchable text.

Typical ASR damage seen in practice/questions.csv (asr_text vs question):
  'ele-zero zero one to'   -> 'ELE-0012'   ("to" is the homophone of "two")
  'two zero two six'       -> '2026'
  'u p i', 're fund'       -> 'upi', 'refund'
  leading fillers: 'uh', 'so', 'haan', 'hello'
Everything here is deterministic code: no LLM call, no quota.
"""
import re
from functools import lru_cache

DIGITS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
          "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
# Hindi digit words (spoken order IDs / PINs from Hindi speakers). Treated as digits only in runs of >= 3 tokens.
HI_DIGITS = {"शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
             "छह": "6", "छः": "6", "छे": "6", "सात": "7", "आठ": "8", "नौ": "9",
             "shunya": "0", "ek": "1", "teen": "3", "char": "4", "chaar": "4", "paanch": "5", "panch": "5",
             "chhe": "6", "chhah": "6", "cheh": "6", "saat": "7", "aath": "8", "nau": "9"}
HI_HOMO = {"दो": "2", "do": "2"}                                   # also ordinary words ("give", "do"): digits only inside a run
DEV_NUM = str.maketrans("०१२३४५६७८९", "0123456789")
# homophones that are only treated as digits when sitting inside a digit run
HOMO = {"to": "2", "too": "2", "for": "4", "won": "1", "ate": "8"}
UNAMBIG = set(DIGITS) - {"oh"}
FILLERS = {"uh", "um", "umm", "hmm", "er"}
LEADING = {"so", "haan", "hello", "hi", "ok", "okay", "well", "ji", "haanji"}

_WORD = re.compile(r"[\w\u0900-\u097F]+(?:-[\w]+)*|[^\w\s]", re.U)


_ALLD = {**DIGITS, **HI_DIGITS}
_ALLH = {**HOMO, **HI_HOMO}


def _digit_runs(tokens):
    """Collapse spoken-digit runs ('zero zero one to', 'सात नौ तीन छह') into digit strings."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _ALLD or t in _ALLH:
            j = i
            while j < len(tokens) and (tokens[j] in _ALLD or tokens[j] in _ALLH):
                j += 1
            run = tokens[i:j]
            hindi = any(w in HI_DIGITS or w in HI_HOMO for w in run)
            unamb = any(w in UNAMBIG or w in HI_DIGITS for w in run)
            if len(run) >= (3 if hindi else 2) and unamb:
                out.append("".join(_ALLD.get(w) or _ALLH[w] for w in run))
            else:                       # a lone 'to'/'one'/'ek aur' etc.: leave untouched
                out.extend(run)
            i = j
        else:
            out.append(t)
            i += 1
    return out


def _sku_digits(tokens):
    """Handle 'ele-zero zero one to' where the prefix is glued to the first spoken digit."""
    fixed = []
    for t in tokens:
        m = re.fullmatch(r"([a-z]{3})-(zero|oh|one|two|three|four|five|six|seven|eight|nine)", t)
        if m:
            fixed.extend([m.group(1) + "-", m.group(2)])
        else:
            fixed.append(t)
    return fixed


def _join_sku(tokens):
    out, i = [], 0
    while i < len(tokens):
        if re.fullmatch(r"[a-z]{3}-", tokens[i]) and i + 1 < len(tokens) and tokens[i + 1].isdigit():
            out.append(tokens[i] + tokens[i + 1].zfill(4)[-4:] if len(tokens[i + 1]) <= 4 else tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def _join_upi(tokens):
    """Speech recognisers often split the payment method: 'up i' -> 'upi' (very common in refund questions)."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "up" and i + 1 < len(tokens) and tokens[i + 1] == "i":
            out.append("upi")
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def _join_letters(tokens):
    """'u p i' -> 'upi' (runs of >=3 single letters)."""
    out, i = [], 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha():
            j += 1
        if j - i >= 3:
            out.append("".join(tokens[i:j]))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


@lru_cache(maxsize=1)
def _vocab():
    """(content_words, all_words). content_words exclude stopwords; all_words keep them."""
    from src.kb.loader import load_chunks
    from src.retrieval.bm25 import tokenize
    content, allw = set(), set()
    for c in load_chunks():
        content.update(tokenize(c.search_text))
        allw.update(re.findall(r"[\w]+", c.search_text.lower()))
    allw.update(content)
    return content, allw


@lru_cache(maxsize=1)
def _known_words():
    """Words we can safely split glued tokens into: KB vocabulary + Hinglish + product title words + common English."""
    import csv
    from src import config
    from src.agent.lang import HINGLISH
    from src.voice.langid import COMMON_EN
    content, allw = _vocab()
    known = set(allw) | set(HINGLISH) | set(COMMON_EN) | {"at", "the", "was", "is", "a", "an", "of", "to", "in", "on", "it", "my", "i", "if", "by", "for", "with", "and", "or", "me", "ko", "ke", "ki", "ka", "se", "ne", "pe"}
    try:
        for row in csv.DictReader(open(config.DATA_DIR / "products.csv", encoding="utf-8")):
            known.update(re.findall(r"[a-z0-9]+", row["title"].lower()))
    except Exception:
        pass
    return known


def _split_glued(tokens):
    """'deliveryat' -> 'delivery at'; 'cardwas' -> 'card was'. Only splits an UNKNOWN token into two KNOWN words."""
    known, out = _known_words(), []
    for t in tokens:
        if len(t) >= 5 and t.isalpha() and t.isascii() and t not in known:
            cands = [(a, b) for k in range(2, len(t) - 1) for a, b in [(t[:k], t[k:])] if a in known and b in known]
            if cands:
                a, b = max(cands, key=lambda ab: min(len(ab[0]), len(ab[1])))
                out.extend([a, b])
                continue
        out.append(t)
    return out


def _join_split_words(tokens):
    """'re fund' -> 'refund' (conservative: precision over recall)."""
    content, allw = _vocab()
    out, i = [], 0
    while i < len(tokens):
        # join only if the result is a KB word AND neither fragment is a known word
        # (so "the re" never becomes "there", but "re fund" becomes "refund")
        if (i + 1 < len(tokens) and tokens[i].isalpha() and tokens[i + 1].isalpha()
                and tokens[i] + tokens[i + 1] in content
                and tokens[i] not in allw and tokens[i + 1] not in allw):
            out.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def normalize_asr(text: str) -> str:
    text = re.sub(r"(?<=\d),(?=\d)", "", text.translate(DEV_NUM))      # ITN writes "1,859 9,253": drop separators inside numbers
    text = re.sub(r"(?<=[a-z]{4})(?=\d)", " ", text.lower())                 # "refund12th" -> "refund 12th" (>=4 letters, so k7 / x2 / ele0012 stay)
    toks = [t for t in _WORD.findall(text) if t.strip()]
    toks = [t for t in toks if t not in FILLERS]
    while toks and toks[0] in LEADING:
        toks = toks[1:]
    toks = _sku_digits(toks)
    toks = _digit_runs(toks)
    toks = _join_sku(toks)
    toks = _join_letters(toks)
    toks = _join_upi(toks)
    toks = _split_glued(toks)
    toks = _join_split_words(toks)
    return " ".join(toks)
