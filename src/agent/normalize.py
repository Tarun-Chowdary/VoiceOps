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
# homophones that are only treated as digits when sitting inside a digit run
HOMO = {"to": "2", "too": "2", "for": "4", "won": "1", "ate": "8"}
UNAMBIG = set(DIGITS) - {"oh"}
FILLERS = {"uh", "um", "umm", "hmm", "er"}
LEADING = {"so", "haan", "hello", "hi", "ok", "okay", "well", "ji", "haanji"}

_WORD = re.compile(r"[\w\u0900-\u097F]+(?:-[\w]+)*|[^\w\s]", re.U)


def _digit_runs(tokens):
    """Collapse spoken-digit runs ('zero zero one to') into digit strings."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in DIGITS or t in HOMO:
            j = i
            while j < len(tokens) and (tokens[j] in DIGITS or tokens[j] in HOMO):
                j += 1
            run = tokens[i:j]
            if len(run) >= 2 and any(w in UNAMBIG for w in run):
                out.append("".join(DIGITS.get(w) or HOMO[w] for w in run))
            else:                       # a lone 'to'/'one' etc.: leave untouched
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
    toks = [t for t in _WORD.findall(text.lower()) if t.strip()]
    toks = [t for t in toks if t not in FILLERS]
    while toks and toks[0] in LEADING:
        toks = toks[1:]
    toks = _sku_digits(toks)
    toks = _digit_runs(toks)
    toks = _join_sku(toks)
    toks = _join_letters(toks)
    toks = _join_split_words(toks)
    return " ".join(toks)
