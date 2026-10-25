"""Word error rate on normalised tokens (case, punctuation, fillers and spoken digits are normalised first)."""
import re
from src.agent.normalize import normalize_asr


def _tokens(s: str) -> list[str]:
    return re.findall(r"[\w\u0900-\u097F]+", normalize_asr(s))


def wer(ref: str, hyp: str) -> float:
    r, h = _tokens(ref), _tokens(hyp)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)
