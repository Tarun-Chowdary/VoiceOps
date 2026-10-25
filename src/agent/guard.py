"""Output guard: the agent must never say it DID something that its code did not do.

Sentences claiming a completed action (cancelled, approved, credited, refunded, ticket created ...)
are removed. A ticket claim is allowed only when create_ticket really returned created=True.
"""
import re

_ACTION_EN = r"(cancel+ed|approved|credited|issued|processed|refunded|added|applied|arranged|scheduled|replaced|waived)"
_PATTERNS = [
    re.compile(rf"\b(i['’]?ve|i have|we['’]?ve|we have|has been|have been|is now|was)\s+(?:\w+\s+)?{_ACTION_EN}\b", re.I),
    re.compile(r"\b(cancel|approve|credit|refund|add|apply|replace)\s+kar\s+(diya|di|dii|chuka|chuki|di hai|diye)\b", re.I),
    re.compile(r"\b(voucher|coupon|credit)\s+(added|applied|issued)\b", re.I),
    re.compile(r"(कैंसिल|रद्द|अप्रूव|मंज़ूर|मंजूर|क्रेडिट|रिफ़ंड|रिफंड)\s*(कर\s*)?(दिया|दी|कर दिया|कर दी|हो गया|हो गई)"),
]
_TICKET = [
    re.compile(r"\b(ticket)\b.{0,40}\b(created|raised|logged|opened|made)\b|\b(created|raised|logged|opened|made)\b.{0,40}\bticket\b", re.I),
    re.compile(r"टिकट.{0,25}(बना|बनाया|बनाई|दर्ज)"),
    re.compile(r"ticket.{0,25}(bana|ban gaya|ban chuka|raise|create)", re.I),
]
_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def strip_unsupported_claims(reply: str, ticket_created: bool = False, allowed_amounts=frozenset()) -> tuple[str, list[str]]:
    kept, removed = [], []
    for sent in _SPLIT.split(reply.strip()):
        claims = any(p.search(sent) for p in _PATTERNS)
        if claims and allowed_amounts:       # reporting a credit/refund that the ORDER RECORD shows is a fact, not an invented action
            nums = {n.replace(",", "") for n in re.findall(r"\d[\d,]*", sent)}
            claims = not (nums & set(allowed_amounts))
        bad = claims or (not ticket_created and any(p.search(sent) for p in _TICKET))
        (removed if bad else kept).append(sent)
    return " ".join(kept).strip(), removed
