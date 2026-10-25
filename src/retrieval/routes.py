"""Intent -> policy sections that must be in front of the LLM ("pinned" context).

Authoring note (disclose this in the report): written by reading the KB, then cross-checked
against the practice labels, so scores on the practice set are optimistic. These pins are
only a PRIOR: hybrid retrieval still adds whatever else matches the question.
The LLM answers only from section TEXT; nothing here contains an answer.
"""
ROUTES: dict[str, list[str]] = {
    # refunds are counted in business days (KB-00.3); FAQs contradict policy, policy wins (KB-00.1)
    "refund_timeline": ["REF-2", "KB-00.3", "KB-00.1"],
    # windows depend on category (RET-2), the 1-Aug-2026 electronics change (RET-3), festive sale (RET-8)
    "return_window": ["RET-2", "RET-3", "RET-8"],
    "return_eligibility": ["RET-2", "RET-4", "RET-5", "RET-6"],     # RET-6: reverse pickup fee
    "return_fee": ["RET-6", "PLS-2", "DWM-2"],
    "cancel_order": ["CAN-1", "CAN-2", "REF-1"],  # before shipping / after shipping / when the refund starts
    "order_status": ["TRK-1"],
    "damaged_item": ["DWM-1", "DWM-3", "ESC-1"],
    "late_delivery": ["DEL-4"],
    "delivery_attempts": ["DEL-3"],
    "delivery_area": ["DEL-5"],
    "warranty": ["WAR-1", "WAR-2"],
    "fraud": ["ESC-1"],
    "policy_override": ["ESC-4", "IDV-4"],
}


def pinned_sections(intent: str, payment_method: str | None = None, category: str | None = None, has_refund: bool = False, has_return: bool = False) -> list[str]:
    pins = list(ROUTES.get(intent, []))
    if intent == "refund_timeline" and payment_method == "cod":
        pins.append("REF-3")                              # cash-on-delivery refunds
    if intent == "return_window" and category == "electronics":
        pins.append("RETv3-2")                            # RET-3: pre-1-Aug electronics keep the v3 window
    if intent == "refund_timeline" and has_refund:
        pins.append("REF-6")                              # share the bank reference number (RRN) when money has not arrived
    if intent == "refund_timeline" and has_return:
        pins.append("RET-7")                              # QC / refund-initiation steps of a return in progress
    return pins
