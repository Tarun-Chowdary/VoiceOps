"""Deterministic policy calculator. The LLM never does date arithmetic or table lookups.

All numbers are PARSED FROM THE KB TEXT (REF-2, RET-2, RET-3, RET-8, RETv3-2, DWM-3) and the
holiday table, not hardcoded, so if the documents change, the agent follows. build_facts()
returns short, citable sentences that the generator must treat as verified.
"""
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import pandas as pd
from src import config

DAYNAME = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fmt(d: date) -> str:
    return f"{d.day} {d:%B %Y} ({DAYNAME[d.weekday()]})"


def dm(d: date) -> str:
    return f"{d.day} {d:%B}"


class Calendar:
    """Business day = Monday to Saturday, excluding warehouse holidays (KB-00.3)."""
    def __init__(self, holidays: set[date]):
        self.holidays = holidays

    @classmethod
    def load(cls):
        df = pd.read_csv(config.DATA_DIR / "holidays.csv", dtype=str)
        return cls({date.fromisoformat(x) for x in df["date"]})

    def is_business_day(self, d: date) -> bool:
        return d.weekday() != 6 and d not in self.holidays

    def add_business_days(self, start: date, n: int) -> date:
        d, k = start, 0
        while k < n:
            d += timedelta(days=1)
            if self.is_business_day(d):
                k += 1
        return d

    def business_days_between(self, a: date, b: date) -> int:
        """Business days in (a, b]; negative if b is before a."""
        if b < a:
            return -self.business_days_between(b, a)
        d, k = a, 0
        while d < b:
            d += timedelta(days=1)
            k += self.is_business_day(d)
        return k


# ---------- parsing the KB ----------
@dataclass
class RefundRule:
    label: str
    min_days: int
    max_days: int
    instant: bool


_REF_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", re.M)
_RET_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(Not returnable|(\d+)\s*days?)\s*\|\s*`([a-z_]+)`\s*\|", re.M | re.I)
_V3_ELEC = re.compile(r"^\|\s*Electronics[^|]*\|\s*(\d+)\s*days?\s*\|", re.M | re.I)


def parse_refund_rules(text: str) -> list[RefundRule]:
    rules = []
    for label, t in _REF_ROW.findall(text):
        m = re.search(r"(\d+)\s+to\s+(\d+)\s+business days", t, re.I)
        if m:
            rules.append(RefundRule(label, int(m.group(1)), int(m.group(2)), False))
        elif "instant" in t.lower():
            rules.append(RefundRule(label, 0, 0, True))
    return rules


def parse_return_windows(text: str) -> dict[str, tuple[str, int | None]]:
    return {code: (label, int(days) if days else None) for label, _, days, code in _RET_ROW.findall(text)}


def parse_v3_electronics_days(text: str) -> int | None:
    m = _V3_ELEC.search(text)
    return int(m.group(1)) if m else None


def parse_cutoff(text: str) -> date | None:
    m = re.search(r"From \*\*(\d{1,2} [A-Za-z]+ \d{4})\*\*", text)
    return datetime.strptime(m.group(1), "%d %B %Y").date() if m else None


def parse_festive(text: str):
    """RET-8: (window_start, window_end, days) for orders placed in the festive window."""
    m = re.search(r"placed between (\d{1,2}) ([A-Za-z]+) and (\d{1,2}) ([A-Za-z]+) (\d{4})", text)
    d = re.search(r"within (\d+) days", text)
    if not (m and d):
        return None
    y = m.group(5)
    s = datetime.strptime(f"{m.group(1)} {m.group(2)} {y}", "%d %B %Y").date()
    e = datetime.strptime(f"{m.group(3)} {m.group(4)} {y}", "%d %B %Y").date()
    return s, e, int(d.group(1))


def parse_reverse_fee(text: str):
    m = re.search(r"fee of ₹\s?(\d+) applies to change-of-mind returns of ([a-z, ]+?)\.", text, re.I)
    return (int(m.group(1)), m.group(2).strip().lower()) if m else None


def parse_refusal_days(text: str):
    m = re.search(r"usually (\d+) to (\d+) calendar days", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def parse_qc_days(text: str) -> int | None:
    m = re.search(r"within (\d+) business days of pickup", text)
    return int(m.group(1)) if m else None


def parse_high_value(text: str) -> int:
    m = re.search(r"more than ₹\s?([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else 10000


def rules_for_method(method: str | None, rules: list[RefundRule]) -> list[RefundRule]:
    m = (method or "").lower()
    pick = {
        "upi": lambda r: "upi" in r.label.lower(),
        "credit_card": lambda r: "card" in r.label.lower(),
        "debit_card": lambda r: "card" in r.label.lower(),
        "net_banking": lambda r: "net banking" in r.label.lower(),
        "wallet": lambda r: "wallet" in r.label.lower() and "cash" not in r.label.lower(),
        "cod": lambda r: "cash-on-delivery" in r.label.lower(),
        "card": lambda r: "card" in r.label.lower(),
        "netbanking": lambda r: "net banking" in r.label.lower(),
        "cod_bank": lambda r: "cash-on-delivery" in r.label.lower() and ("bank" in r.label.lower() or "neft" in r.label.lower()),
        "cod_wallet": lambda r: "cash-on-delivery" in r.label.lower() and "wallet" in r.label.lower(),
    }.get(m)
    return [r for r in rules if pick(r)] if pick else []


# ---------- facts for the generator ----------
def order_events(order: dict | None) -> list[dict]:
    """Dated events taken from a VERIFIED order record (authoritative over what the customer said)."""
    if not order:
        return []
    ev = []
    if order.get("delivered_on"):
        ev.append({"event": "delivered", "date": order["delivered_on"]})
    if order.get("placed_at"):
        ev.append({"event": "order_placed", "date": str(order["placed_at"])[:10]})
    for k in ("return_refund_initiated_on", "cancellation_refund_initiated_on", "rto_refund_initiated_on"):
        if order.get(k):
            ev.append({"event": "refund_initiated", "date": order[k]})
            break
    return ev


def order_refund_method(order: dict | None) -> str | None:
    if not order:
        return None
    return (order.get("return_refund_method") or order.get("cancellation_refund_method")
            or order.get("rto_refund_method") or order.get("payment_method"))


def _ev(und, name, extra=None):
    for e in list(extra or []) + list(getattr(und, "events", None) or []):
        if e.get("event") == name:
            try:
                return date.fromisoformat(e["date"])
            except Exception:
                return None
    return None


def build_facts_with_sources(und, prod, by_id: dict, cal: Calendar, today: date | None = None, order: dict | None = None):
    """Returns (facts, sections_consulted): the KB sections the code actually read to compute the facts."""
    today = today or config.TODAY
    src: list[str] = []
    oev = order_events(order)
    t = lambda sid: by_id[sid].text if sid in by_id else ""
    facts: list[str] = []
    intent = getattr(und, "intent", "")

    windows = parse_return_windows(t("RET-2"))
    hv = parse_high_value(t("DWM-3"))

    if prod:
        facts.append(f"Catalog match: {prod.title} (SKU {prod.sku}), return category `{prod.return_category}`, "
                     f"price ₹{prod.price_inr}, ZentroMart Assured = {prod.assured}, catalog warranty_months = {prod.warranty_months}.")
        try:
            if int(prod.price_inr) > hv:
                facts.append(f"This item's price is above the ₹{hv:,} high-value limit: damaged, defective or wrong-item "
                             f"claims must be handed to a human (DWM-3, ESC-1).")
                src += ["DWM-3", "ESC-1"]
        except ValueError:
            pass

    # ---- refunds ----
    if intent == "refund_timeline":
        method = (order_refund_method(order) if order else None) or und.payment_method
        rules = rules_for_method(method, parse_refund_rules(t("REF-2")))
        start = _ev(und, "refund_initiated", oev)
        if order and order.get("is_plus") and order.get("return_status") in (None, "requested"):
            facts.append("The caller is a ZentroMart Plus member: a returned item's refund is initiated at pickup instead of after QC (RET-7, PLS-2).")
            src.append("RET-7")
        if rules:
            src.append("REF-2")
        for r in rules:
            if r.instant:
                facts.append(f"REF-2: {r.label} refunds are instant (within 2 hours of initiation).")
            elif start:
                lo = cal.add_business_days(start, r.min_days)
                hi = cal.add_business_days(start, r.max_days)
                facts.append(f"REF-2: {r.label} refunds take {r.min_days} to {r.max_days} business days. Refund initiated "
                             f"{fmt(start)} → expected between {fmt(lo)} and {fmt(hi)}; the customer should have it by {fmt(hi)}. "
                             f"(Business days = Mon–Sat excluding warehouse holidays, KB-00.3.)")
                src.append("KB-00.3")
                late = cal.business_days_between(hi, today)
                if late > 0:
                    facts.append(f"Today is {fmt(today)}: the refund is {late} business day(s) past its maximum timeline."
                                 + (" That is more than 3, so ESC-1 requires a human handover." if late > 3 else ""))
                    if late > 3:
                        src.append("ESC-1")
            else:
                facts.append(f"REF-2: {r.label} refunds take {r.min_days} to {r.max_days} business days after the refund is initiated.")
        if not rules:
            facts.append("The payment method is not known: the refund timeline depends on it (REF-2), so ask for it.")
        for k, label in (("return_rrn", "return"), ("cancellation_rrn", "cancellation"), ("rto_rrn", "undelivered-parcel")):
            if order and order.get(k):
                facts.append(f"The {label} refund's reference number (RRN) is {order[k]}; REF-6 says to share it when the customer says the money has not arrived.")
                src.append("REF-6")
                break

    # ---- cancellations: which rule applies depends on the order's status ----
    if intent == "cancel_order" and order and order.get("status"):
        st = order["status"]
        before = st in ("placed", "packed")
        facts.append(f"Order status is '{st}', i.e. {'before' if before else 'after'} shipping, so {'CAN-1' if before else 'CAN-2'} applies.")
        src.append("CAN-1" if before else "CAN-2")
        if not before:
            pm, days = order.get("payment_method"), parse_refusal_days(t("REF-1"))
            if pm == "cod":
                facts.append("Cash-on-delivery order: if the customer refuses the parcel nothing is charged (CAN-2).")
            elif pm and days:
                facts.append(f"The order is prepaid ({pm}): if the customer refuses the parcel at the door, the refund follows REF-1, usually "
                             f"{days[0]} to {days[1]} calendar days after the last delivery attempt, once the parcel reaches the warehouse.")
                src.append("REF-1")

    # ---- a return already in progress: say where it is ----
    if intent == "refund_timeline" and order and order.get("return_status"):
        steps = [(lab, order.get(k)) for lab, k in (("requested", "return_requested_on"), ("picked up", "return_pickup_on"),
                 ("QC passed", "return_qc_passed_on"), ("refund initiated", "return_refund_initiated_on"),
                 ("refund completed", "return_refund_completed_on")) if order.get(k)]
        facts.append(f"Return progress: status '{order['return_status']}' (" + ", ".join(f"{a} {b}" for a, b in steps) + ").")
        if order.get("return_pickup_on") and not order.get("return_qc_passed_on") and not order.get("return_refund_initiated_on"):
            n = parse_qc_days(t("RET-7"))
            if order.get("is_plus"):
                facts.append("The item was picked up. The caller is a ZentroMart Plus member, so the refund is initiated at pickup (RET-7).")
            elif n:
                due = cal.add_business_days(date.fromisoformat(order["return_pickup_on"]), n)
                facts.append(f"The returned item was picked up and is at the warehouse quality check (QC). The refund is initiated only after it passes QC, "
                             f"within {n} business days of pickup, i.e. by {fmt(due)} (RET-7); refund timelines after initiation are in REF-2.")
            src.append("RET-7")

    # ---- returns ----
    if intent in ("return_window", "return_eligibility") and prod and prod.return_category:
        cat = prod.return_category
        label, days = windows.get(cat, (cat, None))
        if days is None and cat in windows:
            facts.append(f"RET-2/RET-5: category `{cat}` ({label}) cannot be returned for change of mind.")
        elif days is not None:
            delivered, placed = _ev(und, "delivered", oev), _ev(und, "order_placed", oev)
            cutoff = parse_cutoff(t("RET-3"))
            v3 = parse_v3_electronics_days(t("RETv3-2"))
            fest = parse_festive(t("RET-8"))
            src.append("RET-2")
            used, why = days, f"RET-2: {label} = {days} calendar days from delivery"
            if cat == "electronics" and cutoff and v3:
                src.append("RET-3")
                if delivered:
                    if delivered < cutoff:
                        src.append("RETv3-2")
                        used, why = v3, f"RET-3/RETv3-2: electronics delivered before {fmt(cutoff)} keep the older {v3}-day window"
                    else:
                        why = f"RET-2/RET-3: electronics delivered on or after {fmt(cutoff)} get {days} days"
                else:
                    why += f" (electronics delivered before {fmt(cutoff)} keep {v3} days under RET-3; delivery date unknown)"
            if fest and label.lower().startswith("clothing") and placed and fest[0] <= placed <= fest[1]:
                src.append("RET-8")
                used, why = fest[2], f"RET-8: clothing, footwear and bags ordered between {dm(fest[0])} and {dm(fest[1])} {fest[1].year} get {fest[2]} days"
            facts.append(f"Return window for this item: {used} calendar days from delivery ({why}).")
            fee = parse_reverse_fee(t("RET-6"))
            if fee and label.lower() == fee[1]:
                src.append("RET-6")
                if order and order.get("is_plus"):
                    facts.append("Reverse pickup fee: none, the caller is a ZentroMart Plus member (RET-6, PLS-2).")
                else:
                    facts.append(f"A reverse pickup fee of ₹{fee[0]} applies to this change-of-mind return (RET-6); it is waived for Plus members "
                                 f"and for damaged, defective or wrong items.")
            if delivered:
                dl = delivered + timedelta(days=used)
                facts.append(f"Delivered {fmt(delivered)} → return deadline {fmt(dl)}; today is {fmt(today)}, so the window is "
                             f"{'still open' if today <= dl else 'closed'}.")
    elif intent in ("return_window", "return_eligibility"):
        facts.append("No specific product was identified, so the return category is unknown: ask which item (one question).")
    return facts, list(dict.fromkeys(src))


def build_facts(und, prod, by_id: dict, cal: Calendar, today: date | None = None, order: dict | None = None) -> list[str]:
    return build_facts_with_sources(und, prod, by_id, cal, today, order)[0]


# ---------- ESC-1 triggers that code (not the LLM) must detect ----------
def escalation_reasons(order: dict | None, und, prod, by_id: dict, cal: Calendar, today: date | None = None) -> list[tuple[str, str]]:
    """Returns [(code, human-readable reason)] for ESC-1 conditions derivable from data."""
    today = today or config.TODAY
    t = lambda sid: by_id[sid].text if sid in by_id else ""
    out: list[tuple[str, str]] = []
    intent = getattr(und, "intent", "")
    if order and (order.get("contacts_last_7_days") or 0) >= 2:
        out.append(("repeat_contact", f"Third or later contact about the same issue within 7 days "
                                      f"(contacts_last_7_days={order['contacts_last_7_days']}, ESC-1)."))
    has_damage_record = bool(order and order.get("damage_sku"))
    if intent == "damaged_item" or (has_damage_record and intent in ("policy_override", "return_eligibility", "return_window")):
        hv = parse_high_value(t("DWM-3"))
        price = None
        if prod:
            try:
                price = int(prod.price_inr)
            except ValueError:
                pass
        if price is None and order:
            for it in order.get("items") or []:
                if it.get("sku") == order.get("damage_sku"):
                    price = it.get("unit_price_inr")
        if price is not None and price > hv:
            out.append(("high_value_damage", f"Damaged/defective/wrong item priced ₹{price} (> ₹{hv:,}) needs Priority Resolution (DWM-3, ESC-1)."))
    if order and intent in ("refund_timeline", "return_window", "return_eligibility", "other") and order.get("return_refund_initiated_on") and not order.get("return_refund_completed_on"):
        rules = rules_for_method(order.get("return_refund_method"), parse_refund_rules(t("REF-2")))
        if rules:
            r = max(rules, key=lambda x: x.max_days)
            start = date.fromisoformat(order["return_refund_initiated_on"])
            late = cal.business_days_between(cal.add_business_days(start, r.max_days), today)
            if late > 3:
                out.append(("refund_overdue", f"Return refund initiated {order['return_refund_initiated_on']} is {late} business days "
                                              f"past its maximum {r.max_days}-business-day timeline (REF-2, ESC-1)."))
    return out
