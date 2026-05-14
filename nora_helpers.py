"""Shared runtime helpers for the Nora debt-collection agent.

- Najdi Arabic number/date pronunciation helpers.
- Payment-ladder policy: pure function the `evaluate_offer` tool wraps.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

_AR_UNITS = ["", "واحد", "اثنين", "ثلاثة", "أربعة", "خمسة",
             "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"]
_AR_TEENS = ["عشرة", "أحد عشر", "اثنا عشر", "ثلاثة عشر", "أربعة عشر",
             "خمسة عشر", "ستة عشر", "سبعة عشر", "ثمانية عشر", "تسعة عشر"]
_AR_TENS = ["", "", "عشرين", "ثلاثين", "أربعين", "خمسين",
            "ستين", "سبعين", "ثمانين", "تسعين"]
_AR_HUNDREDS = ["", "مية", "مئتين", "ثلاث مية", "أربع مية", "خمس مية",
                "ست مية", "سبع مية", "ثمان مية", "تسع مية"]
_AR_MONTHS = ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
              "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]
_AR_WEEKDAYS = ["الإثنين", "الثلاثاء", "الأربعاء", "الخميس",
                "الجمعة", "السبت", "الأحد"]


def _ar_below_1000(n: int) -> str:
    if n == 0:
        return ""
    parts: list[str] = []
    h, rem = divmod(n, 100)
    if h:
        parts.append(_AR_HUNDREDS[h])
    if rem:
        if rem < 10:
            parts.append(_AR_UNITS[rem])
        elif rem < 20:
            parts.append(_AR_TEENS[rem - 10])
        else:
            t, u = divmod(rem, 10)
            if u:
                parts.append(f"{_AR_UNITS[u]} و{_AR_TENS[t]}")
            else:
                parts.append(_AR_TENS[t])
    return " و".join(parts)


def ar_amount_words(n: int) -> str:
    """Speak a SAR integer amount the way a Najdi Saudi would say it."""
    n = int(round(n))
    if n == 0:
        return "صفر"
    parts: list[str] = []
    millions, n = divmod(n, 1_000_000)
    thousands, n = divmod(n, 1_000)
    if millions:
        if millions == 1:
            parts.append("مليون")
        elif millions == 2:
            parts.append("مليونين")
        elif 3 <= millions <= 10:
            parts.append(f"{_AR_UNITS[millions]} ملايين")
        else:
            parts.append(f"{_ar_below_1000(millions)} مليون")
    if thousands:
        if thousands == 1:
            parts.append("ألف")
        elif thousands == 2:
            parts.append("ألفين")
        elif 3 <= thousands <= 10:
            parts.append(f"{_AR_UNITS[thousands]} آلاف")
        else:
            parts.append(f"{_ar_below_1000(thousands)} ألف")
    if n:
        parts.append(_ar_below_1000(n))
    return " و".join(parts)


def ar_date_words(d: datetime.date) -> str:
    """Day + month name in Arabic (e.g. 'خمسة عشر مايو')."""
    return f"{_ar_below_1000(d.day)} {_AR_MONTHS[d.month - 1]}"


def ar_full_date(d: datetime.date) -> str:
    """Weekday + day + month, Arabic spoken form."""
    return f"{_AR_WEEKDAYS[d.weekday()]} {ar_date_words(d)}"


# ---------- Payment-ladder policy ----------

# Rungs (descending). Each = (label, minimum fraction of remaining).
_LADDER = [
    ("full", 1.00),
    ("half", 0.50),
    ("ten_pct", 0.10),
    ("five_pct", 0.05),
]
MIN_ACCEPTABLE_FRACTION = 0.05  # 5% of original total per single payment
# Tawafuq policy: full recovery within 3 months from the start of the next month
# after the debtor sheet is received, with a 2-week safety margin. Until that
# dataset-driven date is wired in, default to (3 months − 2 weeks) = 76 days.
# build_payment_context() honors `meta["plan_deadline_iso"]` if upstream supplies it.
MAX_PLAN_DAYS = 76


@dataclass
class OfferEvaluation:
    decision: str                 # "accept" | "below_threshold" | "reject_too_low"
    rung: str                     # "full" | "half" | "ten_pct" | "five_pct" | "below"
    offer_pct_of_total: float
    offer_pct_of_remaining: float
    remaining_after_sar: float
    is_full_settlement: bool
    counter_floor_sar: float | None  # amount to propose in the gentle push


def evaluate_offer(
    amount_sar: float,
    total_due_sar: float,
    already_committed_sar: float = 0.0,
) -> OfferEvaluation:
    """Pure-policy evaluation of one proposed payment."""
    remaining = max(0.0, total_due_sar - already_committed_sar)
    pct_total = (amount_sar / total_due_sar) if total_due_sar > 0 else 0.0
    if remaining <= 0:
        return OfferEvaluation(
            decision="accept",
            rung="full",
            offer_pct_of_total=100.0,
            offer_pct_of_remaining=100.0,
            remaining_after_sar=0.0,
            is_full_settlement=True,
            counter_floor_sar=None,
        )
    pct = amount_sar / remaining
    rung = "below"
    for label, frac in _LADDER:
        if pct >= frac - 1e-9:
            rung = label
            break
    counter_target = round(total_due_sar * 0.25, 2)
    is_full = amount_sar >= remaining * 0.99
    is_first_installment = already_committed_sar <= 1e-9

    # Three-tier policy aligned with the PDF's "first installment vs total"
    # thresholds, with a hard floor at MIN_ACCEPTABLE_FRACTION (5%) — anything
    # below the floor is not lockable, only escalatable.
    #
    # Subsequent installments (after a first lock) just need to be positive,
    # since they're filling the remainder of an already-agreed plan.
    if is_full:
        decision = "accept"
        counter = None
    elif not is_first_installment:
        decision = "accept" if amount_sar > 0 else "reject_too_low"
        counter = None if amount_sar > 0 else counter_target
    elif pct_total >= 0.25 - 1e-9:
        decision = "accept"
        counter = None
    elif pct_total >= MIN_ACCEPTABLE_FRACTION - 1e-9:
        # 5%–25%: PDF says one gentle push, then lock with follow-up if refused.
        decision = "below_threshold"
        counter = counter_target
    else:
        # < 5%: token offer. Refuse to lock; force escalation.
        decision = "reject_too_low"
        counter = counter_target

    return OfferEvaluation(
        decision=decision,
        rung=rung,
        offer_pct_of_total=round(pct_total * 100, 1),
        offer_pct_of_remaining=round(pct * 100, 1),
        remaining_after_sar=round(max(0.0, remaining - amount_sar), 2),
        is_full_settlement=is_full,
        counter_floor_sar=counter,
    )


def build_payment_context(meta: dict) -> dict:
    """Compute everything the prompt template needs from the dispatch metadata.

    Returns a dict with raw numbers + pre-formatted Arabic strings + reference
    dates the model can copy verbatim instead of doing math at runtime.
    """
    try:
        total = float(str(meta.get("amount", "0")).replace(",", "").strip() or 0)
    except ValueError:
        total = 0.0
    total_int = int(round(total))
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    in_one_week = today + datetime.timedelta(days=7)
    # Per-company collection deadline: prefer the date the dispatcher supplied
    # (debtor sheet end-date), otherwise fall back to today + MAX_PLAN_DAYS.
    deadline_override = meta.get("plan_deadline_iso") or meta.get("plan_end_date")
    if deadline_override:
        try:
            plan_deadline = datetime.date.fromisoformat(str(deadline_override).strip())
        except ValueError:
            plan_deadline = today + datetime.timedelta(days=MAX_PLAN_DAYS)
    else:
        plan_deadline = today + datetime.timedelta(days=MAX_PLAN_DAYS)

    return {
        "total_due_sar": total,
        "total_words": ar_amount_words(total_int),
        "half_words": ar_amount_words(int(round(total * 0.50))),
        "ten_pct_words": ar_amount_words(int(round(total * 0.10))),
        "five_pct_words": ar_amount_words(int(round(total * 0.05))),
        "half_sar": int(round(total * 0.50)),
        "ten_pct_sar": int(round(total * 0.10)),
        "five_pct_sar": int(round(total * 0.05)),
        "today_iso": today.isoformat(),
        "today_words": ar_full_date(today),
        "tomorrow_iso": tomorrow.isoformat(),
        "tomorrow_words": ar_full_date(tomorrow),
        "in_one_week_iso": in_one_week.isoformat(),
        "in_one_week_words": ar_full_date(in_one_week),
        "plan_deadline_iso": plan_deadline.isoformat(),
        "plan_deadline_words": ar_full_date(plan_deadline),
    }
