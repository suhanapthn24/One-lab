"""Business-day and money helpers shared by the generator and the matcher."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")
BASIS = Decimal("0.0001")
ZERO = Decimal("0")


def to_bank_amount(amount: Decimal) -> Decimal:
    """Round a 4dp ledger amount to the bank's 2dp, half-up (away from zero on .5)."""
    return amount.quantize(CENT, rounding=ROUND_HALF_UP)


def is_business_day(d: date) -> bool:
    """Monday-Friday are business days (no bank holidays modelled)."""
    return d.weekday() < 5


def add_business_days(d: date, n: int) -> date:
    """Return the date `n` business days after `d`, skipping weekends.

    A Saturday or Sunday start rolls forward: Sat + 1 business day is Monday.
    """
    current = d
    remaining = n
    while remaining > 0:
        current += timedelta(days=1)
        if is_business_day(current):
            remaining -= 1
    return current


def business_days_between(start: date, end: date) -> int:
    """Number of business days strictly after `start` up to and including `end`.

    Negative if `end` is before `start`.
    """
    if end < start:
        return -business_days_between(end, start)
    count = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if is_business_day(current):
            count += 1
    return count


def last_business_days(year: int, month: int, n: int) -> list[date]:
    """The final `n` business days of a calendar month, in ascending order."""
    first_next = date(year + (month == 12), month % 12 + 1, 1)
    days: list[date] = []
    current = first_next - timedelta(days=1)
    while len(days) < n:
        if is_business_day(current):
            days.append(current)
        current -= timedelta(days=1)
    return sorted(days)
