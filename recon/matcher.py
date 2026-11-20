"""Tiered ledger-vs-bank matching, gap classification and the bridge."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, time, timezone
from decimal import Decimal
from typing import Iterable

import pandas as pd

from recon.bizdays import (ZERO, business_days_between, last_business_days,
                           to_bank_amount)
from recon.models import ReconResult

AMOUNT_TOLERANCE = Decimal("0.005")
FUZZY_MAX_BUSINESS_DAYS = 3
TIMING_WINDOW_BUSINESS_DAYS = 3

# Fixed confidence for rule-based classifications (how sure we are of the label).
CONFIDENCE = {
    "timing": 0.9,
    "duplicate": 0.95,
    "orphan_refund": 0.9,
    "rounding": 1.0,
    "unmatched": 0.5,
}


def _norm_ref(value: object) -> str:
    """Normalise a reference / txn_id for fuzzy comparison (case, spaces, dashes)."""
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _dec(value: object) -> Decimal:
    """Coerce a money cell to Decimal without ever passing through float."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        raise TypeError(f"float money value {value!r}: use Decimal")
    return Decimal(str(value))


def _period(ledger: pd.DataFrame, period: tuple[int, int] | None) -> tuple[int, int]:
    """The reconciliation month: explicit, else the most common month in the ledger."""
    if period is not None:
        return period
    months = Counter((ts.year, ts.month) for ts in ledger["created_at"])
    return months.most_common(1)[0][0]


def reconcile(
    ledger_df: pd.DataFrame,
    bank_df: pd.DataFrame,
    prior_month_txn_ids: Iterable[str] | None = None,
    period: tuple[int, int] | None = None,
) -> ReconResult:
    """Match ledger rows to bank rows, classify what is left, and build the bridge.

    Gap amounts are the signed effect on (ledger total - bank total), so
    ledger_total - bank_total == sum(gap amounts) + unexplained.
    """
    prior = set(prior_month_txn_ids or ())
    ledger = ledger_df.reset_index(drop=True).to_dict("records")
    bank = bank_df.reset_index(drop=True).to_dict("records")
    for row in ledger + bank:
        row["amount"] = _dec(row["amount"])

    matches: list[dict] = []
    ledger_used: set[int] = set()
    bank_used: set[int] = set()

    # --- Tier 1: exact (reference == txn_id, amount equal at 2dp) -------------
    bank_by_ref: dict[str, list[int]] = defaultdict(list)
    for j, b in enumerate(bank):
        bank_by_ref[str(b["reference"])].append(j)
    for i, l in enumerate(ledger):
        target = to_bank_amount(l["amount"])
        for j in bank_by_ref.get(str(l["txn_id"]), []):
            if j not in bank_used and bank[j]["amount"] == target:
                matches.append(_match(l, bank[j], "exact", 1.0))
                ledger_used.add(i)
                bank_used.add(j)
                break

    # --- Tier 2: fuzzy on the leftovers -----------------------------------------
    customer_of = {_norm_ref(l["txn_id"]): l["customer_id"] for l in ledger}
    candidates = []
    for i, l in enumerate(ledger):
        if i in ledger_used:
            continue
        for j, b in enumerate(bank):
            if j in bank_used:
                continue
            score = _fuzzy_score(l, b, customer_of)
            if score is not None:
                candidates.append((score, i, j))
    for score, i, j in sorted(candidates, key=lambda c: (-c[0], c[1], c[2])):
        if i in ledger_used or j in bank_used:
            continue
        matches.append(_match(ledger[i], bank[j], "fuzzy", score))
        ledger_used.add(i)
        bank_used.add(j)

    # --- Tier 3: classify leftovers ---------------------------------------------
    year, month = _period(ledger_df, period)
    window_start = datetime.combine(
        last_business_days(year, month, TIMING_WINDOW_BUSINESS_DAYS)[0], time(0), tzinfo=timezone.utc
    )
    ledger_ids = {str(l["txn_id"]) for l in ledger}

    # Duplicate groups: identical customer, amount and timestamp. The matched row (or
    # the first one) is the keeper; every other unmatched copy is a duplicate.
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, l in enumerate(ledger):
        groups[(l["customer_id"], l["amount"], l["created_at"])].append(i)
    duplicate_of: dict[int, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        keeper = next((i for i in members if i in ledger_used), members[0])
        for i in members:
            if i != keeper and i not in ledger_used:
                duplicate_of[i] = str(ledger[keeper]["txn_id"])

    gaps: list[dict] = []
    for i, l in enumerate(ledger):
        if i in ledger_used:
            continue
        txn_id, amount = str(l["txn_id"]), l["amount"]
        original = l.get("original_txn_id")
        has_original = isinstance(original, str) and original != ""
        if i in duplicate_of:
            gaps.append(_gap("duplicate", txn_id, amount,
                             f"Same customer, amount and timestamp as {duplicate_of[i]}; "
                             "only one bank settlement exists, so this copy is a double entry."))
        elif l["type"] == "refund" and has_original and original not in ledger_ids and original not in prior:
            gaps.append(_gap("orphan_refund", txn_id, amount,
                             f"Refund points at {original}, which is not in this month's ledger "
                             "or the prior-month reference set, and has no bank settlement."))
        elif l["created_at"] >= window_start:
            gaps.append(_gap("timing", txn_id, amount,
                             f"Created {l['created_at']:%a %d %b %H:%M} UTC, inside the last "
                             f"{TIMING_WINDOW_BUSINESS_DAYS} business days; settlement falls in "
                             "the next month's bank file."))
        else:
            gaps.append(_gap("unmatched", txn_id, amount,
                             "Ledger entry has no bank settlement and fits no known gap pattern."))

    for j, b in enumerate(bank):
        if j in bank_used:
            continue
        gaps.append(_gap("unmatched", str(b["reference"]), -b["amount"],
                         f"Bank settlement {b['settlement_id']} has no matching ledger entry."))

    # --- Tier 4: rounding drift across matched pairs ------------------------------
    drift = sum((m["difference"] for m in matches), ZERO)
    if drift != ZERO:
        lines = sum(1 for m in matches if m["difference"] != ZERO)
        worst = max(abs(m["difference"]) for m in matches)
        gaps.append(_gap("rounding", None, drift,
                         f"{lines} matched lines differ by at most {worst} each (within the "
                         f"{AMOUNT_TOLERANCE} per-line tolerance) but sum to {drift} because the "
                         "bank rounds every 4dp ledger amount to 2dp."))

    # --- Bridge -----------------------------------------------------------------
    ledger_total = sum((l["amount"] for l in ledger), ZERO)
    bank_total = sum((b["amount"] for b in bank), ZERO)
    difference = ledger_total - bank_total
    explained = sum((g["amount"] for g in gaps), ZERO)
    bridge = {
        "ledger_total": ledger_total,
        "bank_total": bank_total,
        "difference": difference,
        "explained": explained,
        "unexplained": difference - explained,
    }
    return ReconResult(gaps=gaps, bridge=bridge, matches=matches)


def _fuzzy_score(l: dict, b: dict, customer_of: dict[str, str]) -> float | None:
    """Confidence (0-1) that bank row `b` settles ledger row `l`, or None if ineligible.

    Eligible when amounts are within tolerance, settlement is 0-3 business days after
    creation, and the reference matches the txn_id (after normalising) or points at a
    ledger entry of the same customer.
    """
    diff = abs(l["amount"] - b["amount"])
    if diff >= AMOUNT_TOLERANCE:
        return None
    lag = business_days_between(l["created_at"].date(), b["settled_at"].date())
    if not 0 <= lag <= FUZZY_MAX_BUSINESS_DAYS:
        return None
    ref = _norm_ref(b["reference"])
    if ref == _norm_ref(l["txn_id"]):
        ref_score = 1.0
    elif customer_of.get(ref) == l["customer_id"]:
        ref_score = 0.6
    else:
        return None
    amount_score = 1.0 - float(diff / AMOUNT_TOLERANCE)  # a score, not money
    lag_score = 1.0 if 1 <= lag <= 2 else 0.5
    return round(0.5 * ref_score + 0.3 * amount_score + 0.2 * lag_score, 3)


def _match(l: dict, b: dict, method: str, confidence: float) -> dict:
    return {
        "txn_id": str(l["txn_id"]),
        "settlement_id": str(b["settlement_id"]),
        "method": method,
        "confidence": confidence,
        "ledger_amount": l["amount"],
        "bank_amount": b["amount"],
        "difference": l["amount"] - b["amount"],
    }


def _gap(gap_type: str, txn_id: str | None, amount: Decimal, reason: str) -> dict:
    return {"type": gap_type, "txn_id": txn_id, "amount": amount,
            "reason": reason, "confidence": CONFIDENCE[gap_type]}
