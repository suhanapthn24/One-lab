"""Seeded synthetic ledger and bank data for October 2026, with optional planted gaps."""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pandas as pd

from recon.bizdays import add_business_days, to_bank_amount

YEAR, MONTH = 2026, 10
N_PAYMENTS = 300
N_REFUNDS = 5
N_CUSTOMERS = 120
# 80 lines x 0.0040..0.0049 => total drift 0.32..0.392. (40 lines cannot reach 0.30
# while each stays under 0.005, so the line count was raised to hit the target total.)
N_DRIFT = 80
LAST_CLEAN_DAY = 28  # clean rows settle in October even with a 2-business-day lag

LEDGER_COLUMNS = ["txn_id", "created_at", "amount", "type", "original_txn_id", "customer_id"]
BANK_COLUMNS = ["settlement_id", "settled_at", "amount", "reference"]

# Small reference set of September transaction ids that October refunds may point at.
PRIOR_MONTH_TXN_IDS: frozenset[str] = frozenset(
    f"TXN-2609-{n:05d}" for n in (17, 42, 88, 113, 161, 204, 259, 287)
)
DECOY_ORIGINAL_TXN_ID = "TXN-2609-00113"
# Exists nowhere: not in October, not in the September reference set.
ORPHAN_ORIGINAL_TXN_ID = "TXN-2608-00999"

# Filled by generate_data(plant_gaps=True) with the txn_ids of each planted item.
# Keys: timing, rounding (list), duplicate, duplicate_of, orphan_refund, decoy_refund.
PLANTED: dict[str, object] = {}


def _ts(rng: random.Random, day: int) -> datetime:
    """A random UTC timestamp on the given October day."""
    seconds = rng.randint(0, 24 * 3600 - 1)
    return datetime(YEAR, MONTH, day, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def _amount(cents: int, sub_cent: int) -> Decimal:
    """Build a 4dp amount from whole cents plus a signed sub-cent offset (in 0.0001 units)."""
    return Decimal(cents * 100 + sub_cent).scaleb(-4)


def _settle_at(created: datetime, lag: int) -> datetime:
    """Settlement timestamp: `lag` business days after creation, at 04:30 UTC."""
    d: date = add_business_days(created.date(), lag)
    return datetime.combine(d, time(4, 30), tzinfo=timezone.utc)


def generate_data(seed: int, plant_gaps: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ledger_df, bank_df) for October 2026. Same seed => identical frames.

    Money columns hold `decimal.Decimal` objects (object dtype); timestamps are UTC.
    """
    rng = random.Random(seed)
    PLANTED.clear()
    customers = [f"CUST-{i:04d}" for i in range(1, N_CUSTOMERS + 1)]

    def record(kind: str, created: datetime, cents: int, sub: int = 0, *,
               customer: str | None = None, original: object = None,
               settles: bool = True, tag: str | None = None) -> dict:
        return {
            "type": kind, "created_at": created, "cents": cents, "sub": sub,
            "customer_id": customer or rng.choice(customers), "original": original,
            "settles": settles, "lag": rng.randint(1, 2), "tag": tag,
        }

    # --- Clean payments -------------------------------------------------------
    payments = [
        record("payment", _ts(rng, rng.randint(1, LAST_CLEAN_DAY)), rng.randint(10_000, 5_000_000))
        for _ in range(N_PAYMENTS)
    ]

    order = list(range(N_PAYMENTS))
    rng.shuffle(order)
    drift_idx = order[:N_DRIFT] if plant_gaps else []
    pair_idx = order[N_DRIFT:] if plant_gaps else order
    for i in drift_idx:
        payments[i]["sub"] = rng.randint(40, 49)  # rounds down: ledger exceeds bank by < 0.005
        payments[i]["tag"] = "rounding"
    # Clean sub-cent noise in cancelling pairs: +k rounds down, -k rounds up, net zero.
    for a, b in zip(pair_idx[0::2], pair_idx[1::2]):
        k = rng.randint(1, 49)
        payments[a]["sub"], payments[b]["sub"] = k, -k

    # --- Clean refunds of October payments ------------------------------------
    candidates = [p for p in payments if p["created_at"].day <= LAST_CLEAN_DAY - 4]
    refunds = []
    for original in rng.sample(candidates, N_REFUNDS):
        day = min(original["created_at"].day + rng.randint(1, 3), LAST_CLEAN_DAY)
        rupees = rng.randint(1, max(1, original["cents"] // 100))
        refunds.append(record("refund", _ts(rng, day), -rupees * 100,
                              customer=original["customer_id"], original=original))

    records = payments + refunds

    # --- Planted gaps -----------------------------------------------------------
    if plant_gaps:
        records.append(record("payment", _ts(rng, rng.choice([30, 31])),
                              rng.randint(10_000, 5_000_000), tag="timing"))

        dup_src = record("payment", _ts(rng, rng.randint(3, 25)),
                         rng.randint(10_000, 5_000_000), tag="duplicate_of")
        dup_copy = dict(dup_src, settles=False, tag="duplicate")
        records += [dup_src, dup_copy]

        records.append(record("refund", _ts(rng, rng.randint(5, 25)),
                              -rng.randint(5, 500) * 100, original=ORPHAN_ORIGINAL_TXN_ID,
                              settles=False, tag="orphan_refund"))

        records.append(record("refund", _ts(rng, rng.randint(2, 20)),
                              -rng.randint(5, 500) * 100, original=DECOY_ORIGINAL_TXN_ID,
                              tag="decoy_refund"))

    # --- Assign ids in creation order (stable: a duplicate copy follows its source) ---
    records.sort(key=lambda r: r["created_at"])
    for n, rec in enumerate(records, start=1):
        rec["txn_id"] = f"TXN-2610-{n:05d}"
        rec["amount"] = _amount(rec["cents"], rec["sub"])

    PLANTED["rounding"] = []
    for rec in records:
        tag = rec["tag"]
        if tag == "rounding":
            PLANTED["rounding"].append(rec["txn_id"])
        elif tag:
            PLANTED[tag] = rec["txn_id"]
    if not plant_gaps:
        PLANTED.clear()

    ledger_rows = []
    bank_rows = []
    for rec in records:
        original = rec["original"]
        original_id = original["txn_id"] if isinstance(original, dict) else original
        ledger_rows.append({
            "txn_id": rec["txn_id"], "created_at": rec["created_at"], "amount": rec["amount"],
            "type": rec["type"], "original_txn_id": original_id, "customer_id": rec["customer_id"],
        })
        if not rec["settles"]:
            continue
        settled = _settle_at(rec["created_at"], rec["lag"])
        if (settled.year, settled.month) != (YEAR, MONTH):
            continue  # settles next month: not in the October bank file
        bank_rows.append({"settled_at": settled, "amount": to_bank_amount(rec["amount"]),
                          "reference": rec["txn_id"]})

    bank_rows.sort(key=lambda r: (r["settled_at"], r["reference"]))
    for n, row in enumerate(bank_rows, start=1):
        row["settlement_id"] = f"STL-2610-{n:05d}"

    ledger_df = _frame(ledger_rows, LEDGER_COLUMNS, "created_at")
    bank_df = _frame(bank_rows, BANK_COLUMNS, "settled_at")
    return ledger_df, bank_df


def _frame(rows: list[dict], columns: list[str], ts_col: str) -> pd.DataFrame:
    """Build a DataFrame keeping `amount` as Decimal objects and timestamps as UTC."""
    df = pd.DataFrame(rows, columns=columns)
    df["amount"] = pd.Series([r["amount"] for r in rows], dtype=object)
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    return df
