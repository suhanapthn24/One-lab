"""Tests for the generator, the matcher and the bridge."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from recon import generator
from recon.bizdays import add_business_days, to_bank_amount
from recon.generator import PLANTED, PRIOR_MONTH_TXN_IDS, generate_data
from recon.matcher import AMOUNT_TOLERANCE, reconcile

SEED = 42


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def ledger_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=generator.LEDGER_COLUMNS)
    df["amount"] = pd.Series([Decimal(r["amount"]) for r in rows], dtype=object)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df


def bank_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=generator.BANK_COLUMNS)
    df["amount"] = pd.Series([Decimal(r["amount"]) for r in rows], dtype=object)
    df["settled_at"] = pd.to_datetime(df["settled_at"], utc=True)
    return df


def payment(txn_id, created, amount, customer, original=None, kind="payment"):
    return {"txn_id": txn_id, "created_at": created, "amount": amount, "type": kind,
            "original_txn_id": original, "customer_id": customer}


def settlement(sid, settled, amount, reference):
    return {"settlement_id": sid, "settled_at": settled, "amount": amount, "reference": reference}


@pytest.fixture(scope="module")
def planted_run():
    ledger, bank = generate_data(SEED, plant_gaps=True)
    planted = {k: (list(v) if isinstance(v, list) else v) for k, v in PLANTED.items()}
    result = reconcile(ledger, bank, prior_month_txn_ids=PRIOR_MONTH_TXN_IDS)
    return ledger, bank, planted, result


def gaps_by_txn(result) -> dict:
    return {g["txn_id"]: g for g in result.gaps if g["txn_id"] is not None}


# --- Planted gaps ---------------------------------------------------------------

def test_planted_ids_recorded(planted_run):
    _, _, planted, _ = planted_run
    for key in ("timing", "duplicate", "duplicate_of", "orphan_refund", "decoy_refund"):
        assert isinstance(planted[key], str)
    assert len(planted["rounding"]) == generator.N_DRIFT


def test_timing_gap_detected(planted_run):
    ledger, bank, planted, result = planted_run
    txn = planted["timing"]
    created = ledger.loc[ledger.txn_id == txn, "created_at"].iloc[0]
    assert created.day in (30, 31)
    assert txn not in set(bank.reference)
    gap = gaps_by_txn(result)[txn]
    assert gap["type"] == "timing"
    assert gap["amount"] == ledger.loc[ledger.txn_id == txn, "amount"].iloc[0]


def test_duplicate_gap_detected(planted_run):
    ledger, bank, planted, result = planted_run
    dup, src = planted["duplicate"], planted["duplicate_of"]
    a = ledger.set_index("txn_id").loc[dup]
    b = ledger.set_index("txn_id").loc[src]
    assert (a.customer_id, a.amount, a.created_at) == (b.customer_id, b.amount, b.created_at)
    assert list(bank.reference).count(src) == 1 and dup not in set(bank.reference)
    gaps = gaps_by_txn(result)
    assert gaps[dup]["type"] == "duplicate"
    assert src not in gaps
    assert [g["txn_id"] for g in result.gaps if g["type"] == "duplicate"] == [dup]


def test_orphan_refund_detected(planted_run):
    ledger, bank, planted, result = planted_run
    txn = planted["orphan_refund"]
    row = ledger.set_index("txn_id").loc[txn]
    assert row.type == "refund" and row.amount < 0
    assert row.original_txn_id not in set(ledger.txn_id)
    assert row.original_txn_id not in PRIOR_MONTH_TXN_IDS
    gap = gaps_by_txn(result)[txn]
    assert gap["type"] == "orphan_refund"
    assert gap["amount"] == row.amount


def test_rounding_gap_detected(planted_run):
    _, _, planted, result = planted_run
    rounding = [g for g in result.gaps if g["type"] == "rounding"]
    assert len(rounding) == 1
    assert Decimal("0.30") <= rounding[0]["amount"] <= Decimal("0.50")


def test_exact_gap_set(planted_run):
    _, _, planted, result = planted_run
    kinds = sorted(g["type"] for g in result.gaps)
    assert kinds == ["duplicate", "orphan_refund", "rounding", "timing"]


def test_decoy_refund_not_flagged(planted_run):
    ledger, bank, planted, result = planted_run
    txn = planted["decoy_refund"]
    row = ledger.set_index("txn_id").loc[txn]
    assert row.type == "refund" and row.original_txn_id in PRIOR_MONTH_TXN_IDS
    assert row.original_txn_id not in set(ledger.txn_id)
    assert txn in set(bank.reference)
    assert txn not in gaps_by_txn(result)
    assert txn in {m["txn_id"] for m in result.matches}


def test_prior_month_refund_without_bank_row_is_not_orphan():
    """Classification itself honours the prior-month set, not just matching."""
    sept = sorted(PRIOR_MONTH_TXN_IDS)[0]
    ledger = ledger_frame([
        payment("R1", utc(2026, 10, 12, 9), "-100.0000", "C1", original=sept, kind="refund"),
        payment("R2", utc(2026, 10, 12, 9), "-50.0000", "C2", original="TXN-NOWHERE", kind="refund"),
    ])
    bank = bank_frame([])
    result = reconcile(ledger, bank, prior_month_txn_ids=PRIOR_MONTH_TXN_IDS, period=(2026, 10))
    types = {g["txn_id"]: g["type"] for g in result.gaps}
    assert types == {"R1": "unmatched", "R2": "orphan_refund"}


# --- Bridge and types -----------------------------------------------------------

def test_bridge_closes_exactly(planted_run):
    ledger, bank, _, result = planted_run
    br = result.bridge
    for value in br.values():
        assert isinstance(value, Decimal)
    assert br["ledger_total"] == sum(ledger.amount, Decimal("0"))
    assert br["bank_total"] == sum(bank.amount, Decimal("0"))
    assert br["difference"] == br["ledger_total"] - br["bank_total"]
    assert br["explained"] == sum((g["amount"] for g in result.gaps), Decimal("0"))
    assert br["ledger_total"] - br["bank_total"] == br["explained"] + br["unexplained"]
    assert br["unexplained"] == Decimal("0")
    assert br["difference"] != Decimal("0")


def test_all_gap_amounts_are_decimal(planted_run):
    _, _, _, result = planted_run
    assert result.gaps
    for gap in result.gaps:
        assert type(gap["amount"]) is Decimal
        assert 0.0 <= gap["confidence"] <= 1.0


def test_no_floats_in_money_columns(planted_run):
    ledger, bank, _, _ = planted_run
    assert ledger.amount.dtype == object and bank.amount.dtype == object
    assert all(type(v) is Decimal for v in ledger.amount)
    assert all(type(v) is Decimal for v in bank.amount)
    assert all(v.as_tuple().exponent == -4 for v in ledger.amount)
    assert all(v.as_tuple().exponent == -2 for v in bank.amount)


def test_rounding_passes_per_line_fails_in_aggregate(planted_run):
    ledger, bank, planted, result = planted_run
    bank_by_ref = bank.set_index("reference").amount
    ledger_by_id = ledger.set_index("txn_id").amount
    diffs = [ledger_by_id[t] - bank_by_ref[t] for t in planted["rounding"]]
    assert all(Decimal("0") < d < AMOUNT_TOLERANCE for d in diffs)
    total = sum(diffs, Decimal("0"))
    assert total >= AMOUNT_TOLERANCE
    assert Decimal("0.30") <= total <= Decimal("0.50")
    # Every matched line individually passes tolerance too.
    assert all(abs(m["difference"]) < AMOUNT_TOLERANCE for m in result.matches)
    rounding_gap = next(g for g in result.gaps if g["type"] == "rounding")
    assert rounding_gap["amount"] == total  # clean rows' sub-cent noise nets to zero


# --- Duplicates: different customers are not duplicates -------------------------

def test_same_amount_same_day_different_customers_not_duplicates():
    ts = utc(2026, 10, 14, 10, 15)
    ledger = ledger_frame([
        payment("T1", ts, "999.0000", "CUST-A"),
        payment("T2", ts, "999.0000", "CUST-B"),
        payment("T3", utc(2026, 10, 14, 16), "999.0000", "CUST-C"),
    ])
    bank = bank_frame([
        settlement("S1", utc(2026, 10, 15, 4, 30), "999.00", "T1"),
        settlement("S2", utc(2026, 10, 15, 4, 30), "999.00", "T2"),
        settlement("S3", utc(2026, 10, 16, 4, 30), "999.00", "T3"),
    ])
    result = reconcile(ledger, bank)
    assert result.gaps == []
    assert result.matched_count == 3
    assert result.bridge["unexplained"] == Decimal("0")


def test_same_amount_different_customers_one_unsettled_is_not_duplicate():
    ts = utc(2026, 10, 14, 10, 15)
    ledger = ledger_frame([
        payment("T1", ts, "999.0000", "CUST-A"),
        payment("T2", ts, "999.0000", "CUST-B"),
    ])
    bank = bank_frame([settlement("S1", utc(2026, 10, 15, 4, 30), "999.00", "T1")])
    result = reconcile(ledger, bank, period=(2026, 10))
    assert [(g["txn_id"], g["type"]) for g in result.gaps] == [("T2", "unmatched")]


# --- Clean data --------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 42, 2026, 99999])
def test_clean_data_has_zero_gaps(seed):
    ledger, bank = generate_data(seed, plant_gaps=False)
    assert PLANTED == {}
    result = reconcile(ledger, bank, prior_month_txn_ids=PRIOR_MONTH_TXN_IDS)
    assert result.gaps == []
    assert result.bridge["unexplained"] == Decimal("0")
    assert result.bridge["difference"] == Decimal("0")
    assert result.matched_count == len(ledger) == len(bank)
    assert len(ledger) >= 300


@pytest.mark.parametrize("seed", [0, 1, 7, 2026, 99999])
def test_planted_gaps_hold_for_many_seeds(seed):
    ledger, bank = generate_data(seed, plant_gaps=True)
    result = reconcile(ledger, bank, prior_month_txn_ids=PRIOR_MONTH_TXN_IDS)
    types = {g["txn_id"]: g["type"] for g in result.gaps if g["txn_id"]}
    assert types == {PLANTED["timing"]: "timing", PLANTED["duplicate"]: "duplicate",
                     PLANTED["orphan_refund"]: "orphan_refund"}
    assert result.bridge["unexplained"] == Decimal("0")


# --- Determinism ---------------------------------------------------------------------

def test_same_seed_identical_data():
    l1, b1 = generate_data(123)
    l2, b2 = generate_data(123)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(b1, b2)


def test_different_seeds_differ():
    l1, b1 = generate_data(123)
    l2, b2 = generate_data(124)
    assert not l1.equals(l2)
    assert not b1.equals(b2)


# --- Calendar ------------------------------------------------------------------------

def test_weekend_settlement_rolls_to_monday():
    friday, saturday, sunday = date(2026, 10, 9), date(2026, 10, 10), date(2026, 10, 11)
    monday = date(2026, 10, 12)
    assert add_business_days(friday, 1) == monday
    assert add_business_days(saturday, 1) == monday
    assert add_business_days(sunday, 1) == monday
    assert add_business_days(friday, 2) == date(2026, 10, 13)


def test_generated_settlements_never_on_weekends(planted_run):
    _, bank, _, _ = planted_run
    assert all(ts.weekday() < 5 for ts in bank.settled_at)
    assert all((ts.year, ts.month) == (2026, 10) for ts in bank.settled_at)


def test_oct_31_transaction_crosses_month_boundary():
    oct31 = date(2026, 10, 31)  # a Saturday
    assert add_business_days(oct31, 1) == date(2026, 11, 2)
    assert add_business_days(oct31, 2) == date(2026, 11, 3)

    ledger = ledger_frame([
        payment("T1", utc(2026, 10, 12, 9), "500.0000", "CUST-A"),
        payment("T2", utc(2026, 10, 31, 23, 59), "750.2500", "CUST-B"),
    ])
    bank = bank_frame([settlement("S1", utc(2026, 10, 13, 4, 30), "500.00", "T1")])
    result = reconcile(ledger, bank)
    assert [(g["txn_id"], g["type"], g["amount"]) for g in result.gaps] == [
        ("T2", "timing", Decimal("750.2500"))
    ]
    assert result.bridge["unexplained"] == Decimal("0")


# --- Fuzzy tier ----------------------------------------------------------------------

def test_fuzzy_match_on_mangled_reference():
    ledger = ledger_frame([payment("TXN-2610-00007", utc(2026, 10, 7, 9), "1234.5649", "CUST-A")])
    # Bank lower-cased the reference and replaced dashes with spaces, so tier 1 misses it.
    bank = bank_frame([settlement("S1", utc(2026, 10, 8, 4, 30), "1234.56", "txn 2610 00007")])
    result = reconcile(ledger, bank)
    assert result.matches[0]["method"] == "fuzzy"
    assert 0 < result.matches[0]["confidence"] <= 1
    assert [g["type"] for g in result.gaps] == ["rounding"]
    assert result.gaps[0]["amount"] == Decimal("0.0049")
    assert result.bridge["unexplained"] == Decimal("0")


def test_bank_round_half_up():
    assert to_bank_amount(Decimal("10.0050")) == Decimal("10.01")
    assert to_bank_amount(Decimal("10.0049")) == Decimal("10.00")
    assert to_bank_amount(Decimal("-10.0050")) == Decimal("-10.01")
