"""Result types returned by the matcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

GAP_TYPES = ("timing", "duplicate", "orphan_refund", "rounding", "unmatched")

GAP_COLUMNS = ["type", "txn_id", "amount", "reason", "confidence"]


@dataclass
class ReconResult:
    """Outcome of a reconciliation run.

    gaps:    list of {type, txn_id, amount (Decimal, ledger minus bank), reason, confidence}
    bridge:  {ledger_total, bank_total, difference, explained, unexplained}, all Decimal
    matches: one dict per matched ledger/bank pair (method is "exact" or "fuzzy")
    """

    gaps: list[dict]
    bridge: dict[str, Decimal]
    matches: list[dict] = field(default_factory=list)

    @property
    def matched_count(self) -> int:
        return len(self.matches)

    @property
    def is_balanced(self) -> bool:
        return self.bridge["unexplained"] == Decimal("0")

    def gaps_df(self) -> pd.DataFrame:
        """Gaps as a DataFrame; `amount` stays as Decimal objects (object dtype)."""
        return pd.DataFrame(self.gaps, columns=GAP_COLUMNS)

    def totals_by_type(self) -> dict[str, Decimal]:
        """Sum of gap amounts per gap type, in the canonical GAP_TYPES order."""
        totals: dict[str, Decimal] = {}
        for gap_type in GAP_TYPES:
            amounts = [g["amount"] for g in self.gaps if g["type"] == gap_type]
            if amounts:
                totals[gap_type] = sum(amounts, Decimal("0"))
        return totals
