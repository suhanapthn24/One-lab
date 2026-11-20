"""Streamlit UI for the October 2026 payments reconciliation."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from recon.generator import PRIOR_MONTH_TXN_IDS, generate_data
from recon.matcher import reconcile
from recon.models import GAP_TYPES, ReconResult

ASSUMPTIONS = [
    "Month is October 2026, about 300 transactions, single currency (INR).",
    "All money uses decimal.Decimal, never float. The chart converts to numbers only to draw bars.",
    "All timestamps are UTC.",
    "The bank settles each transaction individually, 1-2 business days after creation, skipping weekends.",
    "Ledger amounts have 4 decimal places (fee-adjusted); bank amounts are rounded to 2dp per line, ROUND_HALF_UP.",
    "Refunds settle as negative amounts. A refund's original may be in the prior month.",
    "Bank rows carry a `reference` that normally equals the ledger `txn_id`.",
    "Timing window = the last 3 business days of the month (Oct 28-30, plus the Oct 31 weekend).",
    "Rounding drift is planted on 80 lines (0.0040-0.0049 each) so the total lands in 0.30-0.50; "
    "40 lines under 0.005 each cannot exceed 0.20.",
]

GAP_LABELS = {
    "timing": "Timing (next-month settlement)",
    "duplicate": "Duplicate ledger entry",
    "orphan_refund": "Orphan refund",
    "rounding": "Rounding drift",
    "unmatched": "Unmatched",
}

st.set_page_config(page_title="Payments Reconciliation", page_icon="🧾", layout="wide")


def money(value: Decimal) -> str:
    """Format a Decimal as INR with thousands separators, keeping its precision."""
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,}"


@st.cache_data(show_spinner=False)
def run(seed: int, plant_gaps: bool) -> tuple[pd.DataFrame, pd.DataFrame, ReconResult]:
    ledger, bank = generate_data(seed, plant_gaps=plant_gaps)
    return ledger, bank, reconcile(ledger, bank, prior_month_txn_ids=PRIOR_MONTH_TXN_IDS)


def waterfall(result: ReconResult) -> go.Figure:
    """Ledger total -> minus each gap type -> minus unexplained -> bank total."""
    br = result.bridge
    labels = ["Ledger total"]
    steps: list[Decimal] = [br["ledger_total"]]
    measures = ["absolute"]
    for gap_type, total in result.totals_by_type().items():
        labels.append(GAP_LABELS[gap_type])
        steps.append(-total)  # walking from ledger towards bank removes each gap
        measures.append("relative")
    labels.append("Unexplained")
    steps.append(-br["unexplained"])
    measures.append("relative")
    labels.append("Bank total")
    steps.append(br["bank_total"])
    measures.append("total")

    text = [money(br["ledger_total"])] + [money(s) for s in steps[1:-1]] + [money(br["bank_total"])]
    # Running levels, to zoom the y-axis onto the part of the walk that moves.
    levels, running = [], Decimal("0")
    for step, measure in zip(steps, measures):
        running = step if measure in ("absolute", "total") else running + step
        levels.append(running)
    low, high = min(levels), max(levels)
    pad = max((high - low) * Decimal("0.6"), Decimal("1"))

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=[str(s) for s in steps],  # strings keep exact Decimal text; Plotly parses for drawing
        measure=measures,
        text=text,
        textposition="outside",
        connector={"line": {"color": "rgba(128,128,128,0.5)", "width": 1}},
        decreasing={"marker": {"color": "#2a78d6"}},
        increasing={"marker": {"color": "#eb6834"}},
        totals={"marker": {"color": "#8a8985"}},
        hovertemplate="%{x}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        height=430,
        margin={"t": 30, "b": 10, "l": 10, "r": 10},
        showlegend=False,
        yaxis={"range": [str(low - pad), str(high + pad)], "title": "INR", "tickformat": ",.0f"},
    )
    return fig


# --- Sidebar ------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    with st.form("controls"):
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
        plant = st.toggle("Plant gaps", value=True,
                          help="Off = perfectly clean data that reconciles with zero gaps.")
        st.form_submit_button("Regenerate data", type="primary", width="stretch")
    st.caption("Same seed always produces identical data.")

ledger_df, bank_df, result = run(int(seed), bool(plant))
br = result.bridge

st.title("Payments reconciliation · October 2026")
st.caption("Platform ledger vs bank settlements. Every rupee of difference is traced to a gap.")

# --- Metrics -----------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ledger total", money(br["ledger_total"]), help=f"{len(ledger_df)} ledger rows")
c2.metric("Bank total", money(br["bank_total"]), help=f"{len(bank_df)} bank settlements")
c3.metric("Difference (ledger − bank)", money(br["difference"]))
with c4:
    colour = "green" if result.is_balanced else "red"
    st.markdown(f"<div style='font-size:0.875rem'>Unexplained</div>"
                f"<div style='font-size:2.25rem;color:{colour};font-weight:600'>"
                f"{money(br['unexplained'])}</div>", unsafe_allow_html=True)

if result.is_balanced:
    st.success(f"Bridge closes: the difference of {money(br['difference'])} is fully explained "
               f"by {len(result.gaps)} gap(s).", icon="✅")
else:
    st.error(f"UNEXPLAINED DIFFERENCE of {money(br['unexplained'])}. Gaps explain "
             f"{money(br['explained'])} of the {money(br['difference'])} difference. "
             "Investigate before closing the books.", icon="🚨")

# --- Waterfall ---------------------------------------------------------------------
st.subheader("Bridge: ledger total → bank total")
st.plotly_chart(waterfall(result), width="stretch")

# --- Gaps table ----------------------------------------------------------------------
st.subheader("Gaps")
gaps = result.gaps_df()
present = [t for t in GAP_TYPES if t in set(gaps["type"])]
selected = st.multiselect("Filter by type", options=present, default=present,
                          format_func=lambda t: GAP_LABELS[t])
view = gaps[gaps["type"].isin(selected)].copy()
view["amount"] = view["amount"].map(money)
view["txn_id"] = view["txn_id"].fillna("(aggregate)")
if view.empty:
    st.info("No gaps." if gaps.empty else "No gaps of the selected types.")
else:
    st.dataframe(view, hide_index=True, width="stretch",
                 column_config={
                     "reason": st.column_config.TextColumn("reason", width="large"),
                     "confidence": st.column_config.ProgressColumn(
                         "confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                 })

# --- Details -------------------------------------------------------------------------
with st.expander("Matching details and assumptions"):
    exact = sum(1 for m in result.matches if m["method"] == "exact")
    fuzzy = result.matched_count - exact
    st.markdown(f"**Matched rows:** {result.matched_count} "
                f"({exact} exact, {fuzzy} fuzzy) out of {len(ledger_df)} ledger rows "
                f"and {len(bank_df)} bank rows.")
    st.markdown("**Assumptions**")
    st.markdown("\n".join(f"- {a}" for a in ASSUMPTIONS))

# --- Downloads -----------------------------------------------------------------------
st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button("ledger.csv", ledger_df.to_csv(index=False), "ledger.csv", "text/csv",
                   width="stretch")
d2.download_button("bank.csv", bank_df.to_csv(index=False), "bank.csv", "text/csv",
                   width="stretch")
d3.download_button("gaps_report.csv", gaps.to_csv(index=False), "gaps_report.csv", "text/csv",
                   width="stretch")
