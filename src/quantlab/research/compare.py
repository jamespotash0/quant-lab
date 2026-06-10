"""Build a ranked comparison table across strategies + baselines.

Takes the per-strategy metrics produced by :mod:`quantlab.research.runner` (net and
gross of costs) and renders one sorted table, so a sweep of candidates can be judged on
the same axes the decision gate cares about: risk-adjusted, net-of-cost return that
beats the dumb baselines.
"""

from __future__ import annotations

import pandas as pd

from ..strategies.baselines import AlwaysLong, RandomEntry
from ..strategies.buy_and_hold import BuyAndHold
from .runner import evaluate

#: The bar every candidate must clear.
BASELINES = {
    "Benchmark(SPY)": BuyAndHold("SPY"),
    "AlwaysLong": AlwaysLong(),
    "RandomEntry": RandomEntry(),
}

_COLS = ["sharpe", "cagr", "ann_vol", "max_drawdown", "avg_daily_turnover", "total_return"]


def baseline_rows(start: str | None = None) -> dict[str, dict]:
    """Evaluate the baselines so candidates can be ranked against them in one table."""
    kw = {"start": start} if start else {}
    return {name: evaluate(strat, **kw) for name, strat in BASELINES.items()}


def to_frame(results: dict[str, dict], which: str = "net") -> pd.DataFrame:
    """``results`` maps label -> {"net": {...}, "gross": {...}}. Returns a DataFrame of
    the chosen cost basis, sorted by Sharpe descending."""
    rows = {label: {c: r[which].get(c) for c in _COLS} for label, r in results.items()}
    df = pd.DataFrame(rows).T
    return df.sort_values("sharpe", ascending=False)


def render(results: dict[str, dict], which: str = "net") -> str:
    """Pretty, percent-formatted ranking table."""
    df = to_frame(results, which).copy()
    for c in ("cagr", "ann_vol", "max_drawdown", "avg_daily_turnover", "total_return"):
        df[c] = (df[c] * 100).map(lambda x: f"{x:7.2f}%")
    df["sharpe"] = df["sharpe"].map(lambda x: f"{x:6.2f}")
    return df.to_string()
