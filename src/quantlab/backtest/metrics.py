"""Performance metrics and sanity baselines.

The bar a strategy must clear (docs/00-PLAN.md section 5): beat buy-and-hold SPY on a
*risk-adjusted, net-of-cost* basis, and clearly beat dumb baselines (always-long,
random-entry) with comparable exposure. If it doesn't beat a coin flip, it isn't an edge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Metrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    time_under_water: float  # fraction of days not at a new high-water mark
    hit_rate: float          # fraction of up days
    avg_daily_turnover: float


def _cagr(equity: pd.Series) -> float:
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = n / TRADING_DAYS
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def time_under_water(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity < peak).mean())


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def compute_metrics(result) -> Metrics:
    """``result`` is a BacktestResult (duck-typed: needs .equity/.returns/.turnover)."""
    equity = result.equity
    returns = result.returns
    return Metrics(
        total_return=float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        cagr=_cagr(equity),
        ann_vol=float(returns.std() * np.sqrt(TRADING_DAYS)),
        sharpe=sharpe(returns),
        max_drawdown=max_drawdown(equity),
        time_under_water=time_under_water(equity),
        hit_rate=float((returns > 0).mean()),
        avg_daily_turnover=float(result.turnover.mean()),
    )


def summarize(results: Mapping[str, object]) -> pd.DataFrame:
    """Build a comparison table: one row per strategy/baseline.

    ``results`` maps a label -> BacktestResult. Returns a DataFrame sorted by Sharpe.
    """
    rows = {label: asdict(compute_metrics(res)) for label, res in results.items()}
    df = pd.DataFrame(rows).T
    return df.sort_values("sharpe", ascending=False)
