"""Parameter sweep — the hand-off surface for a human quant.

Grid-search a strategy's constructor knobs and rank the combinations by *out-of-sample*
Sharpe, so a trader can see which parameters are robust (broad plateaus of good OOS
performance) versus overfit (lone in-sample spikes that collapse out of sample). The
golden rule baked in here: **never tune on the out-of-sample period** — we report it,
we don't optimize to it. Pick parameters on in-sample behaviour and sanity-check that
OOS holds up.

Usage:
    from quantlab.research.sweep import sweep
    from quantlab.strategies.research.dual_momentum import DualMomentum
    df = sweep(DualMomentum, {"k": [3, 4, 5, 6], "lookback": [126, 189, 252]})
    print(df.head(15).to_string())
"""

from __future__ import annotations

import copy
from itertools import product

import pandas as pd

from ..backtest.costs import DEFAULT_COST
from ..backtest.engine import run_backtest
from ..backtest.metrics import compute_metrics
from ..strategies.base import Strategy
from .runner import START, load_panel
from .scorecard import SPLIT


def sweep(
    strategy_cls: type[Strategy],
    grid: dict[str, list],
    start: str = START,
    split: str = SPLIT,
    end: str | None = None,
) -> pd.DataFrame:
    """Backtest every combination in ``grid`` (a {param: [values]} dict). Returns a
    DataFrame with one row per combination, the param values, and in-sample / out-of-sample
    metrics, sorted by OOS Sharpe descending."""
    bars = load_panel(start, end)
    keys = list(grid)
    rows: list[dict] = []
    for combo in product(*(grid[k] for k in keys)):
        kwargs = dict(zip(keys, combo))
        try:
            strat = strategy_cls(**kwargs)
            is_m = compute_metrics(
                run_backtest(bars, copy.deepcopy(strat), cost_model=DEFAULT_COST,
                             start=start, end=split)
            )
            oos_m = compute_metrics(
                run_backtest(bars, copy.deepcopy(strat), cost_model=DEFAULT_COST,
                             start=split, end=end)
            )
        except Exception as exc:  # a bad param combo shouldn't sink the whole sweep
            rows.append({**kwargs, "error": str(exc)[:60]})
            continue
        rows.append({
            **kwargs,
            "is_sharpe": round(is_m.sharpe, 3),
            "oos_sharpe": round(oos_m.sharpe, 3),
            "oos_cagr": round(oos_m.cagr, 4),
            "oos_maxdd": round(oos_m.max_drawdown, 4),
            "oos_turnover": round(oos_m.avg_daily_turnover, 4),
        })
    df = pd.DataFrame(rows)
    if "oos_sharpe" in df.columns:
        df = df.sort_values("oos_sharpe", ascending=False, na_position="last")
    return df.reset_index(drop=True)
