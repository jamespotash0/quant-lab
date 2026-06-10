"""Rolling walk-forward (out-of-sample) validation harness.

The scorecard's single in-sample/out-of-sample split (``SPLIT``) answers one question:
did the edge survive *one* held-out period? That's a thin sample — a strategy can look
great on a single lucky OOS window. Walk-forward stress-tests the same claim across
*many* consecutive out-of-sample windows: after an initial training period we step
forward in fixed test slices, and for each slice we backtest a FRESH strategy on data it
never had a hand in. A robust edge shows a CONSISTENTLY positive OOS Sharpe across most
folds — not one outlier carrying the average.

No lookahead by construction: each fold only ever backtests over its own test window
(``run_backtest(..., start=fold_start, end=fold_end)``). The engine itself never reads
future rows (see :mod:`quantlab.backtest.engine`), so a per-fold start/end is a genuine
out-of-sample evaluation. We never fit anything on the train period — it exists only to
let each fold's backtest spend its indicator warmup on pre-window data and to anchor the
first test window — so there is nothing to leak across the boundary.

Usage:
    from quantlab.research.walkforward import walk_forward, render
    from quantlab.strategies.research.recommended import SwingMomentumV3
    print(render(walk_forward(SwingMomentumV3)))
"""

from __future__ import annotations

import copy
from typing import Callable, Union

import numpy as np
import pandas as pd

from ..backtest.costs import DEFAULT_COST
from ..backtest.engine import run_backtest
from ..backtest.metrics import max_drawdown, sharpe
from ..strategies.base import Strategy
from .runner import START, load_panel

#: A factory is either a zero-arg callable returning a Strategy (e.g. ``lambda: S()``),
#: or a Strategy class/subclass (called with no args), or a ready-made Strategy instance
#: we deep-copy per fold.
StrategyFactory = Union[Callable[[], Strategy], type[Strategy], Strategy]


def _make_strategy(factory: StrategyFactory) -> Strategy:
    """Materialise one fresh Strategy from ``factory``.

    Accepts a class (instantiated with no args), a zero-arg callable, or an existing
    instance (deep-copied so mutable strategy state never leaks between folds)."""
    if isinstance(factory, type):
        return factory()
    if isinstance(factory, Strategy):
        return copy.deepcopy(factory)
    return factory()


def _fold_windows(
    start: str, end: str | None, train_years: float, test_months: int
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build consecutive (test_start, test_end) windows.

    The first test window opens ``train_years`` after ``start`` (the initial training
    period), then we step forward in ``test_months`` slices until we run past ``end``.
    Windows are non-overlapping and consecutive, so stitching their OOS equity gives one
    continuous out-of-sample track record."""
    begin = pd.Timestamp(start)
    final = pd.Timestamp(end) if end is not None else pd.Timestamp.utcnow().tz_localize(None)
    if final.tzinfo is not None:
        final = final.tz_localize(None)

    first_test = begin + pd.DateOffset(months=round(train_years * 12))
    step = pd.DateOffset(months=test_months)

    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = first_test
    while cursor < final:
        stop = min(cursor + step, final)
        if stop <= cursor:
            break
        windows.append((cursor, stop))
        cursor = stop
    return windows


def walk_forward(
    strategy_factory: StrategyFactory,
    *,
    start: str = START,
    end: str | None = None,
    train_years: float = 2.0,
    test_months: int = 6,
) -> pd.DataFrame:
    """Rolling walk-forward validation of ``strategy_factory``.

    The strategies here are *online-adaptive*: they recompute their signals from past data
    on every rebalance and have no separate fit step, so a single full backtest already
    trades each day on strict-past information only. Walk-forward then means partitioning
    that one continuous out-of-sample track into consecutive windows and checking the edge
    is *consistent* across them — not carried by one lucky period. (Backtesting each fold on
    a clipped data slice would instead starve a high-warmup strategy of the history it needs
    and is the bug this replaces.)

    Returns one row per window (``fold_start``/``fold_end``, ``sharpe``, ``cagr``, ``maxdd``,
    ``turnover``, ``days``) plus a final ``ALL`` aggregate over the whole post-warmup track.
    The first window opens ``train_years`` after ``start`` to skip the indicator warmup.
    """
    bars = load_panel(start, end)
    strat = _make_strategy(strategy_factory)
    res = run_backtest(bars, strat, cost_model=DEFAULT_COST, start=start, end=end)

    returns = res.returns.copy()
    returns.index = pd.DatetimeIndex(returns.index).tz_localize(None)
    turn = res.turnover.copy()
    turn.index = returns.index

    windows = _fold_windows(start, end, train_years, test_months)
    rows: list[dict[str, object]] = []
    turnovers: list[float] = []

    for fold_start, fold_end in windows:
        mask = (returns.index >= fold_start) & (returns.index < fold_end)
        seg = returns[mask]
        if len(seg) < 2:
            continue
        equity = (1.0 + seg).cumprod()
        t = float(turn[mask].mean())
        turnovers.append(t)
        rows.append({
            "fold_start": fold_start.strftime("%Y-%m-%d"),
            "fold_end": fold_end.strftime("%Y-%m-%d"),
            "days": int(len(seg)),
            "sharpe": sharpe(seg),
            "cagr": float(equity.iloc[-1] ** (252.0 / len(seg)) - 1.0),
            "maxdd": max_drawdown(equity),
            "turnover": t,
        })

    df = pd.DataFrame(
        rows,
        columns=["fold_start", "fold_end", "days", "sharpe", "cagr", "maxdd", "turnover"],
    )

    if rows:
        first = pd.Timestamp(rows[0]["fold_start"])  # type: ignore[arg-type]
        oos = returns[returns.index >= first]
        equity = (1.0 + oos).cumprod()
        agg = {
            "fold_start": "ALL",
            "fold_end": f"{len(df)} folds",
            "days": int(len(oos)),
            "sharpe": sharpe(oos),
            "cagr": float(equity.iloc[-1] ** (252.0 / len(oos)) - 1.0) if len(oos) >= 2 else 0.0,
            "maxdd": max_drawdown(equity),
            "turnover": float(np.mean(turnovers)) if turnovers else float("nan"),
        }
        df = pd.concat([df, pd.DataFrame([agg])], ignore_index=True)

    return df


def render(df: pd.DataFrame) -> str:
    """A readable fixed-width table of the walk-forward folds + aggregate row."""
    name = "walk-forward"
    header = f"{'window':<26}{'days':>6}{'sharpe':>9}{'cagr':>9}{'maxdd':>9}{'turn':>9}"
    lines = [f"WALK-FORWARD OOS VALIDATION  ({name})", "-" * len(header), header,
             "-" * len(header)]

    for _, row in df.iterrows():
        is_agg = str(row["fold_start"]) == "ALL"
        if is_agg:
            lines.append("-" * len(header))
            window = f"ALL ({row['fold_end']})"
        else:
            window = f"{row['fold_start']} -> {row['fold_end']}"
        lines.append(
            f"{window:<26}"
            f"{int(row['days']):>6}"
            f"{float(row['sharpe']):>9.2f}"
            f"{float(row['cagr']):>8.1%}"
            f"{float(row['maxdd']):>8.1%}"
            f"{float(row['turnover']):>9.3f}"
        )

    folds = df[df["fold_start"] != "ALL"]
    if len(folds):
        pos = int((folds["sharpe"] > 0).sum())
        lines.append("-" * len(header))
        lines.append(
            f"{pos}/{len(folds)} folds with positive OOS Sharpe   "
            f"(median {float(folds['sharpe'].median()):.2f}, "
            f"mean {float(folds['sharpe'].mean()):.2f})"
        )
    return "\n".join(lines)
