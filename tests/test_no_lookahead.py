"""Guards on the no-lookahead guarantee."""

from __future__ import annotations

import pandas as pd
import pytest

from quantlab.backtest.engine import run_backtest
from quantlab.strategies.base import History, Strategy
from quantlab.strategies.momentum import MomentumReversal


def test_history_rejects_future_rows():
    idx = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    df = pd.DataFrame({"SPY": range(5)}, index=idx)
    with pytest.raises(ValueError):
        History(as_of=idx[2], close=df, open=df)  # df extends past as_of


class _Spy(Strategy):
    """Records the last date it was shown; asserts it never exceeds the decision date."""

    def __init__(self):
        self.violations = 0
        self.calls = 0

    def target_weights(self, history):
        self.calls += 1
        if not history.close.empty and history.close.index.max() > history.as_of:
            self.violations += 1
        return {}


def test_engine_never_shows_future_data(bars):
    spy = _Spy()
    run_backtest(bars, spy)
    assert spy.calls > 0
    assert spy.violations == 0


def test_momentum_runs_and_respects_warmup(bars):
    res = run_backtest(bars, MomentumReversal(regime_ma=50, mom_lookback=60))
    # Before warmup the strategy must hold nothing (all-zero weight rows early on).
    early = res.weights.iloc[:40].abs().sum(axis=1)
    assert (early == 0).all()
    # And it produces a finite equity curve.
    assert res.equity.notna().all()
    assert res.equity.iloc[-1] > 0
