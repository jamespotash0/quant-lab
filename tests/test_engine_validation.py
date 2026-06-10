"""The cornerstone test: holding 100% of one instrument through the engine must
reproduce that instrument's own return. If this fails, engine accounting is wrong and
nothing downstream can be trusted (docs/00-PLAN.md section 5)."""

from __future__ import annotations

import numpy as np

from quantlab.backtest.costs import ZERO_COST
from quantlab.backtest.engine import run_backtest
from quantlab.data_pipeline.loaders import to_panel
from quantlab.strategies.buy_and_hold import BuyAndHold


def test_buy_and_hold_reproduces_underlying_returns(single_bars):
    res = run_backtest(single_bars, BuyAndHold("SPY"), cost_model=ZERO_COST)

    close = to_panel(single_bars, "close")["SPY"]
    spy_ret = close.pct_change()

    # Entry happens at the open of bar 1 (decision made at close of bar 0). From bar 2
    # onward the portfolio is fully invested, so its daily return must EQUAL SPY's
    # close-to-close return to floating-point tolerance.
    port = res.returns
    assert np.allclose(port.iloc[2:].to_numpy(), spy_ret.iloc[2:].to_numpy(), atol=1e-12)


def test_buy_and_hold_total_return_matches_hold_from_entry(single_bars):
    res = run_backtest(single_bars, BuyAndHold("SPY"), cost_model=ZERO_COST)
    open_ = to_panel(single_bars, "open")["SPY"]
    close = to_panel(single_bars, "close")["SPY"]
    # Bought at open[1], marked at the final close.
    expected = close.iloc[-1] / open_.iloc[1] - 1.0
    assert abs(res.total_return - expected) < 1e-9


def test_costs_reduce_return(single_bars):
    from quantlab.backtest.costs import DEFAULT_COST

    free = run_backtest(single_bars, BuyAndHold("SPY"), cost_model=ZERO_COST)
    costed = run_backtest(single_bars, BuyAndHold("SPY"), cost_model=DEFAULT_COST)
    # A single entry trade pays cost, so the costed run must end strictly lower.
    assert costed.total_return < free.total_return
