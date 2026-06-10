"""Event-driven backtest engine.

No-lookahead by construction:
  * The strategy is called once per day with a History whose panels are sliced to end
    at that day's close (``close.loc[:date]``). It physically cannot read future rows.
  * The rebalance it implies is executed at the *next* session's OPEN, never at the
    close it just observed. You decide on bar t, you trade at t+1 open.

Accounting is share-based and explicit so it can be audited: positions are shares,
cash is tracked through every trade, costs are charged on traded notional. Fractional
shares are allowed (Alpaca supports them; fine for daily research).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import pandas as pd

from ..strategies.base import History, Strategy
from .costs import DEFAULT_COST, CostModel


class RiskOverlay(Protocol):
    """A hard, model-independent risk layer. Given the next order and the realized equity
    history so far, it may veto/scale the order (e.g. flatten to cash). See
    :class:`quantlab.backtest.risk.CircuitBreaker`."""

    def apply(self, pending: dict[str, float], equity: list[float]) -> dict[str, float]: ...


@dataclass
class BacktestResult:
    equity: pd.Series          # portfolio equity marked at each day's close
    returns: pd.Series         # daily portfolio returns
    weights: pd.DataFrame      # target weights per day (index=date, cols=symbol)
    turnover: pd.Series        # one-way traded notional / equity, per rebalance
    initial_cash: float
    strategy_name: str

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.initial_cash - 1.0)


def _valuation_panel(close: pd.DataFrame) -> pd.DataFrame:
    """Prices used to mark positions: forward-filled so a transient missing print
    doesn't produce a NaN equity. ffill uses only past data — no lookahead."""
    return close.ffill()


def run_backtest(
    bars: dict[str, pd.DataFrame],
    strategy: Strategy,
    *,
    initial_cash: float = 100_000.0,
    cost_model: CostModel = DEFAULT_COST,
    start: str | None = None,
    end: str | None = None,
    risk_overlay: "RiskOverlay | None" = None,
) -> BacktestResult:
    """Run ``strategy`` over ``bars`` (a {symbol: OHLCV} dict) and return results."""
    from ..data_pipeline.loaders import to_panel

    close = to_panel(bars, "close")
    open_ = to_panel(bars, "open")
    # Align open to the close calendar.
    open_ = open_.reindex(close.index)

    if start is not None:
        close = cast(pd.DataFrame, close.loc[pd.Timestamp(start, tz="UTC") :])
        open_ = cast(pd.DataFrame, open_.loc[pd.Timestamp(start, tz="UTC") :])
    if end is not None:
        close = cast(pd.DataFrame, close.loc[: pd.Timestamp(end, tz="UTC")])
        open_ = cast(pd.DataFrame, open_.loc[: pd.Timestamp(end, tz="UTC")])

    dates = close.index
    close_val = _valuation_panel(close)
    open_val = open_.ffill()

    cash = initial_cash
    shares: dict[str, float] = {}
    pending: dict[str, float] | None = None

    equity_records: list[float] = []
    weight_records: list[dict[str, float]] = []
    turnover_records: list[float] = []

    def position_value(prices: pd.Series) -> float:
        return sum(sh * prices.get(sym, 0.0) for sym, sh in shares.items() if sh != 0.0)

    for i, date in enumerate(dates):
        # 1) Execute yesterday's decision at today's OPEN.
        if pending is not None:
            op = open_val.loc[date]
            equity_at_open = cash + position_value(op)
            traded_notional = 0.0
            symbols = set(shares) | set(pending)
            for sym in symbols:
                price = op.get(sym)
                if price is None or pd.isna(price) or price <= 0:
                    continue  # can't trade what we can't price
                target_shares = pending.get(sym, 0.0) * equity_at_open / price
                delta = target_shares - shares.get(sym, 0.0)
                if delta == 0.0:
                    continue
                traded_notional += abs(delta * price)
                cash -= delta * price
                shares[sym] = target_shares
            cash -= cost_model.cost(traded_notional)
            turnover_records.append(
                traded_notional / equity_at_open if equity_at_open > 0 else 0.0
            )
            pending = None
        else:
            turnover_records.append(0.0)

        # 2) Mark to market at today's CLOSE.
        cl = close_val.loc[date]
        equity = cash + position_value(cl)
        equity_records.append(equity)

        # 3) Decide using history through today's close (executed next open).
        history = History(
            as_of=date,
            close=cast(pd.DataFrame, close.loc[:date]),
            open=cast(pd.DataFrame, open_.loc[:date]),
        )
        pending = strategy.target_weights(history)
        # Safety net: the circuit breaker can veto/flatten the order based on equity alone,
        # independent of the strategy. Applied to what we actually intend to trade.
        if risk_overlay is not None:
            pending = risk_overlay.apply(pending, equity_records)
        weight_records.append(dict(pending) if pending else {})

    equity = pd.Series(equity_records, index=dates, name="equity")
    weights = pd.DataFrame(weight_records, index=dates).fillna(0.0)
    turnover = pd.Series(turnover_records, index=dates, name="turnover")
    returns = equity.pct_change().fillna(0.0).rename("returns")

    return BacktestResult(
        equity=equity,
        returns=returns,
        weights=weights,
        turnover=turnover,
        initial_cash=initial_cash,
        strategy_name=strategy.name,
    )
