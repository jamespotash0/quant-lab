"""Volatility-managed portfolio (Moreira & Muir, 2017).

Take a slow, fixed base book — an equal weight over the broad US-equity ETFs — and
scale its *total* exposure inversely to recent realized volatility, aiming at a constant
annualized target vol. Because this is long-only and capped at gross <= 1.0, the lever
only ever *de-risks*: in calm regimes we hold the full book (gross ~1.0), and in turbulent
regimes we cut exposure toward cash, parking the remainder.

The realized-vol estimate moves slowly (it is an annualized rolling std of daily returns),
so day-to-day exposure changes are small. We snap the scale to a coarse grid and only
re-book the weights every few days, which keeps turnover well under the cost budget.

Moreira & Muir's central empirical result: cutting exposure *before* vol spikes (vol is
persistent, so today's high vol forecasts tomorrow's) reliably raises the Sharpe ratio.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy

#: Broad US-equity sleeve — the diversified, slow base book we scale up and down.
BASE_BOOK: tuple[str, ...] = ("SPY", "QQQ", "IWM", "DIA", "VTI")


class VolatilityManaged(Strategy):
    """Equal-weight broad-equity base book, total exposure scaled inversely to its own
    trailing realized vol to target a constant ~10-12% annualized vol, capped at gross
    1.0 (long-only). De-risks in high-vol regimes; holds the full book when calm."""

    def __init__(
        self,
        target_vol: float = 0.11,
        vol_lookback: int = 42,
        max_gross: float = 1.0,
        rebalance_days: int = 5,
        scale_grid: float = 0.05,
    ) -> None:
        self.target_vol = target_vol
        self.vol_lookback = vol_lookback
        self.max_gross = max_gross
        self.rebalance_days = rebalance_days
        self.scale_grid = scale_grid
        # Need enough bars for the vol estimate (plus one for the return diff).
        self.warmup = vol_lookback + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only re-book every ``rebalance_days`` days; otherwise repeat the prior weights
        # verbatim so turnover stays near zero between rebalances.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)

        # Investable slice of the base book: priced today and with a real vol estimate.
        members = [s for s in BASE_BOOK if s in close.columns]
        priced = [s for s in members if not np.isnan(close[s].iloc[-1])]
        if not priced:
            return {}

        # Portfolio-level realized vol: vol of the equal-weight base book's daily return.
        book_close = close[priced]
        book_ret = book_close.pct_change().mean(axis=1)
        rv = book_ret.rolling(self.vol_lookback).std().iloc[-1] * np.sqrt(252)
        if not np.isfinite(rv) or rv <= 0.0:
            # Fall back to a per-name average if the book-level estimate is degenerate.
            rv = float(np.nanmean(realized_vol(book_close, self.vol_lookback).iloc[-1]))
        if not np.isfinite(rv) or rv <= 0.0:
            return {}

        # Vol-target lever, capped at full investment (long-only de-risking only).
        scale = self.target_vol / rv
        scale = float(np.clip(scale, 0.0, self.max_gross))
        # Snap to a coarse grid so small vol wiggles don't churn the book.
        scale = round(scale / self.scale_grid) * self.scale_grid

        per_name = scale / len(priced)
        self._held = {s: per_name for s in priced}
        self._age = 1
        return dict(self._held)
