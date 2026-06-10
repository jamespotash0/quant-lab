"""Naive risk parity (inverse-volatility) allocation.

No return forecasting whatsoever. We hold the whole (priced) universe and weight each
asset inversely to its trailing realized volatility, so every name contributes roughly
equal risk to the book. Weights are normalized to sum ~1.0 (fully invested, long-only).

Because low-vol assets get large weights, the bond ETFs (TLT/IEF/AGG) naturally dominate
the book, which historically smooths the equity curve — lower return, higher Sharpe. The
allocation moves slowly (vol is a slow signal), and we only recompute it every
``rebalance_days`` via a self counter, so day-to-day turnover stays low.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class RiskParity(Strategy):
    """Inverse-vol allocator: weight_i ∝ 1 / vol_i, normalized to sum to ``target_gross``.

    A pure risk-based baseline with no alpha signal. Rebalanced every ``rebalance_days``
    so turnover is driven only by slow drift in trailing volatility.
    """

    def __init__(
        self,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        target_gross: float = 1.0,
    ) -> None:
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        # Need enough bars for the realized-vol window (plus one for the return diff).
        self.warmup = vol_lookback + 1
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only recompute the book every ``rebalance_days``; otherwise hold steady so the
        # engine sees near-identical weights day-to-day (low turnover).
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)

        # Trailing annualized realized vol per symbol, as of today (last row).
        vol = realized_vol(close, self.vol_lookback).iloc[-1]

        inv_vol: dict[str, float] = {}
        for sym in close.columns:
            v = vol[sym]
            # Skip newly-listed / illiquid names with NaN or zero vol.
            if np.isnan(v) or v <= 0.0:
                continue
            # Require a current price too (guard against trailing-NaN symbols).
            if np.isnan(close[sym].iloc[-1]):
                continue
            inv_vol[sym] = 1.0 / v

        total = sum(inv_vol.values())
        if total <= 0.0:
            return {}

        self._held = {s: self.target_gross * w / total for s, w in inv_vol.items()}
        self._age = 1
        return dict(self._held)
