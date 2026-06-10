"""Range mean-reversion (Bollinger-band style).

The counter-trend complement to Donchian breakout: instead of chasing names that
have *escaped* their range, we fade names that have *overshot* the bottom of it.

Each decision day we measure every priced ETF's position within its own recent
range as a Bollinger z-score / %b::

    z = (close - SMA(n)) / rolling_std(n)

A deeply negative z means the close sits near (or below) the lower Bollinger band —
statistically oversold relative to its own trailing distribution. We:

  1. Rank names by z ascending and go LONG the ``max_names`` most oversold names
     whose z clears the ``entry_z`` threshold (e.g. z <= -1).
  2. Hold a name until it reverts back toward the middle of its range
     (``z >= exit_z``, ~0), then drop it — we harvest the snap-back, not the trend.
  3. Inverse-vol weight survivors so a single jumpy name doesn't dominate, and cap
     each name at ``max_weight`` of the book.

The book is recomputed only every ``rebalance_days`` sessions via a self counter,
and positions are sticky (held until they revert), so day-to-day turnover stays low.

thesis: in liquid index ETFs short-horizon moves overshoot and mean-revert; buying
statistically oversold lower-band names harvests the reversion premium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class RangeMeanReversion(Strategy):
    """Buy the range LOW / fade the range HIGH. Long the ``max_names`` most oversold
    ETFs (most negative Bollinger z-score, near the lower band), held until they
    revert toward the band centre. Inverse-vol weighted, per-name capped, rebalanced
    every ~4 sessions to keep turnover and drawdown bounded."""

    def __init__(
        self,
        band_lookback: int = 15,
        entry_z: float = -1.25,
        exit_z: float = 0.0,
        max_names: int = 5,
        rebalance_days: int = 5,
        vol_lookback: int = 63,
        max_weight: float = 0.30,
        target_gross: float = 1.0,
    ) -> None:
        self.band_lookback = band_lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_names = max_names
        self.rebalance_days = rebalance_days
        self.vol_lookback = vol_lookback
        self.max_weight = max_weight
        self.target_gross = target_gross
        # Need enough bars for the band and vol windows.
        self.warmup = max(band_lookback, vol_lookback) + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def _bollinger_z(self, close: pd.DataFrame) -> pd.Series:
        """Most-recent-row Bollinger z-score per name: (close - SMA(n)) / std(n).
        Point-in-time — row t uses only closes up to t."""
        n = self.band_lookback
        sma = close.rolling(n).mean()
        std = close.rolling(n).std()
        z = (close - sma) / std.replace(0.0, np.nan)
        return z.iloc[-1]

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only refresh the book every ``rebalance_days`` sessions; otherwise hold the
        # previous weights verbatim to keep turnover near zero.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        z = self._bollinger_z(close)
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        last = close.iloc[-1]

        priced = [
            s
            for s in close.columns
            if not np.isnan(z.get(s, np.nan)) and not np.isnan(last.get(s, np.nan))
        ]
        if not priced:
            self._held = {}
            return {}

        # Carry over names already held that have NOT yet reverted (z < exit_z): the
        # reversion hasn't completed, so keep harvesting it.
        keep = [
            s
            for s in self._held
            if s in priced and z.get(s, np.nan) < self.exit_z
        ]

        # New entries: most oversold names below the entry threshold, not already kept.
        candidates = sorted(
            (s for s in priced if z[s] <= self.entry_z and s not in keep),
            key=lambda s: z[s],
        )

        # Fill up to max_names: held survivors first, then the most oversold newcomers.
        selected = keep + candidates
        selected = selected[: self.max_names]
        if not selected:
            self._held = {}
            return {}

        # Inverse-vol weight; guard zero/NaN vol.
        raw: dict[str, float] = {}
        for s in selected:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = inv

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        weights = {s: self.target_gross * w / total for s, w in raw.items()}

        # Per-name cap, then renormalize the uncapped remainder so gross stays on target.
        weights = self._apply_cap(weights)

        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp each name to ``max_weight`` and redistribute the excess across the
        uncapped names proportionally (iteratively, until stable)."""
        cap = self.max_weight
        w = dict(weights)
        for _ in range(len(w) + 1):
            over = {s: v for s, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for s in over:
                w[s] = cap
            under = {s: v for s, v in w.items() if v < cap - 1e-12}
            pool = sum(under.values())
            if pool <= 0.0:
                break
            for s in under:
                w[s] += excess * w[s] / pool
        return w
