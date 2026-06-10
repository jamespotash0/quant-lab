"""Time-series (absolute) momentum — the classic managed-futures trend strategy.

For *each asset independently*, we ask one question: is its own trailing ~12-month
return positive? If yes, the asset is eligible for a long position; if no, we hold
nothing in it (sit in cash on that name). This is "time-series" momentum (TSMOM /
absolute momentum), distinct from cross-sectional momentum: an asset competes against
its *own* past, not against its peers. It is the most-validated trend anomaly in the
literature (Moskowitz-Ooi-Pedersen) and tends to behave defensively — it de-risks into
cash as trends roll over in bear markets.

Sizing is inverse-realized-vol so every eligible name contributes similar risk; gross is
scaled toward ~1.0 and each name is capped to avoid concentration. We rebalance only
every ~21 days via a self counter, so day-to-day turnover stays low.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import momentum, realized_vol
from ..base import History, Strategy


class TimeSeriesMomentum(Strategy):
    """Absolute-momentum trend follower: long any asset whose own trailing return is
    positive, sized inverse-vol, rebalanced on a slow ~monthly cadence."""

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        target_gross: float = 1.0,
        max_weight: float = 0.20,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.max_weight = max_weight
        # Need enough bars for the trailing-return window plus the vol window.
        self.warmup = lookback + max(vol_lookback, skip) + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only recompute the book every ``rebalance_days`` sessions; otherwise return the
        # standing weights so the engine sees a near-static target (≈ zero turnover).
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        # Trailing absolute momentum and inverse-vol weights, point-in-time (last row).
        mom = momentum(close, lookback=self.lookback, skip=self.skip).iloc[-1]
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        last = close.iloc[-1]

        inv_vol: dict[str, float] = {}
        for sym in history.symbols:
            m = mom.get(sym, np.nan)
            v = vol.get(sym, np.nan)
            px = last.get(sym, np.nan)
            # Skip newly-listed/NaN names, names without a positive trend, and any with a
            # non-finite or zero vol (avoid divide-by-zero).
            if not np.isfinite(m) or not np.isfinite(v) or not np.isfinite(px):
                continue
            if m <= 0.0 or v <= 0.0:
                continue
            inv_vol[sym] = 1.0 / v

        if not inv_vol:
            self._held = {}
            return {}

        total = sum(inv_vol.values())
        weights = {s: self.target_gross * iv / total for s, iv in inv_vol.items()}

        # Cap per-name exposure, then redistribute the freed weight to uncapped names so
        # gross stays near target without ever exceeding it.
        weights = self._apply_cap(weights)

        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Cap each name at ``max_weight`` and spill the excess onto uncapped names
        proportionally, iterating until everything is within the cap."""
        w = dict(weights)
        for _ in range(len(w) + 1):
            over = {s: x for s, x in w.items() if x > self.max_weight + 1e-12}
            if not over:
                break
            excess = sum(x - self.max_weight for x in over.values())
            for s in over:
                w[s] = self.max_weight
            room = {s: x for s, x in w.items() if x < self.max_weight - 1e-12}
            base = sum(room.values())
            if base <= 0.0:
                break
            for s in room:
                w[s] += excess * (w[s] / base)
        return w
