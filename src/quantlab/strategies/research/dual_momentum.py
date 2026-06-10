"""Dual momentum (Antonacci GEM style).

Combine *relative* momentum (cross-sectional ranking) with *absolute* momentum
(an own-asset trend filter). Each decision day we:

  1. Score every priced name by its trailing ~12-month total return (relative
     momentum), and pick the top ``k`` winners.
  2. Apply an absolute-momentum gate: a winner is only held if its own 12m return
     also clears a safe asset (a bond ETF, here IEF). Names that fail the gate
     have their sleeve rotated into the bond ETF instead — flee to bonds when
     momentum is broadly negative.
  3. Inverse-vol weight the resulting sleeves so a single jumpy name doesn't
     dominate, then normalize to the target gross exposure.

The book is recomputed only every ``rebalance_days`` sessions (a self counter),
and the underlying signal is a slow 12-month trend, so day-to-day turnover stays
low — the strategy holds a near-static handful of names for weeks at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import momentum, realized_vol
from ..base import History, Strategy

#: Safe / defensive asset used both as the absolute-momentum hurdle and as the
#: parking spot for sleeves whose winner fails the trend gate.
SAFE_ASSET = "IEF"


class DualMomentum(Strategy):
    """Top-``k`` 12-month relative-momentum winners, gated by absolute momentum
    against a bond ETF (IEF); failing sleeves rotate into bonds. Inverse-vol
    weighted, rebalanced every ~21 sessions."""

    def __init__(
        self,
        k: int = 4,
        lookback: int = 252,
        skip: int = 21,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        safe_asset: str = SAFE_ASSET,
        target_gross: float = 1.0,
    ) -> None:
        self.k = k
        self.lookback = lookback
        self.skip = skip
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.safe_asset = safe_asset
        self.target_gross = target_gross
        # Need enough bars for the longest trailing window plus the skip offset.
        self.warmup = lookback + skip + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only refresh the book every ``rebalance_days`` sessions; otherwise return
        # the previously chosen weights verbatim to keep turnover near zero.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        # Trailing 12-1 style total return (relative momentum), most recent row.
        mom = momentum(close, lookback=self.lookback, skip=self.skip).iloc[-1]
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]

        # Candidates: names with a finite momentum score and a current price.
        last = close.iloc[-1]
        candidates = [
            s
            for s in close.columns
            if not np.isnan(mom.get(s, np.nan)) and not np.isnan(last.get(s, np.nan))
        ]
        if not candidates:
            self._held = {}
            return {}

        # Absolute-momentum hurdle: the safe asset's own 12m return (cash=0 if the
        # safe asset isn't priced yet).
        hurdle = mom.get(self.safe_asset, np.nan)
        if np.isnan(hurdle):
            hurdle = 0.0

        # Relative momentum: rank candidates high-to-low, take the top k winners.
        ranked = sorted(candidates, key=lambda s: mom[s], reverse=True)
        winners = ranked[: self.k]

        # Absolute-momentum gate: a winner that fails the hurdle rotates its sleeve
        # into the safe asset (bonds). If the safe asset is unavailable, that sleeve
        # falls to cash.
        sleeves: list[str] = []
        for s in winners:
            if mom[s] > hurdle:
                sleeves.append(s)
            elif self.safe_asset in candidates:
                sleeves.append(self.safe_asset)
            # else: leave the sleeve in cash (drop it).

        if not sleeves:
            self._held = {}
            return {}

        # Inverse-vol weight the (possibly duplicated) sleeves; a name picked for
        # multiple sleeves accumulates weight. Guard against zero/NaN vol.
        raw: dict[str, float] = {}
        for s in sleeves:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = raw.get(s, 0.0) + inv

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        weights = {s: self.target_gross * w / total for s, w in raw.items()}
        self._held = weights
        return dict(weights)
