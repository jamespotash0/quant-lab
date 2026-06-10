"""Low-vol bull strategy — aggressive momentum participation in calm uptrends.

The book is built to lean *into* the market when conditions are benign. Each
rebalance we:

  1. Gauge the broad-market regime from the benchmark's (SPY) trailing realized
     volatility relative to its own slower history. When that vol sits in the calm
     part of its distribution we treat the tape as a low-vol bull and go fully
     invested; when vol spikes we de-risk into cash, because the thesis only holds
     while drawdown risk is contained.
  2. Score every priced name by the cross-sectional z-score of its 126/21
     (6-month, skip-1-month) momentum and pick the top handful of leaders.
  3. Inverse-vol size the winners so a single jumpy name doesn't dominate, apply a
     per-name cap, and normalize to the target gross exposure.

The book is recomputed only every ``rebalance_days`` sessions (a self counter) and
the momentum signal is slow, so turnover stays low — a near-static set of leaders
held for weeks at a time.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ...data_pipeline.features import (
    cross_sectional_zscore,
    momentum,
    realized_vol,
)
from ..base import History, Strategy

#: Benchmark used to read the broad-market volatility regime.
REGIME_ASSET = "SPY"


class LowVolBullStrategy(Strategy):
    """Top-``n_names`` cross-sectional momentum leaders, inverse-vol weighted and
    fully invested while the benchmark's realized volatility sits in the calm part
    of its own distribution; de-risks to cash when vol spikes. Rebalanced every
    ~21 sessions, per-name capped for risk management."""

    thesis = (
        "When market volatility is low the trend is your friend and drawdown risk "
        "is contained, so concentrate capital in the strongest momentum leaders to "
        "capture the upside; when volatility spikes the regime breaks and we step "
        "aside to cash."
    )

    def __init__(
        self,
        n_names: int = 6,
        mom_lookback: int = 126,
        mom_skip: int = 21,
        vol_lookback: int = 21,
        rebalance_days: int = 21,
        max_weight: float = 0.30,
        target_gross: float = 1.0,
        regime_asset: str = REGIME_ASSET,
        regime_vol_fast: int = 21,
        regime_vol_slow: int = 126,
        regime_buffer: float = 1.10,
    ) -> None:
        self.n_names = n_names
        self.mom_lookback = mom_lookback
        self.mom_skip = mom_skip
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.max_weight = max_weight
        self.target_gross = target_gross
        self.regime_asset = regime_asset
        self.regime_vol_fast = regime_vol_fast
        self.regime_vol_slow = regime_vol_slow
        self.regime_buffer = regime_buffer
        # Enough bars for the longest trailing window plus the skip offset.
        self.warmup = max(mom_lookback + mom_skip, regime_vol_slow) + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def _is_calm(self, close) -> bool:
        """True when the benchmark's fast realized vol is at or below its slower
        baseline (times a small buffer) — i.e. a calm, low-vol tape. Defaults to
        calm if the regime asset isn't priced so the strategy still trades."""
        if self.regime_asset not in close.columns:
            return True
        px = close[self.regime_asset]
        fast = cast(float, realized_vol(px.to_frame(), lookback=self.regime_vol_fast).iloc[-1, 0])
        slow = cast(float, realized_vol(px.to_frame(), lookback=self.regime_vol_slow).iloc[-1, 0])
        if np.isnan(fast) or np.isnan(slow) or slow <= 1e-8:
            return True
        return bool(fast <= slow * self.regime_buffer)

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

        # Regime gate: only deploy capital in calm (low-vol) markets.
        if not self._is_calm(close):
            self._held = {}
            return {}

        # Cross-sectional momentum z-score, most recent row.
        mom = momentum(close, lookback=self.mom_lookback, skip=self.mom_skip)
        z = cross_sectional_zscore(mom).iloc[-1]
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        last = close.iloc[-1]

        # Candidates: finite momentum z-score and a current price. Exclude the
        # regime asset itself from the leader pool? Keep it eligible — SPY can be a
        # legitimate leader; the per-name cap bounds concentration regardless.
        candidates = [
            s
            for s in close.columns
            if not np.isnan(z.get(s, np.nan)) and not np.isnan(last.get(s, np.nan))
        ]
        if not candidates:
            self._held = {}
            return {}

        # Pick the top-n leaders by momentum z-score; require positive momentum so
        # we never concentrate into the strongest of a falling pack.
        ranked = sorted(candidates, key=lambda s: z[s], reverse=True)
        winners = [s for s in ranked[: self.n_names] if z[s] > 0.0]
        if not winners:
            self._held = {}
            return {}

        # Inverse-vol weight, guarding zero/NaN vol.
        raw: dict[str, float] = {}
        for s in winners:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = inv

        weights = self._normalize(raw)
        self._held = weights
        return dict(weights)

    def _normalize(self, raw: dict[str, float]) -> dict[str, float]:
        """Scale raw inverse-vol scores to ``target_gross`` while enforcing the
        per-name cap. Capping can free up gross, which we redistribute to the
        uncapped names over a few passes; any residual stays in cash."""
        cap = self.max_weight * self.target_gross
        total = sum(raw.values())
        if total <= 0.0:
            return {}
        weights = {s: self.target_gross * w / total for s, w in raw.items()}

        for _ in range(8):
            over = {s: w for s, w in weights.items() if w > cap + 1e-12}
            if not over:
                break
            excess = sum(w - cap for s, w in over.items())
            for s in over:
                weights[s] = cap
            free = {s: w for s, w in weights.items() if w < cap - 1e-12}
            free_total = sum(free.values())
            if free_total <= 1e-12:
                break
            for s in free:
                weights[s] += excess * weights[s] / free_total

        # Final hard clamp so no name can ever exceed the cap (residual to cash).
        return {s: min(w, cap) for s, w in weights.items()}
