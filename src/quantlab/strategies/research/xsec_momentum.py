"""Pure cross-sectional momentum, done slowly.

Each rebalance we score the universe on a single signal — the cross-sectional
z-score of 126/21 momentum ("6-month return, skip last month") — and go long the
top ``k`` names, inverse-volatility sized so the book isn't dominated by the one
high-vol winner. There is **no** reversal sleeve: the prior MomentumReversal
strategy bled to death on a fast 5-day reversal leg plus daily churn, so here we
keep momentum alone and only touch the book every ``rebalance_days`` (~21) days.

The question this answers: does slow cross-sectional momentum, on its own and at
low turnover, carry any edge net of costs?
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import (
    cross_sectional_zscore,
    momentum,
    realized_vol,
)
from ..base import History, Strategy


class CrossSectionalMomentum(Strategy):
    """Long the top-``k`` momentum names, inverse-vol weighted, rebalanced monthly.

    Score is purely ``z(momentum(lookback, skip))`` across the universe. We pick the
    highest-scoring ``k`` names, size them by inverse realized vol (capped per name),
    and hold for ``rebalance_days`` before reselecting. Long-only, gross ~``target_gross``.
    """

    def __init__(
        self,
        lookback: int = 126,
        skip: int = 21,
        k: int = 6,
        vol_lookback: int = 21,
        rebalance_days: int = 21,
        per_name_cap: float = 0.25,
        target_gross: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.skip = skip
        self.k = k
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.per_name_cap = per_name_cap
        self.target_gross = target_gross
        # Need enough bars for the momentum window plus a little vol history.
        self.warmup = lookback + vol_lookback + 5
        # Persisted book + age so we only churn every ``rebalance_days``.
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Reuse last book on non-rebalance days => near-zero turnover between resets.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)

        # --- score: cross-sectional z-score of slow momentum (today's row) ---
        mom = momentum(close, lookback=self.lookback, skip=self.skip)
        score = cross_sectional_zscore(mom).iloc[-1]

        # Only names that are priced today and have a finite score are eligible.
        last_close = close.iloc[-1]
        eligible = score[score.notna() & last_close.notna()]
        if eligible.empty:
            return {}

        # Top-k by momentum score (descending).
        winners = list(eligible.sort_values(ascending=False).index[: self.k])
        if not winners:
            return {}

        # --- inverse-vol sizing among the winners ---
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        inv_vol: dict[str, float] = {}
        for s in winners:
            v = vol.get(s, np.nan)
            # Fall back to equal weight if vol is missing/zero.
            inv_vol[s] = 1.0 / v if (np.isfinite(v) and v > 0) else np.nan

        if not np.isfinite(list(inv_vol.values())).any():
            raw = {s: 1.0 for s in winners}
        else:
            med = np.nanmedian([x for x in inv_vol.values() if np.isfinite(x)])
            raw = {s: (x if np.isfinite(x) else med) for s, x in inv_vol.items()}

        total = sum(raw.values())
        if total <= 0:
            return {}
        weights = {s: self.target_gross * x / total for s, x in raw.items()}

        # Per-name cap, then renormalize the survivors back toward target gross.
        weights = {s: min(w, self.per_name_cap) for s, w in weights.items()}
        capped_total = sum(weights.values())
        if capped_total <= 0:
            return {}
        scale = min(1.0, self.target_gross / capped_total)
        weights = {s: w * scale for s, w in weights.items()}

        self._held = weights
        self._age = 1
        return dict(weights)
