"""Strategy v1: cross-sectional momentum + short-term reversal, with a vol-regime
filter and volatility-targeted sizing.

This is deliberately simple and transparent (see docs/00-PLAN.md section 2): we can
explain *why* it works or doesn't. It is a hypothesis to try to kill, not a belief.

Logic, evaluated point-in-time on each decision day:
  1. Score each name = z(momentum 126/21) + reversal_weight * z(5-day reversal).
  2. Go long the top ``n_long`` names by score (long-only to start).
  3. Size each position inversely to its realized vol (equal risk contribution),
     then scale so gross exposure = ``target_gross``.
  4. Regime filter: if the benchmark is below its ``regime_ma`` moving average
     (risk-off), hold cash instead of trading.
"""

from __future__ import annotations

import numpy as np

from ..data_pipeline import features as F
from .base import History, Strategy


class MomentumReversal(Strategy):
    def __init__(
        self,
        n_long: int = 5,
        mom_lookback: int = 126,
        mom_skip: int = 21,
        rev_lookback: int = 5,
        reversal_weight: float = 0.5,
        vol_lookback: int = 21,
        target_gross: float = 1.0,
        max_weight: float = 0.25,
        regime_symbol: str = "SPY",
        regime_ma: int = 200,
    ) -> None:
        self.n_long = n_long
        self.mom_lookback = mom_lookback
        self.mom_skip = mom_skip
        self.rev_lookback = rev_lookback
        self.reversal_weight = reversal_weight
        self.vol_lookback = vol_lookback
        self.target_gross = target_gross
        self.max_weight = max_weight
        self.regime_symbol = regime_symbol
        self.regime_ma = regime_ma
        # Need enough history for the longest lookback plus the regime MA.
        self.warmup = max(mom_lookback, regime_ma) + 5

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # --- regime filter -------------------------------------------------------
        if self.regime_symbol in close.columns:
            bench = close[self.regime_symbol].dropna()
            if len(bench) >= self.regime_ma:
                ma = bench.iloc[-self.regime_ma :].mean()
                if bench.iloc[-1] < ma:
                    return {}  # risk-off: sit in cash

        # --- score ---------------------------------------------------------------
        mom = F.momentum(close, self.mom_lookback, self.mom_skip)
        rev = F.short_term_reversal(close, self.rev_lookback)
        score = F.cross_sectional_zscore(mom) + self.reversal_weight * F.cross_sectional_zscore(rev)
        latest = score.iloc[-1].dropna()
        # Don't let the benchmark dominate its own signal universe oddly — it's fine to
        # hold SPY, so we keep it eligible.
        if latest.empty:
            return {}

        winners = latest.sort_values(ascending=False).head(self.n_long).index.tolist()
        if not winners:
            return {}

        # --- volatility-targeted sizing -----------------------------------------
        vol = F.realized_vol(close, self.vol_lookback).iloc[-1]
        inv_vol = {}
        for sym in winners:
            v = vol.get(sym, np.nan)
            inv_vol[sym] = 1.0 / v if (v and v > 0 and not np.isnan(v)) else 0.0

        total = sum(inv_vol.values())
        if total <= 0:
            # Fall back to equal weight if vols are unusable.
            w = self.target_gross / len(winners)
            raw = {s: w for s in winners}
        else:
            raw = {s: self.target_gross * iv / total for s, iv in inv_vol.items()}

        # Per-name cap, then renormalize to keep gross near target.
        capped = {s: min(w, self.max_weight) for s, w in raw.items()}
        gross = sum(capped.values())
        if gross > self.target_gross and gross > 0:
            capped = {s: w * self.target_gross / gross for s, w in capped.items()}
        return {s: w for s, w in capped.items() if w > 0}
