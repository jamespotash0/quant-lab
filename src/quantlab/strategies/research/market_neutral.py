"""Market-neutral cross-sectional momentum (long/short).

Every long-only strategy in this lab is mostly a bet on the market itself (beta): ~70% of
its return is just "being in equities." This strategy strips the beta out to measure the
*pure signal skill*. Each rebalance we score the universe on cross-sectional momentum, go
LONG the top names and SHORT the bottom names in equal dollar amounts (dollar-neutral), so
the broad market's up-and-down cancels and what's left is the spread between winners and
losers — the momentum premium in isolation.

If this has a positive, cost-surviving Sharpe, the ranking signal has genuine skill (not
just beta). If it doesn't, the long-only strategies' returns were mostly market exposure.

Shorting is real here: the engine is share-based, so negative weights become short
positions with correct cash/marking. Gross exposure = |long| + |short| ~ target_gross;
net exposure ~ 0.
"""

from __future__ import annotations

import numpy as np

from ..base import History, Strategy
from ...data_pipeline import features as F


class MarketNeutralMomentum(Strategy):
    """Dollar-neutral long/short: long the top ``n_side`` momentum names, short the bottom
    ``n_side``, inverse-vol sized within each leg, rebalanced every ``rebalance_days``."""

    thesis = (
        "Cross-sectional momentum (12-1) has skill independent of market beta: winners keep "
        "winning and losers keep losing relative to each other. Going long top and short "
        "bottom in equal dollars isolates that spread and removes market risk."
    )

    def __init__(
        self,
        n_side: int = 6,
        lookback: int = 252,
        skip: int = 21,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        target_gross: float = 1.0,
        max_weight: float = 0.20,
    ) -> None:
        self.n_side = n_side
        self.lookback = lookback
        self.skip = skip
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.max_weight = max_weight
        self.warmup = lookback + skip + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def _leg(self, names: list[str], vol, sign: float) -> dict[str, float]:
        """Inverse-vol weight one leg to a total of ``sign * target_gross/2``."""
        inv = {}
        for s in names:
            v = vol.get(s, np.nan)
            inv[s] = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
        total = sum(inv.values())
        if total <= 0:
            return {}
        budget = self.target_gross / 2.0
        raw = {s: budget * iv / total for s, iv in inv.items()}
        # Per-name cap, then renormalize the leg back to its budget.
        capped = {s: min(w, self.max_weight) for s, w in raw.items()}
        scale = budget / sum(capped.values()) if sum(capped.values()) > 0 else 0.0
        return {s: sign * w * scale for s, w in capped.items()}

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        mom = F.momentum(close, self.lookback, self.skip).iloc[-1].dropna()
        if len(mom) < 2 * self.n_side:
            self._held = {}
            return {}
        vol = F.realized_vol(close, self.vol_lookback).iloc[-1]

        ranked = mom.sort_values(ascending=False).index.tolist()
        longs = ranked[: self.n_side]
        shorts = ranked[-self.n_side :]

        weights = {**self._leg(longs, vol, +1.0), **self._leg(shorts, vol, -1.0)}
        self._held = weights
        return dict(weights)
