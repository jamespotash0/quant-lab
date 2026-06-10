"""Sanity baselines. A real edge must beat these with comparable exposure."""

from __future__ import annotations

import numpy as np

from .base import History, Strategy


class AlwaysLong(Strategy):
    """Equal-weight, fully-invested in the whole universe, rebalanced daily. The
    'dumb beta' baseline: most of any strategy's return is usually just this."""

    def __init__(self, target_gross: float = 1.0) -> None:
        self.target_gross = target_gross

    def target_weights(self, history: History) -> dict[str, float]:
        priced = [s for s in history.symbols if not np.isnan(history.close[s].iloc[-1])]
        if not priced:
            return {}
        w = self.target_gross / len(priced)
        return {s: w for s in priced}


class RandomEntry(Strategy):
    """Hold a random subset of the universe, reselected every ``hold_days``. Seeded so
    a run is reproducible. If your 'edge' can't beat this coin flip at the same gross
    exposure, it isn't an edge."""

    def __init__(self, n: int = 5, hold_days: int = 5, seed: int = 0, target_gross: float = 1.0) -> None:
        self.n = n
        self.hold_days = hold_days
        self.target_gross = target_gross
        self._rng = np.random.default_rng(seed)
        self._held: list[str] = []
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        priced = [s for s in history.symbols if not np.isnan(history.close[s].iloc[-1])]
        if not priced:
            return {}
        if not self._held or self._age >= self.hold_days:
            k = min(self.n, len(priced))
            self._held = list(self._rng.choice(priced, size=k, replace=False))
            self._age = 0
        self._age += 1
        w = self.target_gross / len(self._held)
        return {s: w for s in self._held}
