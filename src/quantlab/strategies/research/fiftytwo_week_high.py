"""The 52-week-high effect (George & Hwang 2004).

A well-documented anomaly (catalogued by Quantpedia): an asset's *nearness to its
own trailing 52-week high* predicts its future returns. Each decision day we:

  1. For every priced name, compute its "nearness ratio" = close / (trailing
     ``lookback``-day max close), a number in (0, 1]. A ratio near 1.0 means the
     name is sitting right at its one-year high; a low ratio means it's deep below.
  2. Optionally require ratio >= ``min_ratio`` so we only ever hold names that are
     *genuinely* near their highs — names that fail the floor are dropped (cash).
  3. Rank the survivors by nearness and go LONG the top ``k`` (closest to 1.0).
  4. Inverse-vol weight the sleeve so a single jumpy name can't dominate, cap any
     one name at ``max_weight``, and normalize to ``target_gross``.

The book is recomputed only every ``rebalance_days`` sessions (a self counter), and
the underlying signal is a slow one-year anchor, so day-to-day turnover stays low —
the strategy holds a near-static handful of names for weeks at a time.

Why it should work: traders anchor on the salient 52-week high and under-react to
good news that pushes a name *toward* it, so winners near their highs keep drifting
up. This is an anchoring-driven continuation effect, distinct from (and empirically
not subsumed by) trailing-return momentum — the high is a reference point, not a
return.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class FiftyTwoWeekHigh(Strategy):
    """Go long the top-``k`` names closest to their own trailing 52-week high
    (nearness = close / 252-day max close), gated by a ``min_ratio`` floor so we
    only hold names genuinely near highs. Inverse-vol weighted, per-name capped,
    rebalanced every ~21 sessions."""

    #: Economic rationale, surfaced by the scorecard's "clear thesis" gate.
    thesis = (
        "Traders anchor on the salient 52-week high and under-react to good news "
        "near it, so ETFs sitting closest to their own one-year high keep drifting "
        "up — an anchoring-driven continuation effect distinct from trailing-return "
        "momentum (George & Hwang 2004)."
    )

    def __init__(
        self,
        lookback: int = 252,
        k: int = 4,
        min_ratio: float = 0.90,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        max_weight: float = 0.40,
        target_gross: float = 1.0,
    ) -> None:
        self.lookback = lookback
        self.k = k
        self.min_ratio = min_ratio
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.max_weight = max_weight
        self.target_gross = target_gross
        # Need a full trailing window for the 52-week max plus a little slack.
        self.warmup = lookback + 5
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

        # Nearness to the trailing 52-week high, computed point-in-time on the
        # window ending today (inclusive). Row .iloc[-1] is today's value.
        window = close.iloc[-self.lookback :]
        last = window.iloc[-1]
        high = window.max()
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]

        # Candidates: a finite current price, a positive 52-week high, and a
        # nearness ratio at or above the floor (genuinely near its own high).
        ratios: dict[str, float] = {}
        for s in close.columns:
            px = last.get(s, np.nan)
            hi = high.get(s, np.nan)
            if np.isnan(px) or np.isnan(hi) or hi <= 0.0:
                continue
            ratio = float(px) / float(hi)
            if ratio >= self.min_ratio:
                ratios[s] = ratio

        if not ratios:
            self._held = {}
            return {}

        # Rank by nearness (closest to 1.0 first) and take the top k.
        ranked = sorted(ratios, key=lambda s: ratios[s], reverse=True)
        winners = ranked[: self.k]

        # Inverse-vol weight the winners; guard against zero/NaN vol.
        raw: dict[str, float] = {}
        for s in winners:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = inv

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        # Normalize to target gross, then enforce the per-name cap. Capping can free
        # up weight; redistribute it once across the uncapped names so we stay near
        # target gross without ever breaching the cap.
        weights = {s: self.target_gross * w / total for s, w in raw.items()}
        weights = self._apply_cap(weights)

        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip every name to ``max_weight`` and redistribute the freed weight across
        the names still below the cap (single pass — sufficient for our small k)."""
        capped = {s: min(w, self.max_weight) for s, w in weights.items()}
        excess = sum(weights.values()) - sum(capped.values())
        if excess <= 1e-12:
            return capped
        room = {s: self.max_weight - w for s, w in capped.items() if w < self.max_weight}
        room_total = sum(room.values())
        if room_total <= 1e-12:
            return capped
        for s, r in room.items():
            capped[s] += excess * (r / room_total)
        return capped
