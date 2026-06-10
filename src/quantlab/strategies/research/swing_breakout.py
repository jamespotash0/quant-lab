"""Swing-high / swing-low pivot breakout (market-structure trading).

Classical technical traders and CTAs read price *structure*: a chart makes a
series of swing highs (local peaks) and swing lows (local troughs). A swing high
is a price PIVOT — a bar whose close exceeds the ``pivot_width`` bars on BOTH
sides; a swing low is the mirror. When price closes ABOVE its most recent
confirmed swing high, buyers have overcome the prior supply level and a new
up-leg is underway (structure break); when it closes BELOW its most recent
confirmed swing low, the up-trend's structure has failed and we exit.

Each decision day we:

  1. Detect each ETF's most recent CONFIRMED swing high and swing low. A pivot at
     time ``t`` is only *knowable* once ``pivot_width`` later bars exist (you need
     bars on the right side to confirm it is a local extremum), so we only ever
     look at pivots dated ``t <= today - pivot_width``. This is the crux of the
     no-lookahead discipline: never peek at the bars that confirm a pivot.
  2. Maintain a per-name "in up-leg" state: a name turns ON when its latest close
     breaks above its most recent confirmed swing high, and turns OFF when its
     latest close breaks below its most recent confirmed swing low. Held between.
  3. From the ON set, take up to ``max_names`` with the freshest/strongest
     breakout, inverse-vol size them (so a jumpy name doesn't dominate), cap any
     single name at 40% of the book, and normalize to ``target_gross``.

The book is recomputed only every ``rebalance_days`` sessions (a self counter),
and the breakout state is slow-moving structure, so day-to-day turnover stays
low — the strategy holds a near-static handful of names for weeks at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class SwingPivotBreakout(Strategy):
    """Long names whose latest close breaks above their most recent CONFIRMED
    swing high (a market-structure up-leg); drop a name once it closes below its
    most recent confirmed swing low. Gated by a broad-market trend regime,
    inverse-vol sized, per-name cap <=40%, rebalanced every ~21 sessions for
    low turnover."""

    #: Economic rationale (read by the scorecard's "clear thesis" gate).
    thesis = (
        "Breaking a confirmed swing high means buyers overcame the prior supply "
        "level — a market-structure signal of trend continuation that classical "
        "technical traders and CTAs both exploit; a break of the swing low marks "
        "structure failure and a clean, rules-based exit."
    )

    def __init__(
        self,
        pivot_width: int = 5,
        max_names: int = 6,
        target_gross: float = 1.0,
        rebalance_days: int = 21,
        vol_lookback: int = 63,
        max_weight: float = 0.40,
        lookback: int = 252,
        regime_ma: int = 150,
        regime_symbol: str = "SPY",
    ) -> None:
        self.pivot_width = pivot_width
        self.max_names = max_names
        self.target_gross = target_gross
        self.rebalance_days = rebalance_days
        self.vol_lookback = vol_lookback
        self.max_weight = max_weight
        # How far back to scan for pivots when recomputing structure each rebalance.
        self.lookback = lookback
        # Risk-off filter: when the broad market trades below its long MA, breakouts
        # are far more likely to be bull-traps, so we de-risk to cash.
        self.regime_ma = regime_ma
        self.regime_symbol = regime_symbol
        # Need enough bars to (a) find a confirmed pivot, (b) measure vol, (c) the MA.
        self.warmup = max(vol_lookback, regime_ma, 2 * pivot_width + 5) + 5
        self._held: dict[str, float] = {}
        # Persistent per-name up-leg state, carried across rebalances.
        self._in_upleg: dict[str, bool] = {}
        self._age = 0

    def _confirmed_pivots(self, prices: np.ndarray) -> tuple[float | None, float | None]:
        """Return (last confirmed swing high, last confirmed swing low) for a single
        name's close series, scanning newest-to-oldest.

        A pivot at index ``i`` is confirmed only when ``pivot_width`` bars exist on
        BOTH sides, so the newest candidate index is ``n - 1 - pivot_width`` — we
        never inspect a pivot that hasn't been confirmed by later bars (no peeking).
        """
        n = len(prices)
        w = self.pivot_width
        swing_high: float | None = None
        swing_low: float | None = None
        # Newest confirmable pivot is at n-1-w (it has w bars to its right).
        lo = max(w, n - 1 - self.lookback)
        for i in range(n - 1 - w, lo - 1, -1):
            p = prices[i]
            if np.isnan(p):
                continue
            left = prices[i - w : i]
            right = prices[i + 1 : i + 1 + w]
            if len(left) < w or len(right) < w:
                continue
            if np.isnan(left).any() or np.isnan(right).any():
                continue
            if swing_high is None and p > left.max() and p > right.max():
                swing_high = float(p)
            if swing_low is None and p < left.min() and p < right.min():
                swing_low = float(p)
            if swing_high is not None and swing_low is not None:
                break
        return swing_high, swing_low

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

        last = close.iloc[-1]
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]

        # Broad-market regime gate: if the market proxy is below its long moving
        # average, treat breakouts as suspect and stand aside in cash. This caps
        # drawdown by sitting out sustained bears (2018Q4, 2020, 2022).
        risk_on = True
        proxy = close.get(self.regime_symbol)
        if proxy is not None and len(proxy) >= self.regime_ma:
            ma = float(proxy.iloc[-self.regime_ma:].mean())
            px_proxy = float(proxy.iloc[-1])
            if not np.isnan(px_proxy) and not np.isnan(ma):
                risk_on = px_proxy > ma
        if not risk_on:
            # Flip every name OFF so structure must re-confirm when the regime heals,
            # then go to cash for this rebalance cycle.
            for s in close.columns:
                self._in_upleg[s] = False
            self._held = {}
            return {}

        # Update per-name up-leg state from confirmed market structure.
        # ``strength`` ranks ON names by how far they've cleared the broken high.
        strength: dict[str, float] = {}
        for s in close.columns:
            px = last.get(s, np.nan)
            if np.isnan(px):
                self._in_upleg[s] = False
                continue
            series = close[s].to_numpy()
            hi, lo = self._confirmed_pivots(series)

            state = self._in_upleg.get(s, False)
            # Structure break UP: close clears the most recent confirmed swing high.
            if hi is not None and px > hi:
                state = True
            # Structure failure: close breaks the most recent confirmed swing low.
            if lo is not None and px < lo:
                state = False
            self._in_upleg[s] = state

            if state and hi is not None:
                # Distance above the broken supply level = breakout momentum.
                strength[s] = (px - hi) / hi if hi > 0 else 0.0

        if not strength:
            self._held = {}
            return {}

        # Take the strongest up-leg names.
        ranked = sorted(strength, key=lambda s: strength[s], reverse=True)
        chosen = ranked[: self.max_names]

        # Inverse-vol weight; guard against zero/NaN vol.
        raw: dict[str, float] = {}
        for s in chosen:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = inv

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        # Normalize to target gross, then enforce the per-name cap and renormalize
        # the spillover onto the uncapped names.
        weights = {s: self.target_gross * w / total for s, w in raw.items()}
        weights = self._apply_cap(weights)
        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp any single name to ``max_weight`` of the book and redistribute the
        excess pro-rata to the names still below the cap (iterating to convergence)."""
        cap = self.max_weight * self.target_gross
        w = dict(weights)
        for _ in range(len(w) + 1):
            over = {s: v for s, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for s in over:
                w[s] = cap
            room = {s: v for s, v in w.items() if v < cap - 1e-12}
            base = sum(room.values())
            if base <= 0.0:
                break
            for s in room:
                w[s] += excess * room[s] / base
        return w
