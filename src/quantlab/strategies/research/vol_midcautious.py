"""Mid-vol cautious risk-parity book with a mild momentum tilt.

This strategy is built for *normal / elevated* volatility regimes — the bulk of
market life that is neither a calm grind higher nor an outright crash. In those
conditions, going all-in (full aggression) leaves you exposed to the frequent
shocks that punctuate average-vol markets, while going all-cash (full defense)
bleeds away the equity risk premium you are paid to hold. The cautious answer is
a *broad, risk-balanced book with a cash buffer*.

Each decision day (refreshed only every ``rebalance_days`` sessions to keep
turnover low) we:

  1. Score every priced name by trailing 12-1 style momentum and keep the top
     ``n_names`` (a broad, diversified set — more names than the low-vol sleeve).
  2. Weight the survivors by *inverse volatility* (risk parity), so a single
     jumpy name cannot dominate the book — every name contributes a similar
     risk budget.
  3. Apply a *mild* multiplicative momentum tilt on top of the inverse-vol base
     so stronger trends earn a modest overweight without abandoning balance.
  4. Cap any single name at ``max_weight`` and normalize the book to
     ``target_gross`` (< 1.0), deliberately *holding cash* as a shock absorber.

The signal is a slow trend plus a slow vol estimate, recomputed roughly monthly,
so day-to-day turnover stays low and the book is near-static for weeks.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import momentum, realized_vol
from ..base import History, Strategy


class MidVolCautiousStrategy(Strategy):
    """Diversified inverse-vol (risk-parity) book over the top ``n_names``
    momentum survivors, with a mild momentum tilt, a per-name cap, and a
    deliberate cash buffer (gross < 1). Rebalanced every ~21 sessions."""

    thesis = (
        "In average-volatility regimes neither full aggression nor full defense "
        "is right — hold a broad, risk-balanced (inverse-vol) book across many "
        "names with a mild momentum tilt and a cash buffer, so no single name or "
        "shock can dominate the portfolio."
    )

    def __init__(
        self,
        n_names: int = 10,
        vol_lookback: int = 189,
        mom_weight: float = 3.0,
        rebalance_days: int = 21,
        target_gross: float = 0.85,
        max_weight: float = 0.20,
        mom_lookback: int = 252,
        mom_skip: int = 21,
    ) -> None:
        self.n_names = n_names
        self.vol_lookback = vol_lookback
        self.mom_weight = mom_weight
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.max_weight = max_weight
        self.mom_lookback = mom_lookback
        self.mom_skip = mom_skip
        # Need enough bars for the longest trailing window plus the skip offset.
        self.warmup = mom_lookback + mom_skip + 5
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

        mom = momentum(close, lookback=self.mom_lookback, skip=self.mom_skip).iloc[-1]
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        last = close.iloc[-1]

        # Candidates: priced names with a finite momentum score.
        candidates = [
            s
            for s in close.columns
            if not np.isnan(mom.get(s, np.nan)) and not np.isnan(last.get(s, np.nan))
        ]
        if not candidates:
            self._held = {}
            return {}

        # Relative momentum: keep the top n_names winners for diversification.
        ranked = sorted(candidates, key=lambda s: mom[s], reverse=True)
        winners = ranked[: self.n_names]

        # Only hold names with non-negative trend — in average-vol regimes we still
        # decline to chase outright down-trenders, which prunes the book toward cash.
        winners = [s for s in winners if mom[s] > 0.0]
        if not winners:
            self._held = {}
            return {}

        # Inverse-vol (risk-parity) base weight, guarded against zero/NaN vol,
        # then a multiplicative momentum tilt: each name's risk-parity weight is
        # scaled by a smooth function of its *cross-sectional* momentum rank, so
        # stronger trends earn a larger (but bounded) overweight. ``mom_weight``
        # controls how aggressive the tilt is; the floor keeps every survivor in
        # the book so the risk-parity diversification is preserved.
        n = len(winners)
        order = sorted(winners, key=lambda s: mom[s])  # worst -> best
        # Linear rank in [0, 1]: worst -> 0, best -> 1.
        rank01 = {s: (i / (n - 1) if n > 1 else 1.0) for i, s in enumerate(order)}
        raw: dict[str, float] = {}
        for s in winners:
            v = vol.get(s, np.nan)
            inv = (
                1.0 / v
                if (isinstance(v, float) and v > 1e-8 and not np.isnan(v))
                else 1.0
            )
            # Tilt multiplier: a small floor (0.15) keeps the weakest survivor in
            # the book, while ``mom_weight`` scales the lead the strongest trend
            # earns. Concentrates risk toward winners without zeroing diversifiers.
            tilt = 0.15 + self.mom_weight * rank01[s]
            raw[s] = inv * max(0.0, tilt)

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        weights = {s: self.target_gross * w / total for s, w in raw.items()}

        # Per-name cap, then renormalize the *uncapped* names so total gross is
        # preserved up to the cap budget. A couple of passes is enough for our
        # small books; iterate until stable or capacity exhausted.
        weights = self._apply_cap(weights)

        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Clamp every weight to ``max_weight`` and redistribute the spillover to
        the uncapped names proportionally, preserving total gross where possible."""
        cap = self.max_weight
        w = dict(weights)
        for _ in range(8):
            over = {s: v for s, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            spill = sum(v - cap for v in over.values())
            for s in over:
                w[s] = cap
            room = {s: cap - v for s, v in w.items() if v < cap - 1e-12}
            room_total = sum(room.values())
            if room_total <= 1e-12:
                break  # no capacity left; remainder stays as extra cash.
            for s, r in room.items():
                w[s] += spill * (r / room_total)
        return w
