"""Donchian channel breakout (Turtle trend following).

A range-expansion trend strategy in the classic Turtle / Donchian mould. For each
ETF we maintain two trailing channels off the close:

  * the ``entry_lookback`` (~55-100 day) **range high** — the highest close over
    that window, and
  * the ``exit_lookback`` (~20-day) **range low** — the lowest close over that
    window.

A name ENTERS long the day its close prints a *new* range high — a breakout out of
consolidation that signals trend ignition. It then STAYS long, riding the trend,
until its close breaks back below the range low, at which point the position is
closed to cash (the Turtle stop). Because the breakout state only flips when price
crosses a multi-week extreme, the held set changes slowly: most days the book is
unchanged, so turnover stays low even though we recompute every session.

We hold the set of names currently in an active breakout, cap the book at
``max_names`` (preferring the freshest/strongest breakouts by distance above their
entry channel), inverse-vol size them so one jumpy name can't dominate, and scale to
``target_gross`` with a per-name cap of 0.40. When nothing is breaking out the book
sits in cash.

thesis: range expansion to new multi-week highs signals trend continuation — markets
that break out of a consolidation tend to keep going, harvesting the trend / momentum
risk premium, while the range-low stop bounds the downside on each name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class DonchianBreakout(Strategy):
    """Turtle-style Donchian breakout: go long names whose close makes a new
    ``entry_lookback``-day high, hold until the close breaks the ``exit_lookback``-day
    low, then exit to cash. Inverse-vol weighted, ``max_names`` cap, per-name cap 0.40,
    scaled to ``target_gross``."""

    #: Economic rationale (read by the scorecard's Clear-thesis gate).
    thesis = (
        "Range expansion to new multi-week highs signals trend continuation: markets "
        "that break out of consolidation tend to keep going, harvesting the trend / "
        "momentum risk premium, while the range-low stop bounds per-name downside."
    )

    def __init__(
        self,
        entry_lookback: int = 100,
        exit_lookback: int = 20,
        max_names: int = 10,
        vol_lookback: int = 63,
        rebalance_days: int = 10,
        target_gross: float = 1.0,
        max_weight: float = 0.40,
    ) -> None:
        self.entry_lookback = entry_lookback
        self.exit_lookback = exit_lookback
        self.max_names = max_names
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.max_weight = max_weight
        # Need enough bars for the longest channel plus the vol window.
        self.warmup = max(entry_lookback, exit_lookback, vol_lookback) + 5
        # Persistent breakout state: the set of names currently "in a trade".
        self._active: set[str] = set()
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        last = close.iloc[-1]

        # --- Update the breakout state every session (cheap, drives the stops) ----
        # Entry channel: prior highest close over entry_lookback (exclude today so a
        # *new* high is a genuine break above the established range).
        entry_high = close.shift(1).rolling(self.entry_lookback).max().iloc[-1]
        # Exit channel: prior lowest close over exit_lookback (likewise excl. today).
        exit_low = close.shift(1).rolling(self.exit_lookback).min().iloc[-1]

        for s in close.columns:
            px = last.get(s, np.nan)
            if np.isnan(px):
                # Unpriced name: it cannot be in an active trade.
                self._active.discard(s)
                continue
            hi = entry_high.get(s, np.nan)
            lo = exit_low.get(s, np.nan)
            if s in self._active:
                # Stop out when the close breaks the trailing range low.
                if not np.isnan(lo) and px < lo:
                    self._active.discard(s)
            else:
                # Enter when the close makes a new range high.
                if not np.isnan(hi) and px > hi:
                    self._active.add(s)

        # --- Only rebuild the weight book every rebalance_days sessions -----------
        # The active set is maintained daily (so stops are honoured promptly), but we
        # only re-solve the sized portfolio periodically to keep turnover low.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            # Drop any name that has since stopped out; keep the rest as-is.
            survivors = {s: w for s, w in self._held.items() if s in self._active}
            if survivors == self._held:
                return dict(self._held)
            self._held = survivors
            return dict(survivors)
        self._age = 1

        active = [s for s in self._active if not np.isnan(last.get(s, np.nan))]
        if not active:
            self._held = {}
            return {}

        # If more names are breaking out than we can hold, prefer the strongest
        # breakouts: those whose close sits furthest above their entry channel.
        if len(active) > self.max_names:
            strength = {}
            for s in active:
                hi = entry_high.get(s, np.nan)
                px = last[s]
                strength[s] = (px / hi - 1.0) if (not np.isnan(hi) and hi > 0) else 0.0
            active = sorted(active, key=lambda s: strength[s], reverse=True)[: self.max_names]

        # Inverse-vol sizing so a single jumpy name doesn't dominate the book.
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        raw: dict[str, float] = {}
        for s in active:
            v = vol.get(s, np.nan)
            inv = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
            raw[s] = inv

        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        # Normalize to target gross, then enforce the per-name cap and redistribute
        # the spilled weight across the remaining names (keeps gross ~target).
        weights = {s: self.target_gross * r / total for s, r in raw.items()}
        weights = self._apply_cap(weights)
        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float]) -> dict[str, float]:
        """Clip any name above ``max_weight`` and redistribute the excess across the
        uncapped names, iterating until no name exceeds the cap."""
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
                w[s] += excess * room[s] / base
        return w
