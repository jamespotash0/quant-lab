"""High-vol defensive sleeve.

The job of this sleeve is to *avoid big drawdowns*. It holds a deliberately
conservative, low-exposure book tilted toward the lowest-volatility, "safe"
assets in the universe — bonds (TLT/IEF/AGG/LQD), defensive equity sectors
(XLP/XLU/XLV) and gold (GLD) — and it cuts exposure further the more turbulent
the market becomes.

Each decision day we:

  1. Read the market-wide volatility regime from the benchmark's trailing
     realized vol versus its own slower baseline. Calm tape -> run closer to
     ``target_gross``; turbulent tape -> shrink gross toward a hard floor and
     park the rest in cash. The whole point is that risk comes *off* exactly
     when realized vol spikes (the regime that precedes and accompanies
     crashes).
  2. From the safe-asset shortlist, keep the ``n_names`` with the *lowest*
     realized volatility (low-vol tilt), and inverse-vol size them so a single
     jumpy name never dominates the book.
  3. Scale the sleeve to the regime-adjusted gross (always well below 1.0, with
     a large cash buffer), and cap any single name at ``max_weight``.

The book is recomputed only every ``rebalance_days`` sessions via a self
counter, and the underlying signals (slow realized vol, slow regime filter)
barely move week to week, so turnover stays low.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import momentum, realized_vol
from ..base import History, Strategy

#: Benchmark whose realized vol defines the market-wide regime.
REGIME_SYMBOL = "SPY"

#: Defensive shortlist: bonds, defensive sectors, gold. The sleeve only ever
#: holds names from this list, so it is structurally low-beta / crash-resistant.
DEFAULT_SAFE_ASSETS: tuple[str, ...] = (
    "TLT", "IEF", "AGG", "LQD",   # bonds / rates / credit
    "XLP", "XLU", "XLV",          # defensive equity sectors
    "GLD",                        # gold
)


class HighVolDefensiveStrategy(Strategy):
    """Low-exposure defensive sleeve: holds the lowest-realized-vol safe assets
    (bonds, defensive sectors, gold), inverse-vol sized, with a large cash
    buffer that grows as benchmark volatility spikes. Rebalanced every ~15
    sessions; per-name cap keeps the book diversified."""

    thesis = (
        "High volatility precedes and accompanies drawdowns; cutting exposure "
        "and rotating into low-vol, safe assets (bonds, defensive sectors, gold) "
        "during turbulent regimes preserves capital, so you compound from a "
        "higher base instead of digging out of deep holes."
    )

    def __init__(
        self,
        n_names: int = 4,
        vol_lookback: int = 63,
        rebalance_days: int = 15,
        target_gross: float = 0.55,
        gross_floor: float = 0.25,
        max_weight: float = 0.25,
        regime_fast: int = 21,
        regime_slow: int = 126,
        trend_lookback: int = 63,
        trend_skip: int = 5,
        safe_assets: tuple[str, ...] = DEFAULT_SAFE_ASSETS,
    ) -> None:
        self.n_names = n_names
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.gross_floor = gross_floor
        self.max_weight = max_weight
        self.regime_fast = regime_fast
        self.regime_slow = regime_slow
        self.trend_lookback = trend_lookback
        self.trend_skip = trend_skip
        self.safe_assets = tuple(safe_assets)
        # Need enough bars for the slowest trailing window.
        self.warmup = max(vol_lookback, regime_slow, trend_lookback + trend_skip) + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def _regime_gross(self, close: pd.DataFrame) -> float:
        """Map the benchmark vol regime to a target gross exposure between
        ``gross_floor`` (turbulent) and ``target_gross`` (calm).

        Ratio = fast realized vol / slow realized vol. <=1 means the tape is no
        more turbulent than its baseline -> run full (defensive) gross. As the
        ratio climbs above 1 we linearly shrink gross toward the floor, hitting
        the floor once fast vol is ~2x its baseline.
        """
        if REGIME_SYMBOL not in close.columns:
            return self.target_gross
        px = close[REGIME_SYMBOL].dropna()
        if len(px) < self.regime_slow + 2:
            return self.target_gross
        rets = px.pct_change()
        fast = rets.tail(self.regime_fast).std()
        slow = rets.tail(self.regime_slow).std()
        if not np.isfinite(fast) or not np.isfinite(slow) or slow <= 1e-12:
            return self.target_gross
        ratio = fast / slow
        # 1.0 -> calm (full gross); 2.0 -> turbulent (floor). Clamp to [0, 1].
        stress = min(max((ratio - 1.0), 0.0), 1.0)
        return self.target_gross - stress * (self.target_gross - self.gross_floor)

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only refresh the book every ``rebalance_days`` sessions; otherwise
        # return the previously chosen weights verbatim to keep turnover low.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        gross = self._regime_gross(close)

        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        trend = momentum(
            close, lookback=self.trend_lookback, skip=self.trend_skip
        ).iloc[-1]
        last = close.iloc[-1]

        # Candidate safe assets: priced today, usable vol, AND in an uptrend.
        # The absolute-trend gate is the key defensive twist — a "safe" asset
        # that is itself falling (e.g. bonds in a rate-hike crash) is not safe,
        # so we refuse to catch the falling knife and let that sleeve sit in cash.
        candidates: list[tuple[str, float]] = []
        for s in self.safe_assets:
            if s not in close.columns:
                continue
            v = vol.get(s, np.nan)
            p = last.get(s, np.nan)
            t = trend.get(s, np.nan)
            if (
                isinstance(v, float)
                and np.isfinite(v)
                and v > 1e-8
                and isinstance(p, float)
                and np.isfinite(p)
                and isinstance(t, float)
                and np.isfinite(t)
                and t > 0.0
            ):
                candidates.append((s, v))

        if not candidates:
            # Nothing safe is trending up -> stay fully in cash. Preserving
            # capital beats forcing a position into a down-trending market.
            self._held = {}
            return {}

        # Low-vol tilt: keep the n_names lowest-realized-vol eligible safe assets.
        candidates.sort(key=lambda kv: kv[1])
        chosen = candidates[: self.n_names]

        # Inverse-vol sizing within the sleeve.
        raw = {s: 1.0 / v for s, v in chosen}
        total = sum(raw.values())
        if total <= 0.0:
            self._held = {}
            return {}

        weights = {s: gross * w / total for s, w in raw.items()}

        # Per-name cap (risk mgmt). Redistribute the trimmed mass proportionally
        # across uncapped names, iterating until stable.
        weights = self._apply_cap(weights, gross)

        self._held = weights
        return dict(weights)

    def _apply_cap(self, weights: dict[str, float], gross: float) -> dict[str, float]:
        """Cap each name at ``max_weight`` and push the excess into the other
        (uncapped) names, preserving the total gross where feasible."""
        cap = self.max_weight
        w = dict(weights)
        for _ in range(len(w) + 1):
            over = {s: v for s, v in w.items() if v > cap + 1e-12}
            if not over:
                break
            excess = sum(v - cap for v in over.values())
            for s in over:
                w[s] = cap
            uncapped = {s: v for s, v in w.items() if v < cap - 1e-12}
            base = sum(uncapped.values())
            if base <= 0.0 or excess <= 0.0:
                break
            for s in uncapped:
                w[s] += excess * uncapped[s] / base
        # Final hard clamp so no name can ever exceed the cap due to rounding.
        return {s: min(v, cap) for s, v in w.items()}
