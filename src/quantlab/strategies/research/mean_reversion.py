"""Short-term mean reversion in index ETFs.

Index ETFs exhibit a real short-horizon overreaction effect: a basket that has just
sold off hard tends to bounce over the following days. We measure "oversoldness" as
how far each name sits below its own short trailing moving average (a per-name,
point-in-time z-score of the gap), go long the ``k`` most-oversold names, and hold
them for a few days before reselecting.

Because the underlying signal flips quickly, this family is naturally high-turnover.
We tame that two ways: (1) a rebalance counter so the book is only reselected every
``hold_days`` sessions, and (2) inverse-vol weighting within the chosen basket, which
keeps weights stable across the hold window even as prices drift.
"""

from __future__ import annotations

import numpy as np

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy


class ShortTermMeanReversion(Strategy):
    """Buy the most oversold ETFs (price far below a short moving average) and hold
    briefly. Long-only, inverse-vol weighted, reselected every ``hold_days`` days.

    Oversold score for name ``s`` is ``-(close - SMA(ma_lb)) / SMA(ma_lb)`` — the
    further price has fallen below its own recent mean, the larger the score. We rank
    cross-sectionally and take the top ``k``.
    """

    def __init__(
        self,
        k: int = 5,
        ma_lb: int = 15,
        hold_days: int = 10,
        vol_lb: int = 21,
        target_gross: float = 1.0,
    ) -> None:
        self.k = k
        self.ma_lb = ma_lb
        self.hold_days = hold_days
        self.vol_lb = vol_lb
        self.target_gross = target_gross
        # Need enough bars for the moving average and the vol estimate.
        self.warmup = max(ma_lb, vol_lb) + 2
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only reselect every ``hold_days`` sessions; otherwise re-assert the prior
        # book so day-to-day turnover stays low.
        if self._held and self._age < self.hold_days:
            self._age += 1
            return dict(self._held)

        last = close.iloc[-1]
        sma = close.tail(self.ma_lb).mean()  # per-symbol mean over trailing window

        # Oversold = how far below its own short-run mean each name trades.
        gap = (last - sma) / sma.replace(0.0, np.nan)
        oversold = -gap

        # Drop names without a valid, finite signal (leading NaNs / zero SMA).
        oversold = oversold.replace([np.inf, -np.inf], np.nan).dropna()
        priced = [s for s in oversold.index if not np.isnan(last[s])]
        oversold = oversold.loc[priced]
        if oversold.empty:
            return {}

        k = min(self.k, len(oversold))
        ranked = list(oversold.sort_values(ascending=False).index)

        # Hysteresis: keep currently-held names that are still oversold (positive
        # score) and within the broader top tier, then fill remaining slots from the
        # freshest oversold names. This caps churn at each reselection.
        tier = set(ranked[: min(2 * self.k, len(ranked))])
        kept = [s for s in self._held if s in tier and oversold.get(s, -1.0) > 0]
        chosen = list(kept)
        for s in ranked:
            if len(chosen) >= k:
                break
            if s not in chosen:
                chosen.append(s)

        # Inverse-vol weight within the basket; fall back to equal weight if vol is
        # missing or degenerate.
        vol = realized_vol(close, self.vol_lb).iloc[-1]
        inv = {}
        for s in chosen:
            v = vol.get(s, np.nan)
            inv[s] = 1.0 / v if (np.isfinite(v) and v > 0) else np.nan

        if all(not np.isfinite(x) for x in inv.values()):
            w = self.target_gross / len(chosen)
            weights = {s: w for s in chosen}
        else:
            fill = np.nanmean([x for x in inv.values() if np.isfinite(x)])
            inv = {s: (x if np.isfinite(x) else fill) for s, x in inv.items()}
            total = sum(inv.values())
            weights = {s: float(self.target_gross * x / total) for s, x in inv.items()}

        self._held = weights
        self._age = 1
        return dict(weights)
