"""Defensive Asset Allocation (Keller & Keuning DAA).

A "canary"-gated tactical model. A small set of crash-sensitive ETFs (the canaries,
here emerging-market equity ``EEM`` and aggregate bonds ``AGG``) act as an early
warning system for risk-off regimes. Each decision day we:

  1. Score every name by a 13612W momentum: the weighted average of its 1/3/6/12-month
     total returns (weights 12/4/2/1), the canonical DAA "fast" momentum measure that
     reacts to recent moves while still respecting the longer trend.
  2. Count how many canaries have *negative* 13612W momentum. That count sets the
     "cash fraction": each bad canary tips a proportional slice of the book out of
     risky assets and into a defensive bond sleeve. With two canaries, 1 bad => 50%
     defensive, 2 bad => 100% defensive (a full flight to bonds).
  3. Allocate the remaining "risk-on" fraction across the top ``k`` risky ETFs by the
     same 13612W momentum (only those with positive momentum count), inverse-vol
     weighted so one jumpy name doesn't dominate.
  4. Route the defensive fraction into the best of a small bond set (IEF/AGG/TLT) by
     its own 13612W momentum — "crash alpha" by holding whichever safe asset is itself
     trending up.

The book is recomputed only every ``rebalance_days`` sessions (a self counter) and the
underlying signal is a slow multi-month blend, so day-to-day turnover stays low — the
strategy holds a near-static handful of names for weeks at a time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...data_pipeline.features import realized_vol
from ..base import History, Strategy

#: Canary (crash-detector) set: a risk-sensitive equity proxy and a broad bond proxy.
#: When either rolls over, DAA scales the book toward safety.
CANARIES: tuple[str, ...] = ("EEM", "AGG")

#: Defensive sleeve: the safe assets the book flees into. The best by 13612W momentum
#: is chosen each rebalance, so we park in whichever bond is itself trending.
DEFENSIVE: tuple[str, ...] = ("IEF", "AGG", "TLT")

#: Risky universe: broad equity, sectors, regions, real assets. Bonds are excluded so
#: the risk-on sleeve is genuinely "risk-on"; the defensive sleeve covers safety.
RISKY: tuple[str, ...] = (
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "EFA", "EEM", "VEA", "VWO", "EWJ",
    "GLD", "SLV", "DBC", "VNQ",
)

#: 13612W lookbacks (trading days) and their weights: 1m/3m/6m/12m at 12/4/2/1.
_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
_MOM_WEIGHTS: tuple[float, ...] = (12.0, 4.0, 2.0, 1.0)


def _momentum_13612w(close: pd.DataFrame) -> pd.Series:
    """13612W momentum for the most recent row: weighted average of the trailing
    1/3/6/12-month total returns (weights 12/4/2/1). Uses only closes up to today.
    Names without enough history for any leg come out NaN and are skipped downstream.
    """
    last = close.iloc[-1]
    num = pd.Series(0.0, index=close.columns)
    wsum = sum(_MOM_WEIGHTS)
    for lb, w in zip(_LOOKBACKS, _MOM_WEIGHTS):
        ref = close.iloc[-1 - lb] if len(close) > lb else pd.Series(np.nan, index=close.columns)
        num = num + w * (last / ref - 1.0)
    return num / wsum


class DefensiveAssetAllocation(Strategy):
    """Canary-gated DAA: 13612W momentum drives a top-``k`` risk-on sleeve, while the
    number of negative-momentum canaries (EEM/AGG) proportionally scales the book into a
    momentum-best bond sleeve (IEF/AGG/TLT). Inverse-vol weighted, rebalanced ~21d."""

    def __init__(
        self,
        k: int = 4,
        vol_lookback: int = 63,
        rebalance_days: int = 21,
        canaries: tuple[str, ...] = CANARIES,
        defensive: tuple[str, ...] = DEFENSIVE,
        risky: tuple[str, ...] = RISKY,
        target_gross: float = 1.0,
    ) -> None:
        self.k = k
        self.vol_lookback = vol_lookback
        self.rebalance_days = rebalance_days
        self.canaries = canaries
        self.defensive = defensive
        self.risky = risky
        self.target_gross = target_gross
        # Need the longest 13612W leg (12 months) plus a small cushion.
        self.warmup = max(_LOOKBACKS) + 5
        self._held: dict[str, float] = {}
        self._age = 0

    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Only refresh the book every ``rebalance_days`` sessions; otherwise return the
        # previously chosen weights verbatim to keep turnover near zero.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        mom = _momentum_13612w(close)
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        last = close.iloc[-1]

        def priced(sym: str) -> bool:
            return not np.isnan(last.get(sym, np.nan)) and not np.isnan(mom.get(sym, np.nan))

        # --- Canary gate: count bad (non-positive momentum) canaries that are priced. ---
        live_canaries = [c for c in self.canaries if priced(c)]
        if not live_canaries:
            # Can't read the crash detector yet — sit fully defensive (or cash).
            bad_frac = 1.0
        else:
            bad = sum(1 for c in live_canaries if mom[c] <= 0.0)
            bad_frac = bad / len(live_canaries)

        risk_on_frac = (1.0 - bad_frac) * self.target_gross
        defensive_frac = bad_frac * self.target_gross

        raw: dict[str, float] = {}

        # --- Risk-on sleeve: top-k risky names with positive 13612W momentum. ---
        if risk_on_frac > 0.0:
            cands = [s for s in self.risky if priced(s) and mom[s] > 0.0]
            ranked = sorted(cands, key=lambda s: mom[s], reverse=True)[: self.k]
            inv = self._inverse_vol(ranked, vol)
            tot = sum(inv.values())
            if tot > 0.0:
                for s, iv in inv.items():
                    raw[s] = raw.get(s, 0.0) + risk_on_frac * iv / tot
            else:
                # No risky name clears the trend hurdle — divert this slice to defense.
                defensive_frac += risk_on_frac

        # --- Defensive sleeve: the single best bond by its own 13612W momentum. ---
        if defensive_frac > 0.0:
            bonds = [s for s in self.defensive if priced(s)]
            if bonds:
                best = max(bonds, key=lambda s: mom[s])
                raw[best] = raw.get(best, 0.0) + defensive_frac
            # else: no bond priced -> that slice stays in cash.

        if not raw:
            self._held = {}
            return {}

        self._held = dict(raw)
        return dict(raw)

    def _inverse_vol(self, syms: list[str], vol: pd.Series) -> dict[str, float]:
        """Inverse-volatility weights for ``syms``; falls back to equal weight where the
        trailing vol is missing or non-positive."""
        out: dict[str, float] = {}
        for s in syms:
            v = vol.get(s, np.nan)
            out[s] = 1.0 / v if (isinstance(v, float) and v > 1e-8 and not np.isnan(v)) else 1.0
        return out
