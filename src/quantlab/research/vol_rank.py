"""Volatility-rank mapping: where does *today's* market volatility sit in its own history?

The orchestrator uses this to decide which volatility regime we're in and therefore which
of the three vol-regime strategies should drive the book. We rank the benchmark's current
realized volatility as a percentile of its trailing history (point-in-time, no lookahead):
0 = calmest in recent memory, 1 = most turbulent. That percentile maps to a low/mid/high
bucket and to soft membership weights so the orchestrator can blend across bucket boundaries
instead of hard-switching (which would whipsaw).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def realized_vol_series(prices: pd.Series, vol_lookback: int = 20) -> pd.Series:
    """Trailing annualized realized vol of a price series (point-in-time)."""
    logp = cast(pd.Series, np.log(prices.astype(float)))
    return logp.diff().rolling(vol_lookback).std() * np.sqrt(252)


def vol_rank(prices: pd.Series, vol_lookback: int = 20, rank_window: int = 252) -> float:
    """Percentile rank in [0, 1] of the latest realized vol within the trailing
    ``rank_window``. Uses only data up to the last row."""
    vol = realized_vol_series(prices, vol_lookback).dropna()
    if len(vol) < 2:
        return 0.5
    window = vol.iloc[-rank_window:]
    current = float(vol.iloc[-1])
    return float((window <= current).mean())


def vol_bucket(rank: float, low: float = 0.34, high: float = 0.66) -> str:
    """Map a vol percentile to a coarse bucket."""
    if rank < low:
        return "low"
    if rank < high:
        return "mid"
    return "high"


def bucket_membership(rank: float) -> dict[str, float]:
    """Soft membership over {low, mid, high} as a function of the vol percentile, using
    overlapping triangular windows centered at 0.17 / 0.50 / 0.83. Returns weights summing
    to 1 — lets the orchestrator blend adjacent strategies near a boundary instead of
    flipping discretely."""
    centers = {"low": 0.17, "mid": 0.50, "high": 0.83}
    # Triangular kernel with half-width 0.33 (adjacent centers overlap).
    raw = {k: max(0.0, 1.0 - abs(rank - c) / 0.33) for k, c in centers.items()}
    total = sum(raw.values())
    if total <= 0:
        return {"low": 0.0, "mid": 1.0, "high": 0.0}
    return {k: v / total for k, v in raw.items()}
