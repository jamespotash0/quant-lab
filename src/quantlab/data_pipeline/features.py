"""Point-in-time features.

Every function here is a pure transform of a price panel. The *engine* is what
guarantees no-lookahead: it only ever hands the strategy a history slice ending at the
decision date. But we add a second layer of safety here — momentum/return/vol features
are defined so that a feature value dated ``t`` uses only prices up to and including
``t``. There is no centering, no future window, no full-sample normalization.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    """Simple close-to-close returns. Row t uses close[t] and close[t-1]."""
    return close.pct_change()


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    # np.log dispatches through DataFrame.__array_ufunc__ and returns a DataFrame at
    # runtime; numpy's ufunc stub types it as ndarray, so narrow it back.
    return cast(pd.DataFrame, np.log(close / close.shift(1)))


def momentum(close: pd.DataFrame, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Cross-sectional momentum: total return over the trailing ``lookback`` days,
    skipping the most recent ``skip`` days (the classic 12-1 style construction that
    skips the short-term reversal window). Row t uses only closes up to t.

    Default 126/21 ≈ "6-month return, skip last month" on a daily calendar.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def short_term_reversal(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Short-term reversal signal: the *negative* of the trailing ``lookback``-day
    return. Recent losers tend to bounce; we go long low (negative) recent returns, so
    we flip the sign. Row t uses only closes up to t.
    """
    return -(close / close.shift(lookback) - 1.0)


def realized_vol(close: pd.DataFrame, lookback: int = 21) -> pd.DataFrame:
    """Trailing realized volatility (std of daily returns) annualized. Used for
    volatility-targeted position sizing. Row t uses returns up to t.
    """
    rets = daily_returns(close)
    return rets.rolling(lookback).std() * np.sqrt(252)


def cross_sectional_zscore(feature: pd.DataFrame) -> pd.DataFrame:
    """Z-score each row across the universe (cross-sectional, point-in-time per row).
    Ranks names against each other on the same day — no time-series leakage.
    """
    mean = feature.mean(axis=1)
    std = feature.std(axis=1)
    return feature.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
