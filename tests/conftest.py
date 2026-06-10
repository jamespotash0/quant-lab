"""Shared fixtures: synthetic OHLCV bars so the engine/strategy tests run offline
(no Alpaca keys, no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_bars(symbols, n_days=400, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_days, freq="B", tz="UTC").normalize()
    dates.name = "date"
    bars = {}
    for k, sym in enumerate(symbols):
        # Geometric random walk for the close; open = prior close * small gap.
        drift = 0.0003 + 0.0001 * k
        rets = rng.normal(drift, 0.01, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        open_ = np.empty(n_days)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1 + rng.normal(0, 0.001, n_days - 1))
        df = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.001,
                "low": np.minimum(open_, close) * 0.999,
                "close": close,
                "volume": rng.integers(1_000_000, 5_000_000, n_days).astype(float),
            },
            index=dates,
        )
        bars[sym] = df
    return bars


@pytest.fixture
def bars():
    return _make_bars(["SPY", "QQQ", "TLT", "GLD", "XLK", "XLF"])


@pytest.fixture
def single_bars():
    return _make_bars(["SPY"])
