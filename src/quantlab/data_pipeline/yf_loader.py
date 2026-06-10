"""Long-history price loader via yfinance — the RESEARCH data source.

Alpaca's free IEX feed only reaches back to ~2020 for the full ETF universe, which is far
too short and single-regime to validate a daily strategy. yfinance provides split- and
dividend-adjusted daily bars back to each ETF's inception (SPY to 1993, sector SPDRs to
1998, etc.), so we can backtest across real stress regimes — 2008, 2015-16, 2018-Q4, 2020,
2022 — instead of one post-COVID bull.

Division of labour: yfinance is the *research/backtest* source (deep history); Alpaca stays
the *execution broker* (live account, order routing). Same OHLCV interface as
``loaders.load_bars`` so the engine, features, and strategies are unchanged.

Cache: data/bars_yf/<SYMBOL>.parquet, one file per symbol.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import cast

import pandas as pd

from ..config import DATA_DIR

YF_CACHE_DIR = DATA_DIR / "bars_yf"
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str) -> str:
    YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return str(YF_CACHE_DIR / f"{symbol}.parquet")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a single-symbol yfinance frame to the canonical OHLCV shape: lowercase
    columns, tz-aware (UTC) DatetimeIndex named 'date'."""
    out = df.rename(columns={c: str(c).lower() for c in df.columns})
    out = out[[c for c in BAR_COLUMNS if c in out.columns]].copy()
    idx = pd.DatetimeIndex(out.index)
    out.index = (idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")).normalize()
    out.index.name = "date"
    return out.dropna(how="all").sort_index()


def _download(symbols: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                      group_by="ticker", progress=False, threads=True)
    out: dict[str, pd.DataFrame] = {}
    if raw is None or len(raw) == 0:
        return out
    multi = isinstance(raw.columns, pd.MultiIndex)
    for sym in symbols:
        try:
            sub = raw[sym] if multi else raw
        except KeyError:
            continue
        norm = _normalize(cast(pd.DataFrame, sub))
        if not norm.empty:
            out[sym] = norm
    return out


def load_bars(
    symbols: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",  # accepted for interface parity; yfinance is daily
    feed: str | None = None,  # ignored; parity with the Alpaca signature
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load adjusted daily bars for ``symbols`` between ``start`` and ``end`` from yfinance,
    caching per symbol to Parquet. Returns {symbol: DataFrame[OHLCV]} with a tz-aware (UTC)
    DatetimeIndex named 'date'. Symbols with no data over the window are simply omitted."""
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []
    for sym in symbols:
        path = _cache_path(sym)
        if use_cache and os.path.exists(path):
            cached = pd.read_parquet(path)
            if not cached.empty and cached.index.min() <= start_ts and cached.index.max() >= end_ts:
                result[sym] = cast(pd.DataFrame, cached.loc[start_ts:end_ts])
                continue
        to_fetch.append(sym)

    if to_fetch:
        fetched = _download(to_fetch, start, end)
        for sym, df in fetched.items():
            df.to_parquet(_cache_path(sym))
            result[sym] = cast(pd.DataFrame, df.loc[start_ts:end_ts])

    missing = [s for s in symbols if s not in result]
    if missing:
        # Don't hard-fail the whole universe over a few unavailable names — just note them.
        import warnings

        warnings.warn(f"yfinance returned no data for: {', '.join(missing)}")
    return result
