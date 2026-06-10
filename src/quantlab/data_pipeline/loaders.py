"""Load daily bars from Alpaca and cache them to Parquet.

Adjustment policy: we request fully-adjusted bars (splits + dividends). Unadjusted or
split-only data silently corrupts returns and is one of the classic ways a backtest
lies. See docs/00-PLAN.md section 4.

Cache layout: data/bars/<SYMBOL>_<TIMEFRAME>_<FEED>.parquet, one file per symbol.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from ..config import BARS_CACHE_DIR, alpaca_config

# Canonical OHLCV columns we expose to the rest of the system.
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(symbol: str, timeframe: str, feed: str) -> Path:
    BARS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return BARS_CACHE_DIR / f"{symbol}_{timeframe}_{feed}.parquet"


def _fetch_from_alpaca(
    symbols: list[str], start: str, end: str, feed: str
) -> dict[str, pd.DataFrame]:
    """Fetch adjusted daily bars from Alpaca. Returns {symbol: OHLCV DataFrame}."""
    # Imported lazily so the package imports (and tests using cached/synthetic data
    # run) even if alpaca-py isn't installed.
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.models import BarSet
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    cfg = alpaca_config()
    if not cfg.is_configured:
        raise RuntimeError(
            "Alpaca API keys not configured. Copy .env.example to .env and fill in "
            "ALPACA_API_KEY / ALPACA_SECRET_KEY (free paper-account keys work)."
        )

    client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        # TimeFrame.Day is a classproperty; alpaca's stubs type it as classproperty
        # rather than TimeFrame, so narrow it.
        timeframe=cast(TimeFrame, TimeFrame.Day),
        start=pd.Timestamp(start, tz="UTC").to_pydatetime(),
        end=pd.Timestamp(end, tz="UTC").to_pydatetime(),
        adjustment=Adjustment.ALL,
        feed=DataFeed(feed),
    )
    # get_stock_bars returns BarSet | dict; the symbol_or_symbols form yields a BarSet.
    bars = cast(BarSet, client.get_stock_bars(req))
    df = bars.df  # MultiIndex (symbol, timestamp)

    out: dict[str, pd.DataFrame] = {}
    if df.empty:
        return out
    for symbol in df.index.get_level_values(0).unique():
        sym_df = cast(pd.DataFrame, df.xs(symbol, level=0)).copy()
        sym_df.index = pd.DatetimeIndex(sym_df.index).tz_convert("UTC").normalize()
        sym_df.index.name = "date"
        out[symbol] = sym_df[BAR_COLUMNS].sort_index()
    return out


def load_bars(
    symbols: list[str],
    start: str,
    end: str | None = None,
    timeframe: str = "1Day",
    feed: str | None = None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load adjusted daily bars for ``symbols`` between ``start`` and ``end``.

    Reads from the Parquet cache when it fully covers the requested window; otherwise
    fetches from Alpaca and writes the cache. Returns {symbol: DataFrame[OHLCV]} with a
    tz-aware (UTC) DatetimeIndex named "date".
    """
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    feed = feed or alpaca_config().data_feed

    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()

    result: dict[str, pd.DataFrame] = {}
    to_fetch: list[str] = []

    for sym in symbols:
        path = _cache_path(sym, timeframe, feed)
        if use_cache and path.exists():
            cached = pd.read_parquet(path)
            covered = (
                not cached.empty
                and cached.index.min() <= start_ts
                and cached.index.max() >= end_ts
            )
            if covered:
                result[sym] = cached.loc[start_ts:end_ts]
                continue
        to_fetch.append(sym)

    if to_fetch:
        fetched = _fetch_from_alpaca(to_fetch, start, end, feed)
        for sym in to_fetch:
            df = fetched.get(sym)
            if df is None or df.empty:
                continue
            _cache_path(sym, timeframe, feed)  # ensure dir
            df.to_parquet(_cache_path(sym, timeframe, feed))
            result[sym] = df.loc[start_ts:end_ts]

    missing = [s for s in symbols if s not in result]
    if missing:
        raise RuntimeError(f"No bars returned for: {', '.join(missing)}")
    return result


def to_panel(bars: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Pivot {symbol: OHLCV} into a wide DataFrame (index=date, columns=symbol) for one
    field. Dates are the union across symbols; missing values are left as NaN so the
    feature layer can decide how to handle them (no silent forward-fill here).
    """
    series = {sym: df[field].rename(sym) for sym, df in bars.items()}
    panel = pd.concat(series.values(), axis=1)
    panel.columns = list(series.keys())
    return panel.sort_index()
