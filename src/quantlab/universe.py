"""The tradable universe: a curated set of liquid ETFs.

Why ETFs (not individual stocks) to start: free/cheap data has almost no
survivorship bias for major ETFs, the names are deeply liquid (tight spreads, so the
cost model isn't fighting reality), and a cross-sectional momentum/reversal strategy
has clean, diversified building blocks (equity sectors, regions, bonds, commodities).

SPY is included and also used as the benchmark and as the engine-validation instrument.
"""

from __future__ import annotations

# ~30 liquid ETFs spanning broad equity, sectors, regions, bonds, and commodities.
LIQUID_ETFS: list[str] = [
    # Broad US equity
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # US sectors (SPDR)
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    # International / regional
    "EFA", "EEM", "VEA", "VWO", "EWJ",
    # Bonds / rates
    "TLT", "IEF", "LQD", "HYG", "AGG",
    # Commodities / real assets
    "GLD", "SLV", "USO", "DBC",
    # Volatility-adjacent / defensive
    "VNQ",
]

# The benchmark every strategy must beat on a risk-adjusted, net-of-cost basis.
BENCHMARK = "SPY"


def _custom_universe() -> list[str] | None:
    """An optional user override of the tradable universe, so you can trade individual
    stocks (AAPL, MSFT, …) or any liquid US equity instead of the default ETFs:

      * the ``QUANTLAB_UNIVERSE`` env var, comma-separated (e.g. "AAPL,MSFT,NVDA"), or
      * a ``data/universe.txt`` file, one ticker per line (# comments allowed).

    Any valid Alpaca-tradable US equity symbol works — the data loader and engine are
    symbol-agnostic. Note: the strategies were *designed and validated* on the ETF set;
    single stocks add idiosyncratic risk and the strategies use no fundamentals, so
    re-validate (scorecard / walk-forward) before trusting a stock universe.
    """
    import os

    env = os.getenv("QUANTLAB_UNIVERSE")
    syms: list[str] = []
    if env:
        syms = [s.strip().upper() for s in env.split(",") if s.strip()]
    else:
        from .config import DATA_DIR

        path = DATA_DIR / "universe.txt"
        if path.exists():
            syms = [
                line.strip().upper()
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
    return syms or None


def universe() -> list[str]:
    """Return the active tradable universe (deduplicated, order-stable). Honors a custom
    override (see :func:`_custom_universe`); otherwise the default liquid-ETF set."""
    source = _custom_universe() or LIQUID_ETFS
    seen: set[str] = set()
    out: list[str] = []
    for sym in source:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out
