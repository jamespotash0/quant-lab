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


def universe() -> list[str]:
    """Return the active tradable universe (deduplicated, order-stable)."""
    seen: set[str] = set()
    out: list[str] = []
    for sym in LIQUID_ETFS:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out
