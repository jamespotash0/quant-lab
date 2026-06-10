"""Per-trade rationale: explain *why* each name is in the book, and log it.

Every rebalance the system decides a set of weights; this module re-derives the
point-in-time signals behind each holding and turns them into a human-readable reason,
so every trade has an auditable explanation rather than appearing from a black box.

The explanation reports the observable drivers a momentum/vol strategy acts on:
  * cross-sectional momentum z-score (the primary "why this name" signal),
  * trailing realized volatility (the position-sizing driver),
  * proximity to the 52-week high (trend confirmation),
  * and the market-wide volatility rank (the risk-on/off backdrop).

It is an honest explanation of the signals driving the book; it is not a claim that the
strategy computed these exact numbers internally. Rationale rows are appended to
``data/trade_log.jsonl`` and surfaced in the dashboard.
"""

from __future__ import annotations

import json
from typing import cast

import pandas as pd

from .config import DATA_DIR
from .data_pipeline.features import cross_sectional_zscore, momentum, realized_vol
from .research.vol_rank import vol_bucket, vol_rank

TRADE_LOG_PATH = DATA_DIR / "trade_log.jsonl"


def _reason(weight: float, mom_z: float, vol: float, near_high: float, bucket: str) -> str:
    bits: list[str] = []
    if mom_z >= 1.0:
        bits.append(f"strong relative momentum (z={mom_z:+.1f})")
    elif mom_z >= 0.0:
        bits.append(f"positive momentum (z={mom_z:+.1f})")
    else:
        bits.append(f"weak momentum (z={mom_z:+.1f})")
    if near_high >= 0.97:
        bits.append("at/near its 52-week high")
    elif near_high >= 0.90:
        bits.append(f"{near_high:.0%} of its 52-week high")
    bits.append(f"{vol:.0%} realized vol → inverse-vol weight {weight:.1%}")
    bits.append(f"market vol-rank {bucket}")
    return "; ".join(bits)


def explain_book(
    weights: dict[str, float],
    close: pd.DataFrame,
    benchmark: str = "SPY",
) -> list[dict]:
    """Return one rationale row per held name (weight > 0), with the signals and a reason."""
    if not weights:
        return []
    mom_z = cross_sectional_zscore(momentum(close, 126, 21)).iloc[-1]
    vol = realized_vol(close, 63).iloc[-1]
    hi = close.iloc[-252:].max()
    last = close.iloc[-1]

    rank = vol_rank(close[benchmark].dropna()) if benchmark in close.columns else 0.5
    bucket = vol_bucket(rank)

    rows: list[dict] = []
    for sym, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if w <= 0:
            continue
        z = float(cast(float, mom_z.get(sym, float("nan"))))
        v = float(cast(float, vol.get(sym, float("nan"))))
        nh = float(cast(float, last.get(sym, float("nan"))) / cast(float, hi.get(sym, float("nan"))))
        rows.append({
            "symbol": sym,
            "weight": round(w, 4),
            "momentum_z": None if z != z else round(z, 2),       # NaN-safe
            "realized_vol": None if v != v else round(v, 4),
            "near_52w_high": None if nh != nh else round(nh, 3),
            "market_vol_rank": round(rank, 3),
            "market_vol_bucket": bucket,
            "reason": _reason(w, 0.0 if z != z else z, 0.0 if v != v else v,
                              0.0 if nh != nh else nh, bucket),
        })
    return rows


def log_rationale(rows: list[dict], as_of: str, strategy: str,
                  path=TRADE_LOG_PATH) -> None:
    """Append a timestamped rationale record (one line per rebalance) to the trade log."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"as_of": as_of, "strategy": strategy, "holdings": rows}
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
