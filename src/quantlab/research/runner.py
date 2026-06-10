"""Uniform single-strategy evaluation for research.

Backtest one Strategy over the standard universe and window and return a metrics dict,
computed both NET of the default (pessimistic) cost model and GROSS (zero-cost) so we
can see how much of any edge survives friction.

CLI (used by the research workflow):
    python -m quantlab.research.runner quantlab.strategies.research.trend:TimeSeriesMomentum

The argument is ``module:attr`` where ``attr`` is either a Strategy subclass (called with
no args) or a ready-made Strategy instance. Prints the metrics as JSON.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from dataclasses import asdict

from ..backtest.costs import DEFAULT_COST, ZERO_COST
from ..backtest.engine import run_backtest
from ..backtest.metrics import compute_metrics
from ..data_pipeline.loaders import load_bars
from ..strategies.base import Strategy
from ..universe import BENCHMARK, universe

#: Standard evaluation window. Starts early enough that the 200-day warmup is spent on
#: pre-2018 data, so the reported stats are an out-of-warmup, multi-regime sample.
START = "2017-01-01"


def load_panel(start: str = START, end: str | None = None):
    """Load the standard universe (+ benchmark) once; shared parquet cache makes repeat
    calls cheap. Returns the {symbol: OHLCV} dict the engine consumes."""
    syms = sorted(set(universe()) | {BENCHMARK})
    return load_bars(syms, start, end, use_cache=True)


def evaluate(strategy: Strategy, start: str = START, end: str | None = None) -> dict:
    """Backtest ``strategy`` net and gross of costs; return {"net": {...}, "gross": {...}}
    where each value is the Metrics dataclass as a dict."""
    bars = load_panel(start, end)
    out: dict[str, dict] = {}
    for label, cost in (("net", DEFAULT_COST), ("gross", ZERO_COST)):
        # Fresh copy per run: strategies may carry mutable state (rebalance counters,
        # cached models), and reusing one instance would leak state across runs.
        res = run_backtest(bars, copy.deepcopy(strategy), cost_model=cost)
        out[label] = asdict(compute_metrics(res))
    return out


def _load_strategy(path: str) -> Strategy:
    mod_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attr', got {path!r}")
    obj = getattr(importlib.import_module(mod_name), attr)
    return obj() if isinstance(obj, type) else obj


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: python -m quantlab.research.runner module:attr", file=sys.stderr)
        return 2
    strat = _load_strategy(argv[0])
    result = evaluate(strat)
    result["strategy"] = getattr(strat, "name", type(strat).__name__)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
