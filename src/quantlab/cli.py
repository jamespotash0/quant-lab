"""Command-line entrypoint: pull data, run the strategy against its baselines, print a
comparison table.

    python -m quantlab.cli --start 2018-01-01
    python -m quantlab.cli --start 2018-01-01 --end 2023-12-31 --cash 100000
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from .backtest.costs import DEFAULT_COST, ZERO_COST
from .backtest.engine import run_backtest
from .backtest.metrics import summarize
from .data_pipeline.loaders import load_bars
from .strategies.baselines import AlwaysLong, RandomEntry
from .strategies.buy_and_hold import BuyAndHold
from .strategies.momentum import MomentumReversal
from .strategies.research.dual_momentum import DualMomentum
from .strategies.research.recommended import SwingMomentumV2, SwingMomentumV3
from .universe import BENCHMARK, universe


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the quant-lab backtest harness.")
    p.add_argument("--start", default="2018-01-01", help="start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="end date YYYY-MM-DD (default: today)")
    p.add_argument("--cash", type=float, default=100_000.0)
    p.add_argument("--no-cache", action="store_true", help="ignore the Parquet cache")
    p.add_argument("--zero-cost", action="store_true", help="disable the cost model")
    args = p.parse_args(argv)

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    syms = sorted(set(universe()) | {BENCHMARK})

    print(f"Loading bars for {len(syms)} symbols, {args.start} → {end} ...")
    bars = load_bars(syms, args.start, end, use_cache=not args.no_cache)

    cost = ZERO_COST if args.zero_cost else DEFAULT_COST

    def bt(strategy):
        return run_backtest(bars, strategy, initial_cash=args.cash, cost_model=cost)

    results = {
        "SwingMomentumV3": bt(SwingMomentumV3()),        # current best (DM+Vol+Swing)
        "SwingMomentumV2": bt(SwingMomentumV2()),        # prior winner (DM+Vol)
        "DualMomentum": bt(DualMomentum()),              # the return core
        "MomentumReversal(v1)": bt(MomentumReversal()),  # the killed v1, for contrast
        f"Benchmark({BENCHMARK})": bt(BuyAndHold(BENCHMARK)),
        "AlwaysLong": bt(AlwaysLong()),
        "RandomEntry": bt(RandomEntry()),
    }

    table = summarize(results)
    pretty = table.copy()
    for col in ("total_return", "cagr", "ann_vol", "max_drawdown", "time_under_water",
                "hit_rate", "avg_daily_turnover"):
        pretty[col] = (table[col] * 100).map(lambda x: f"{x:6.2f}%")
    pretty["sharpe"] = table["sharpe"].map(lambda x: f"{x:6.2f}")

    print(f"\nCost model: {'ZERO' if args.zero_cost else 'pessimistic (5 bps/side)'}\n")
    print(pretty.to_string())
    print(
        "\nDecision gate: the headline strategy must beat the benchmark on Sharpe AND beat "
        "AlwaysLong / RandomEntry — net of costs, out of sample. If it doesn't, kill it.\n"
        "See docs/01-STRATEGY-SWEEP.md for the full 10-strategy sweep and caveats."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
