# quant-lab

A research environment for building, backtesting, and (eventually) live-running a
systematic **price-move** trading bot. Equities / crypto / futures — asset class TBD
(see `docs/00-PLAN.md`, "Decision forks").

> **This is a research lab, not a money printer.** The entire premise of this repo is
> to find out *empirically* whether a small, honest edge survives fees, slippage, and
> live conditions — and to fail cheaply (in backtests and paper trading) before risking
> a dollar. If at any phase the numbers don't hold up, the correct outcome is to *not
> deploy capital*. That's a successful project, not a failed one.

## What's here

| Path | What it is |
|------|------------|
| `docs/00-PLAN.md` | The detailed plan: reality check, strategy taxonomy, architecture, phased roadmap, costs, pitfalls, legal. **Start here.** |
| `docs/01-DECISIONS.md` | Append-only log of locked choices (asset class, broker, strategy, capital). |
| `src/quantlab/` | The harness. `data_pipeline/` (Alpaca loader + point-in-time features), `strategies/` (interface + momentum/reversal v1 + baselines), `backtest/` (event-driven engine, cost model, metrics), `cli.py`. |
| `tests/` | Engine-validation (buy-and-hold reproduces the underlying) + no-lookahead guards. Run offline on synthetic data. |

## Status

**Phase 1 complete — backtester built and validated.** Decision forks locked (US
ETFs, daily-swing, Alpaca, paper-only). The engine is no-lookahead by construction,
costs are pessimistic by default, and the cornerstone validation test passes. No
capital at risk.

## Quickstart

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # 6 tests, offline, no keys needed

cp .env.example .env                     # then add free Alpaca keys
python -m quantlab.cli --start 2018-01-01   # real backtest vs SPY + baselines
```

The only thing standing between you and a real backtest is **Alpaca API keys** in
`.env` (free paper-account keys work for the data API). Everything else runs today.

## What's next (Phase 2 → 3)

- **Phase 2 — research:** walk-forward split, cost-sensitivity sweep, parameter-
  robustness check. Decision gate: does `MomentumReversal` beat SPY on Sharpe *net of
  costs, out of sample*, and beat the AlwaysLong / RandomEntry baselines? If not, kill or
  iterate a *bounded* number of times.
- **Phase 3 — paper:** wire the same Strategy into a live runner (`live/`) against an
  Alpaca paper account; add the risk layer (caps, kill switch), reconciliation, and
  signal-only alerting. Compare paper results to backtest expectation.
