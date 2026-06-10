# The vol-regime trading platform

A full systematic-trading stack: regime detection → volatility-based allocation → hard risk
limits → execution → monitoring. Every layer is independent and individually testable.

```
RegimeEngine (HMM)  research/regime_engine.py   auto model selection, no-lookahead, observability
   │
VolRank             research/vol_rank.py          percentile + soft low/mid/high membership
   │
3 vol strategies    strategies/research/vol_*.py  LowVolBull / MidVolCautious / HighVolDefensive
   │
Orchestrator        strategies/research/orchestrator.py  blends by vol-rank + regime confidence
   │
CircuitBreaker      backtest/risk.py              hard equity kill-switch (model-independent)
   │
Main loop           platform.py                   writes data/system_state.json
   │
Dashboard           dashboard/app.py (Streamlit)  reads the state file and renders it
```

## Run it

```bash
python -m quantlab.platform                                  # detect regime, allocate, write state, print summary
python -m streamlit run src/quantlab/dashboard/app.py        # view the dashboard
python -c "from quantlab.research.walkforward import walk_forward, render; \
  from quantlab.strategies.research.orchestrator import StrategyOrchestrator; \
  print(render(walk_forward(StrategyOrchestrator)))"         # walk-forward validate
```

## The HMM engine (`RegimeEngine`)

- **Automatic model selection**: fits Gaussian HMMs across `state_range` and keeps the one
  minimising **BIC** — the number of regimes is learned, not hard-coded (it currently picks 4).
- **No lookahead**: refits only on past data every `refit_days`; "today's" regime is the
  *filtered* estimate (forward pass over data up to today). `random_state` fixed → reproducible.
- **Stability filter**: a candidate regime must persist `confirm_bars` observations before it's
  accepted, debouncing one-bar flicker.
- **Observability API**: `predict_regime_proba`, `get_transition_matrix`, `get_regime_stability`,
  `detect_regime_change`, `get_regime_flicker_rate`, `is_flickering`, `confidence`/`uncertainty`,
  `regime_metadata`.

## The allocation layer

`VolRank` ranks the benchmark's realized vol into a percentile and soft low/mid/high membership.
The `StrategyOrchestrator` blends the three vol-regime strategies by that membership, and tilts
toward the defensive book when the regime is risk-off or the engine is uncertain/flickering
(uncertainty → caution). Because the defensive strategy runs at low gross and the bull strategy
at full gross, **gross exposure scales down automatically as volatility rises** — the requested
"less capital in turbulent markets" behaviour. The `CircuitBreaker` sits underneath as a hard,
model-independent kill switch.

## Validation — and the honest verdict

Walk-forward (one continuous OOS track partitioned into 6-month windows, post-warmup):

| Strategy | OOS Sharpe (agg) | CAGR | MaxDD | Folds positive |
|---|---|---|---|---|
| SwingMomentumV3 (simple) | **1.19** | 14.3% | −12.6% | **11/12** |
| **StrategyOrchestrator** (this platform) | **0.44** | 3.4% | −12.6% | 7/12 |

**The orchestrator underperforms the simple V3 ensemble badly on risk-adjusted return.** This
is the same result every risk-overlay experiment in this repo has produced (see
`docs/03-RISK-ARCHITECTURE.md`): regime-driven and volatility-driven de-risking reduces returns
more than it reduces risk, because regimes are detected with lag and the 2017–2026 sample had
only fast-recovering crashes. The machinery matches V3's drawdown (−12.6%) but at a quarter of
the return.

So treat this platform for what it is: **production-grade infrastructure** — observable,
validated, modular, with a real kill switch and a live dashboard — **not a source of alpha**.
Its worth is (1) operational rigor, (2) a capital-preservation posture for mandates that value
it, and (3) readiness to drop in a genuinely predictive signal (e.g. the single-stock +
fundamentals universe) the moment we have one. On the current ETF signals, **simple V3 still
wins on return**, and a cheap index fund still beats both.
