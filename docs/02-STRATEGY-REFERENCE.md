# Strategy reference

Per-strategy documentation for every `Strategy` subclass in `src/quantlab/strategies/`.
For the head-to-head sweep results and the recommendation rationale, see
[`01-STRATEGY-SWEEP.md`](01-STRATEGY-SWEEP.md). This file is the *what each one does and
why* reference.

## How a strategy works

Every strategy subclasses `Strategy` ([`strategies/base.py`](../src/quantlab/strategies/base.py))
and implements one method:

```python
def target_weights(self, history: History) -> dict[str, float]:
    # return desired weights {symbol: fraction} as of the decision date
```

- `history` is **point-in-time**: it exposes closes/opens only up to the decision date —
  no lookahead is possible by construction.
- `warmup` (property) is the minimum number of daily bars required before the strategy is
  allowed to trade.
- Weights are *target* fractions of equity. The engine handles the fills, costs, and
  accounting; strategies never see cash or P&L directly.
- All strategies here are **long-only**, fraction-of-equity (no shorting, no leverage up),
  with persistent state (`_held` weights, `_age` counter) carried between decisions so they
  can rebalance on a slow cadence rather than every day.

> ⚠️ State is per-strategy-instance. The runner deepcopies strategies per run to avoid
> state leaking between backtests — the canary for a leak is a **net Sharpe > gross Sharpe**.

---

## Recommended / production

### SwingMomentumV3 ⭐
[`research/recommended.py`](../src/quantlab/strategies/research/recommended.py) · *Ensemble*

The current best and recommended strategy. An `EnsembleStrategy` blending
**50% DualMomentum + 25% VolatilityManaged + 25% SwingPivotBreakout**. Adds a swing-structure
trend sleeve (low-correlation with trailing-return momentum) on top of V2 — three uncorrelated,
individually-validated edges stacked.

- **Params:** `dual_weight=0.50`, `vol_weight=0.25`, `swing_weight=0.25`
- **Net Sharpe 1.19** · CAGR 14.3% · MaxDD −12.6% · max position ~27% (2017→present, 5 bps/side)
- **Validation:** passes 5/5; OOS Sharpe (1.33) > IS Sharpe (0.88). The swing sleeve
  *underperforms in-sample but adds the most out-of-sample* — the signature of a real
  diversifier, not an overfit.
- See: [DualMomentum](#dualmomentum), [VolatilityManaged](#volatilitymanaged),
  [SwingPivotBreakout](#swingpivotbreakout), [EnsembleStrategy](#ensemblestrategy).

### SwingMomentumV2 — fallback
[`research/recommended.py`](../src/quantlab/strategies/research/recommended.py) · *Ensemble*

The prior winner: **70% DualMomentum + 30% VolatilityManaged**. Simpler (two sleeves), fully
valid, and a good fallback if the swing sleeve's in-sample weakness is a concern.

- **Params:** `dual_weight=0.7`, `vol_weight=0.3`, `target_gross=1.0`
- **Net Sharpe 1.13** · CAGR 15.8% · MaxDD −15.6% · OOS Sharpe 1.30 (> IS 0.93). 5/5.

---

## The two components of SwingMomentumV2

### DualMomentum
[`research/dual_momentum.py`](../src/quantlab/strategies/research/dual_momentum.py) · *Antonacci GEM style*

The return engine. Combines **relative** momentum (which winners) with **absolute** momentum
(is the trend even up). Per ~21-day rebalance:

1. **Relative:** rank ETFs by 12-month return (skip last month), pick the top `k`.
2. **Absolute gate:** keep a winner only if its 12m return beats the safe asset's (IEF);
   otherwise that sleeve rotates into IEF — the built-in bear-market exit.
3. **Inverse-vol sizing** across sleeves, normalized to `target_gross`.

- **Params:** `k=4`, `lookback=252`, `skip=21`, `vol_lookback=63`, `rebalance_days=21`,
  `safe_asset="IEF"`, `target_gross=1.0` · warmup 278
- **Standalone:** Net Sharpe 1.10 — the *only* single strategy in the sweep to clear the gate.

### VolatilityManaged
[`research/vol_managed.py`](../src/quantlab/strategies/research/vol_managed.py) · *Moreira & Muir (2017)*

The smoother. De-risks a fixed broad-equity book (SPY/QQQ/IWM/DIA/VTI, equal weight)
inversely to its own realized volatility. Per 5-day rebalance:

1. Compute the equity book's portfolio-level realized vol.
2. `scale = target_vol / realized_vol`, capped at 1.0 (long-only: only ever de-risks).
3. Snap `scale` to a coarse grid (cuts churn), spread evenly across priced names.

- **Params:** `target_vol=0.11`, `vol_lookback=42`, `rebalance_days=5`, `scale_grid=0.05`,
  `max_gross=1.0` · warmup 47
- **Standalone:** Net Sharpe 0.86 — best drawdown control in the sweep.

### EnsembleStrategy
[`research/ensemble.py`](../src/quantlab/strategies/research/ensemble.py) · *meta-strategy*

The blender SwingMomentumV2 is built on. Takes `components: list[(Strategy, mix_weight)]`,
calls each sub-strategy's `target_weights`, scales by normalized mix fraction, sums, and
renormalizes to `target_gross` (capping down to cash, never levering up). Thesis: averaging
weak but uncorrelated signals raises Sharpe above any single component. `warmup` = max of
component warmups.

---

## Baselines (the bar to beat)

These exist to make sure a "real" strategy is actually adding value over dumb defaults.

### BuyAndHold
[`strategies/buy_and_hold.py`](../src/quantlab/strategies/buy_and_hold.py)
Holds 100% of one `symbol` (default SPY). Engine-validation baseline — confirms accounting
reproduces the instrument's own return. `warmup 0`.

### AlwaysLong
[`strategies/baselines.py`](../src/quantlab/strategies/baselines.py)
Equal-weight, fully-invested, daily rebalance across all priced symbols. The "just hold the
market" baseline. `target_gross=1.0`.

### RandomEntry
[`strategies/baselines.py`](../src/quantlab/strategies/baselines.py)
Holds `n=5` random symbols for `hold_days=5`, then resamples (seeded for reproducibility).
The coin-flip baseline — if you can't beat this, you have no edge.

---

## Research strategies

Candidates implemented for the sweep. Most failed the gate (see
[`01-STRATEGY-SWEEP.md`](01-STRATEGY-SWEEP.md)); they're kept as documented negative results
and future building blocks. All are long-only, point-in-time, low-turnover.

### MomentumReversal (v1 — deprecated)
[`strategies/momentum.py`](../src/quantlab/strategies/momentum.py)
The original production strategy, **killed**. Cross-sectional momentum + a 5-day reversal
sleeve with a 200-MA regime filter. Died of ~30%/day turnover from the fast reversal leg
(net Sharpe 0.37). `warmup 278`.

### CrossSectionalMomentum
[`research/xsec_momentum.py`](../src/quantlab/strategies/research/xsec_momentum.py)
Pure cross-sectional momentum with the reversal sleeve removed (the thing that killed v1).
Top-`k` by `z(momentum(126, 21))`, inverse-vol sized, 21-day rebalance. Tests whether slow
momentum alone survives costs — it doesn't (Sharpe 0.59). `warmup 152`.

### TimeSeriesMomentum
[`research/trend_tsmom.py`](../src/quantlab/strategies/research/trend_tsmom.py) · *Moskowitz-Ooi-Pedersen*
Absolute momentum — each asset competes against its own past. Long only if trailing 12m
return (skip 21d) > 0, else cash; inverse-vol sized, capped 0.20. Defensive but too defensive
this regime (Sharpe 0.42). `warmup 340`.

### DefensiveAssetAllocation
[`research/defensive_aa.py`](../src/quantlab/strategies/research/defensive_aa.py) · *Keller & Keuning*
Canary-gated tactical allocation. Counts negative-momentum canaries (EEM, AGG) via 13612W
momentum; that fraction goes to defensive bonds, the rest to top-`k` risky ETFs. Over-hedged
this regime (Sharpe 0.37). `warmup 257`.

### FiftyTwoWeekHigh
[`research/fiftytwo_week_high.py`](../src/quantlab/strategies/research/fiftytwo_week_high.py) · *George & Hwang (2004)*
Anchoring effect: ETFs near their own 52-week high keep drifting up. Filter to names with
`close / 252d-max ≥ 0.90`, rank by nearness, top-`k`, inverse-vol sized. Slow, sticky signal.
`warmup 257`.

### DonchianBreakout
[`research/donchian_breakout.py`](../src/quantlab/strategies/research/donchian_breakout.py) · *Turtle trend-following*
Enter long on a close above the 55-day high; stop out on a close below the 20-day low. Size
the strongest breakouts (furthest above entry) inverse-vol every 5 days, capped 0.40. State
machine runs daily so stops are honored promptly. `warmup 68`.

### SwingPivotBreakout
[`research/swing_breakout.py`](../src/quantlab/strategies/research/swing_breakout.py) · *market structure*
Trades confirmed swing pivots: close above a confirmed swing high → ON, below a confirmed
swing low → OFF. Pivots only count once `pivot_width` bars exist on both sides (no lookahead).
SPY-below-200MA regime filter forces cash in bears. `warmup 205`.

### ShortTermMeanReversion
[`research/mean_reversion.py`](../src/quantlab/strategies/research/mean_reversion.py)
Buys oversold index ETFs expecting a bounce. Oversold score = `-(close - SMA(15)) / SMA(15)`;
hold top-`k`, reselect every 10 days with hysteresis to cap churn. Gross Sharpe 0.95 but costs
eat it (net 0.83). `warmup 23`.

### RangeMeanReversion
[`research/range_reversal.py`](../src/quantlab/strategies/research/range_reversal.py)
Bollinger-band reversion. Enter the most oversold names at z ≤ −1.25, hold until they revert
to z ≥ 0, max 5 names, inverse-vol sized, capped 0.30. Sticky positions, 5-day rebalance.
`warmup 68`.

### RiskParity
[`research/risk_parity.py`](../src/quantlab/strategies/research/risk_parity.py)
Naive risk parity — no return forecast. Weight each asset ∝ 1/realized-vol, 21-day rebalance.
Bonds naturally dominate; smooth but low-return (Sharpe 0.78). `warmup 64`.

### MinimumVariance
[`research/min_variance.py`](../src/quantlab/strategies/research/min_variance.py)
Solves the long-only min-variance QP (`min wᵀΣw s.t. Σw=1, w≥0`) on a Ledoit-Wolf-shrunk
covariance via SLSQP, falling back to equal weight on solver failure. Tiny vol, tiny return
(Sharpe 0.57). `warmup 190`.

### HierarchicalRiskParity
[`research/hrp.py`](../src/quantlab/strategies/research/hrp.py) · *Lopez de Prado (2016)*
Risk allocation without inverting the covariance matrix: correlation → distance →
single-linkage clustering → quasi-diagonalize → recursive bisection sized by cluster variance.
Stable but bond-heavy and underperforms here (Sharpe 0.49). `warmup 253`.

### MLCrossSectionalRanking
[`research/ml_ranking.py`](../src/quantlab/strategies/research/ml_ranking.py)
Gradient-boosted (HistGradientBoostingRegressor) ranking on point-in-time features (momentum
21/63/126/252, vol 21/63, 5-day reversal, distance-from-MA 50/200). Label = forward 21-day
return (no lookahead). Retrains every 21 days on an expanding window, holds top-`k`. Fixed
seed → bit-identical reproducibility. No edge over single factors (Sharpe 0.79). `warmup 303`.

### MLRankingTuned
[`research/ml_ranking_tuned.py`](../src/quantlab/strategies/research/ml_ranking_tuned.py)
Same economics as `MLCrossSectionalRanking`, refactored to expose every boosting
hyperparameter (`learning_rate`, `max_depth`, `max_leaf_nodes`, `min_samples_leaf`,
`l2_regularization`, `max_iter`) as `__init__` kwargs — the handoff surface for grid-search
via `research/sweep.py`. `warmup 303`.

---

## Shared infrastructure

- **Features** ([`data_pipeline/features.py`](../src/quantlab/data_pipeline/features.py)) —
  point-in-time `momentum`, `realized_vol`, `short_term_reversal`, `cross_sectional_zscore`,
  `daily_returns`, `log_returns`. Every strategy builds signals from these.
- **Sweep** ([`research/sweep.py`](../src/quantlab/research/sweep.py)) — grid-search a
  strategy's constructor knobs; ranks by **out-of-sample** Sharpe (never optimizes the OOS
  split).
- **Runner** ([`research/runner.py`](../src/quantlab/research/runner.py)) — single-strategy
  backtest; deepcopies the strategy per run to isolate state.
- **CLI** ([`cli.py`](../src/quantlab/cli.py)) — `python -m quantlab.cli --start 2017-01-01`
  for the headline comparison (SwingMomentumV2 vs DualMomentum vs v1 vs SPY vs baselines).
