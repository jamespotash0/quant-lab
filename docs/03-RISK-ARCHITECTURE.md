# Layered risk architecture: brain → allocator → safety net → broker

A four-layer system that separates *what to hold* (alpha), *how much to hold* (regime
allocation), *hard limits* (circuit breakers), and *execution* (broker). Each layer is
independent and individually testable.

```
BRAIN        quantlab.research.regime.RegimeBrain        HMM → crash/bear/neutral/bull/euphoria
ALLOCATOR    strategies.research.regime_allocator        scales base-strategy exposure by regime
SAFETY NET   backtest.risk.CircuitBreaker (+ live.py)     flattens to cash on a drawdown breach
BROKER       quantlab.live                                executes the final book on Alpaca
```

## The layers

**Brain (HMM).** A 5-state Gaussian HMM fit on SPY's trailing return and realized vol. States
are labelled by their return signature (crash…euphoria). **No lookahead:** refit only on past
data every ~63 days, and "today's" regime is the *filtered* estimate (forward pass over data
up to today only). `random_state` fixed → reproducible.

**Allocator.** Wraps any base strategy and scales its gross exposure by the regime's budget
(`DEFAULT_EXPOSURE`: crash 0%, bear 40%, neutral 70%, bull 100%, euphoria 85%). The base picks
*what*; the regime decides *how much*. A soft, continuous risk dial.

**Safety net (circuit breaker).** A hard, model-blind kill switch on realized equity. Trips to
cash when peak-to-current drawdown exceeds 15% (re-enter within 7% of peak, with hysteresis),
or on a single-day loss worse than 5%. In backtests it's a `risk_overlay` on `run_backtest`;
live, `quantlab.live` enforces it against the real account equity with a persisted high-water
mark. Independent of the brain by design — when the model is wrong in a way it can't see, the
limit still fires.

**Broker.** `quantlab.live` turns the final target book into Alpaca paper orders.

## What the evidence says (read this before turning it on)

Backtested 2017→present, net of costs. **Every configuration that added the brain and/or the
safety net REDUCED risk-adjusted return.** It is honest to state this plainly:

| Configuration | Sharpe | CAGR | Vol | MaxDD |
|---|---|---|---|---|
| V3 (no brain) | **1.19** | 14.3% | 11.9% | −12.6% |
| V3 + Brain (default map) | 0.83 | 6.9% | 8.5% | −9.6% |
| V3 + Brain (crash-only map) | 0.97 | 9.6% | 9.9% | −11.2% |
| AlwaysLong (raw) | 1.06 | 16.8% | 15.7% | −18.4% |
| AlwaysLong + Brain | 0.80 | 8.2% | 10.5% | −11.6% |
| AlwaysLong + **SafetyNet only** | 0.48 | 5.6% | 12.8% | −17.0% |

The layers **do** reduce volatility and drawdown — they achieve the stated goal of "less
capital in turbulent markets." But they cut **return** more than they cut **risk**, so Sharpe
falls. The circuit breaker alone is the worst (Sharpe 1.06 → 0.48): it sold the COVID-crash
bottom and missed the V-shaped recovery.

**Why:**
1. **Whipsaw / lag.** Regimes are detected *after* the move. The system de-risks after the
   drop and re-enters after the bounce — capturing the worst of both.
2. **Sample bias.** 2017–2026 had only *sharp, fast-recovering* crashes (COVID, 2022). De-risking
   pays off in a *slow, grinding* bear (2000–02, 2008) — which this sample doesn't contain.
3. **Redundancy.** V3 already self-de-risks (DualMomentum rotates to bonds; VolatilityManaged
   scales to cash). A third conservative layer over-hedges.

## How to use it, then

- **If your objective is risk-adjusted return:** use **V3 alone**. The brain hurts it here.
- **If your objective is capital preservation** (you cannot stomach a >12% drawdown): V3 + the
  crash-only brain gives a −9.6% to −11% maxDD and ~8–10% vol — a materially calmer ride — at a
  real cost in return. That's a legitimate trade for some mandates; just know you're *paying* for
  it, not getting it free.
- **Keep the circuit breaker live regardless** — but understand what it is: **tail insurance.**
  It costs you in whipsaws (as the table shows) and is expected to, in exchange for protection
  against a genuine prolonged collapse this backtest never sampled. Insurance has a premium.

The architecture is correct and worth having. The honest finding is that, on this data, more
risk machinery did not mean better returns — only smoother ones.
