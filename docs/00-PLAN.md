# Systematic Price-Move Trading Bot — Detailed Plan

**Author:** drafted with Claude (Opus 4.8) · **Date:** 2026-06-10 · **Status:** Phase 0, planning

---

## 0. Read this first — the honest framing

Most retail algo-trading projects lose money. Not because the code is bad, but because
**they never had an edge**, and a backtest told them they did. The single most valuable
thing this project can produce is an *honest answer* to one question:

> Do we have a repeatable, statistically real edge that survives realistic fees,
> slippage, and the difference between a backtest and live execution?

If the answer is no, we don't trade. The whole architecture below is built to surface a
"no" as **early and as cheaply as possible** — in a backtest (free), then in paper
trading (free), long before any real capital is exposed.

Three truths to keep nailed to the wall:

1. **The market already prices in the obvious.** "Stock went up 3 days → it'll go up
   tomorrow" is not an edge; if it were, it'd be arbitraged away. A real edge is usually
   small, specific, and decays over time as others find it.
2. **Costs are the silent killer.** Every round trip pays the bid/ask spread + commission
   + slippage (you don't get the price you saw). A strategy that's +0.05% per trade gross
   can be *negative* net. Many "profitable" backtests die the moment realistic costs are
   modeled.
3. **Backtests lie by default.** Lookahead bias, survivorship bias, overfitting to noise,
   and unrealistic fills make almost every naive backtest look amazing. Discipline in the
   backtester is 80% of the work.

We proceed with curiosity, not conviction. The goal of Phase 1–3 is to *try to kill the
idea*. If it survives our best attempts to kill it, *then* we cautiously fund it.

---

## 1. Decision forks (pick these to make the plan concrete)

The rest of the plan adapts to four choices. My recommended defaults are marked ✅.

### A. Asset class
| Option | Pros | Cons | Fit |
|--------|------|------|-----|
| **US equities / ETFs** ✅ | Deep data, cheap/free APIs, fractional shares, well-understood, paper trading widely available | Market hours only (9:30–16:00 ET), PDT rule under $25k, crowded | **Best starting point** |
| **Crypto** | 24/7, easy API access, no PDT rule, low minimums | Thinner edges now, more manipulation, custody/exchange risk, brutal fees on small caps | Good #2; good for "always-on bot" learning |
| **Futures** | Leverage, tax treatment (1256), deep liquidity in ES/NQ | Leverage = fast ruin, higher capital, more complex | Later |
| **FX** | 24/5, liquid | Retail FX is a graveyard; dealer conflicts | Avoid |

**Recommendation:** Start with **US equities/ETFs via Alpaca** (free paper trading, clean
API, no data vendor needed to begin). It de-risks the *build* so we can focus on the
*edge*. Crypto is a strong fast-follow if you want 24/7 and no PDT constraint.

### B. Holding period / frequency
| Option | What it means | Reality for a solo builder |
|--------|---------------|----------------------------|
| **HFT / intraday scalping** | Seconds–minutes | ❌ You will lose to colocated market makers. Don't. |
| **Intraday swing** | Minutes–hours, flat by close | Viable but fill/slippage-sensitive |
| **Daily / swing** ✅ | Hold 1–10 days, decide on daily bars | **Best risk/reward for solo.** Costs matter less per unit of edge; data is cheap; less infra |
| **Position / trend** | Weeks–months | Viable, very slow to validate (few trades = weak statistics) |

**Recommendation:** **Daily-bar swing trading.** Enough trades to get statistical
significance, but slow enough that latency/colocation don't decide who wins.

### C. Capital
This sets everything: the PDT rule (US pattern-day-trader needs **$25k** to day-trade
freely), position sizing, and whether fixed costs matter. **Tell me the number you're
willing to fully lose.** Plan assumes a starting research budget where *losing 100% would
not hurt you*. If that number is small (say <$2k), we lean crypto or low-frequency equity
swing to dodge PDT.

### D. Automation level
| Option | Description |
|--------|-------------|
| **Signal-only (alerts)** ✅ to start | Bot computes signals, sends you a notification, *you* place the trade. Zero execution risk while validating. |
| **Semi-auto** | Bot proposes orders, you click to confirm |
| **Fully autonomous** | Bot places + manages orders unattended (only after months of proof) |

**Recommendation:** Build toward full auto, but **run signal-only first**. It separates
"is the signal good?" from "is my execution code safe?" — two failures you do not want to
debug simultaneously with real money.

---

## 2. Where a real edge could plausibly come from (strategy taxonomy)

We are *not* trying to predict the future. We're trying to find a **statistical tendency**
that's real, exploitable after costs, and not already arbitraged away. Candidate families
ranked by realism for a solo, daily-frequency builder:

1. **Cross-sectional momentum / mean-reversion** (most promising)
   - Rank a universe (e.g. S&P 500 or liquid ETFs) by a signal, long the top decile /
     short or avoid the bottom. Momentum (12-1 month) and short-term reversal (1–5 day)
     are among the most-documented real anomalies. Edges are small but persistent-ish.
2. **Volatility / regime filters**
   - Not a standalone edge, but *huge* as an overlay: only trade when conditions favor the
     strategy (e.g. trend-follow only in trending regimes, mean-revert only in calm).
3. **Event/calendar effects**
   - Earnings drift (PEAD), index rebalances, month/quarter-end flows. Real but capacity-
     limited and data-hungry.
4. **Statistical arbitrage / pairs**
   - Trade the spread between cointegrated instruments. Classic, but the easy pairs are
     long gone; needs careful stat work and decays fast.
5. **Sentiment / alt-data / ML on features**
   - Tempting and where "AI" gets oversold. ML mostly *overfits* on financial data
     (low signal-to-noise). If used, it's for *combining* a few robust features, not for
     finding magic patterns in raw prices. Treat with maximum skepticism.

**Recommended first strategy to test:** a **simple, transparent cross-sectional
momentum-with-reversal overlay on liquid ETFs**, with a volatility regime filter and hard
risk caps. It's understandable (so we can tell *why* it works or doesn't), well-documented
in literature (so we're not inventing from zero), and low-frequency (so costs and infra
are manageable). We start dead simple and only add complexity that *demonstrably* improves
out-of-sample results.

---

## 3. Architecture

Pluggable, with a hard wall between **research** (backtest) and **live** (execution) that
share the *same strategy code* — so what you test is what you trade.

```
quant-lab/
├── data/            # cached historical bars, universe lists (gitignored, large)
├── strategies/
│   └── base.py      # Strategy interface: signals(features) -> target_weights
│   └── momentum.py  # first concrete strategy
├── backtest/
│   ├── engine.py    # event-driven loop; NO lookahead by construction
│   ├── costs.py     # commission + spread + slippage model (pessimistic)
│   └── metrics.py   # Sharpe, max drawdown, turnover, hit rate, etc.
├── data_pipeline/
│   ├── loaders.py   # pull/cache bars (Alpaca / yfinance / vendor)
│   └── features.py  # returns, vol, momentum ranks — computed point-in-time
├── live/
│   ├── broker.py    # broker adapter (Alpaca first); paper + live same interface
│   ├── runner.py    # the daily loop: load data -> signals -> reconcile -> orders
│   └── risk.py      # position limits, max gross/net, daily loss kill switch
├── notify/          # signal-only mode: push/email/SMS alerts
└── docs/
```

**Key design rules:**
- **Same `Strategy` object** is consumed by both `backtest/engine.py` and `live/runner.py`.
  No "backtest version vs live version" of the logic — that divergence is how people get
  burned.
- **Point-in-time features only.** Every feature is computed using *only data available at
  that moment*. The backtest engine enforces this structurally (it feeds the strategy a
  growing window, never the full series).
- **Risk layer is independent and dominant.** `risk.py` can veto/clamp any order
  regardless of what the strategy wants. Hard caps: max position size, max gross exposure,
  per-day loss kill switch that flattens and halts.
- **Idempotent order reconciliation.** The live runner computes *target* positions and
  diffs against *actual* broker positions, so a crash + restart converges rather than
  double-trading.

---

## 4. Data (the unglamorous, decisive part)

- **To start (equities, daily):** Alpaca's market data API or `yfinance` for free daily
  bars. Good enough to build and validate the harness.
- **Survivorship bias warning:** free data often excludes delisted/bankrupt tickers, which
  inflates backtests (you only test the survivors). For a serious equity backtest you
  eventually want a **point-in-time universe with delisted names** (vendors: Norgate,
  Sharadar/Nasdaq Data Link, Polygon). Budget for this *only after* the harness proves
  out on free data.
- **Crypto:** exchange APIs (Coinbase, Kraken, Binance) give clean historical candles for
  free; less survivorship trouble.
- **Corporate actions:** splits/dividends must be adjusted correctly or your returns are
  garbage. Use adjusted data and verify on a known case.

**Data is the #1 place backtests lie.** Half of Phase 1 is just trusting the data.

---

## 5. Backtesting discipline (the core of the whole project)

Rules the engine enforces or we manually audit:

- **No lookahead.** Decide on bar *t* using data through *t*; execute at *t+1* open (or
  with realistic intraday assumptions). Never use the close you're predicting.
- **Pessimistic costs.** Model commission + half-spread + slippage, and make slippage
  *worse* than you think. If it's only profitable with zero costs, it's not profitable.
- **Out-of-sample / walk-forward.** Develop on one period, validate on a *later, untouched*
  period. Re-fit on a rolling window (walk-forward) to mimic reality. The final test set is
  touched **once**.
- **Beware overfitting.** Every parameter you tune is a chance to fit noise. Prefer few
  parameters, robust across ranges. If performance cliffs when you nudge a parameter, it's
  overfit. Track how many "tries" you've taken (multiple-testing inflates false positives).
- **Honest metrics:** Sharpe (after costs), max drawdown, time-under-water, turnover,
  hit rate, average win/loss, and **capacity** (does it still work at your size?).
- **Benchmark against buy-and-hold.** Beating SPY *risk-adjusted*, net of costs, is the
  bar. Many strategies just secretly take more risk.
- **Sanity baselines:** random-entry and always-long baselines. If your "edge" doesn't
  clearly beat a coin flip with the same exposure, it isn't one.

A strategy graduates from backtest only if it survives walk-forward, realistic costs, and
a parameter-robustness check — and even then we treat it as a *hypothesis*, not a fact.

---

## 6. Risk management (non-negotiable)

Independent of the strategy. These are hard rules in `live/risk.py`:

- **Position sizing:** small fixed fraction or volatility-targeted (size inversely to each
  name's volatility so positions contribute equal risk). Cap any single position (e.g.
  ≤5–10% of capital).
- **Gross/net exposure caps.** Never more than X% deployed; cap leverage at 1x to start
  (no margin until proven).
- **Daily loss kill switch.** If P&L draws down past a threshold in a day, **flatten and
  halt** until manual review. This is the seatbelt.
- **Per-trade stop discipline** (or systematic exit rules) defined *before* entry.
- **Reconciliation + heartbeat.** Bot checks its positions match the broker every cycle;
  alerts and halts on mismatch. Dead-man's switch if the runner stops heart-beating.
- **Start tiny.** First live capital is an amount where 100% loss is irrelevant to you.

---

## 7. Execution & brokers

| Broker | API quality | Paper trading | Notes |
|--------|-------------|---------------|-------|
| **Alpaca** ✅ | Excellent REST/websocket, Python-first | Yes, free, full-featured | Best place to start for US equities/crypto |
| **Interactive Brokers** | Powerful, global, but clunky API | Yes | Graduate here for scale, futures, options, non-US |
| **Tradier** | Good, options-friendly | Yes | Alternative |
| **Crypto exchanges** | Coinbase/Kraken/Binance APIs | Some testnets | For crypto path |

**Recommendation:** **Alpaca paper account** for everything through Phase 3. Same code
path flips to live by swapping a key. Zero excuse to skip paper trading.

---

## 8. Tech stack

- **Language:** Python (the quant ecosystem lives here).
- **Core libs:** `pandas`/`polars`, `numpy`, `vectorbt` or a hand-rolled event engine for
  backtests, `alpaca-py` for broker, `pyarrow` for cached data.
- **Backtest approach:** start with a *simple, auditable, event-driven loop* we control
  (so we trust the no-lookahead guarantee) before reaching for heavyweight frameworks.
- **Scheduling (live):** a daily cron/systemd timer (or cloud function) that runs the
  decision loop after market close / before open. No need for always-on infra at daily
  frequency.
- **Storage:** local Parquet for bars; SQLite/Postgres for trades + positions ledger.
- **Observability:** structured logs, a trades ledger, daily P&L email/push, Sentry-style
  error alerts. You want to *know* when the bot does something weird.
- **Secrets:** broker keys in a local `.env`, never committed. Paper keys ≠ live keys.

---

## 9. Phased roadmap (fail cheap, escalate slowly)

**Phase 0 — Plan & pick lane** *(now)*
- Choose the four decision forks (§1). Log them in `docs/01-DECISIONS.md`.
- Define the *first* concrete strategy hypothesis in one paragraph.

**Phase 1 — Data + backtester** *(the real work)*
- Build data loader + point-in-time feature pipeline on free data.
- Build the event-driven backtest engine with pessimistic costs and a no-lookahead
  guarantee. Validate the engine on a *known* trivial strategy (e.g. buy-and-hold SPY
  should match SPY's actual return — if it doesn't, the engine is wrong).
- Implement metrics + baselines (random, always-long).

**Phase 2 — Strategy research**
- Implement strategy v1 (simple momentum/reversal + vol filter).
- Walk-forward test. Cost-sensitivity test. Parameter-robustness test.
- **Decision gate:** does it beat SPY risk-adjusted, net of pessimistic costs,
  out-of-sample? If no → iterate a *bounded* number of times or kill it. No infinite
  fishing.

**Phase 3 — Paper trading**
- Wire the *same* strategy into the live runner against an Alpaca **paper** account.
- Run **signal-only / paper for weeks**. Compare live paper results to backtest
  expectation. Divergence here = the backtest was optimistic; investigate before risking
  a cent.
- Build risk layer, reconciliation, kill switch, alerting.

**Phase 4 — Tiny live**
- Only if Phase 3 matches expectations. Fund an amount you can fully lose. Run
  semi-auto (you confirm orders) first, then full auto.
- Scale *only* with proof and only slowly.

**Phase 5 — Maintain**
- Monitor for edge decay (strategies die). Periodic re-validation. Kill strategies that
  stop working. Never marry a strategy.

We do **not** advance a phase until the prior phase's gate is met. Most honest projects
stop at Phase 2 or 3 — and that's a *win*, because it cost time, not money.

---

## 10. Costs & realistic expectations

- **Build cost:** mostly your time. Free tools get you through Phase 3.
- **Likely paid later:** point-in-time historical data with delisted names ($ to $$/mo),
  maybe a small cloud box ($5–20/mo).
- **Return expectations:** A *good, real* systematic retail edge is modest — think
  beating the market by a few risk-adjusted percent, with drawdowns, not "double my money."
  Anything promising more is almost certainly overfit or a scam. The realistic upside of a
  successful project is "a modest, diversifying, risk-adjusted edge" — and the realistic
  *most-likely* outcome is "we proved there's no edge worth the risk, cheaply." Both are
  fine outcomes; blowing up an account is not.

---

## 11. Top ways this goes wrong (pin these up)

1. Overfitting a backtest, believing it, funding it, watching it not work live.
2. Ignoring costs/slippage until real money reveals them.
3. Survivorship/lookahead bias inflating results.
4. Skipping paper trading because the backtest looked great.
5. No kill switch / sizing too big → one bad day erases months.
6. Marrying a strategy after its edge has decayed.
7. Using ML to "find patterns" in noise and fitting the noise.
8. Letting curiosity become conviction and conviction become rent money at risk.

---

## 12. Legal / practical

- **US Pattern Day Trader rule:** <$25k accounts are limited to 3 day-trades / 5 days.
  Our daily-swing default mostly sidesteps this, but it constrains intraday plans.
- **Taxes:** active trading generates many taxable events; wash-sale rules apply to
  equities. Keep a clean trade ledger from day one (the bot should log everything).
- **Crypto:** custody/exchange counterparty risk is real; use reputable venues; consider
  jurisdiction.
- **Not investment advice.** This is your capital and your decision. The bot is a tool;
  the risk is yours. Start with money you can fully lose.

---

## 13. Immediate next step

Pick the four decision forks in §1 (asset class, frequency, capital-you-can-lose,
automation level) and I'll:
1. Lock them into `docs/01-DECISIONS.md`.
2. Spec Phase 1 in detail and **scaffold the backtester repo** — data loader + event
   engine + cost model + metrics + a buy-and-hold validation test — so you have something
   you can actually run by the end of the first build session.

My default recommendation if you just want to start: **US equities/ETFs, daily-swing,
signal-only first, Alpaca paper account, simple momentum+reversal strategy.** Say the word
and I'll scaffold it.
