# Decisions Log (append-only)

Records the locked choices for the project. Nothing here yet — populated once the
decision forks in `00-PLAN.md §1` are chosen.

| Date | Decision | Choice | Rationale |
|------|----------|--------|-----------|
| 2026-06-10 | Asset class | **US equities / ETFs** | Free data + Alpaca paper, deep/clean data, well-understood. De-risks the build so we focus on the edge. |
| 2026-06-10 | Frequency / holding period | **Daily-swing (hold 1–10 days)** | Enough trades for real statistics; latency/colocation don't decide the winner; costs manageable. |
| 2026-06-10 | Capital willing to fully lose | **$0 for now — paper only** | Build + backtest + paper trade. Funding decision deferred to Phase 4, only if numbers hold up. |
| 2026-06-10 | Automation level (start) | **Signal-only / paper** | Separate "is the signal good?" from "is the execution code safe?". |
| 2026-06-10 | First strategy hypothesis | **Cross-sectional momentum + short-term reversal**, vol-targeted sizing, 200d regime filter, long-only top-5 of a ~30 ETF universe | Transparent, well-documented anomalies; implemented as `strategies/momentum.py:MomentumReversal`. A hypothesis to try to kill, not a belief. |
| 2026-06-10 | Universe | **~30 liquid ETFs** | Negligible survivorship bias on free data, deep liquidity (cost model isn't fighting reality), clean cross-sectional building blocks. See `src/quantlab/universe.py`. |
| 2026-06-10 | Backtest engine | **Hand-rolled event loop** | Auditable; we can read all of it and trust the no-lookahead guarantee. Validated: buy-and-hold reproduces the underlying to floating-point tolerance. |
| 2026-06-10 | Data source (Phase 1) | **Alpaca data API** (adjusted daily bars, Parquet cache) | Same vendor the live/paper path will use, so the data path matches end to end. yfinance kept as a possible fallback. |

## Notes
- **PDT rule — repealed (per user, 2026-06-10):** The old count-based Pattern Day Trader
  designation and its **$25,000 minimum equity** requirement have been **eliminated**.
  FINRA replaced it via **amendments to Rule 4210** with a **real-time intraday margin
  framework** that monitors actual market exposure instead of capping the number of day
  trades.
  - No limit on number of day trades; no $25k minimum balance.
  - Firms monitor margin accounts in real time; falling short of equity vs. open exposure
    creates an **"intraday margin deficit"**, and repeatedly failing to cure margin calls
    can still trigger trading restrictions.
  - **Effective June 2026**, with an **18-month phase-in** — so individual brokerages may
    roll it out on different timelines. Confirm with the specific broker (Alpaca) at
    Phase 4 before relying on it.
  - *Source: provided by user; not independently verified against FINRA text by Claude
    (post knowledge cutoff). Verify the primary FINRA/SEC notice before live trading.*
  - **Impact on this project:** removes the old reason to avoid intraday for small
    accounts. We still start daily-swing + paper for the statistics/simplicity reasons,
    not because of PDT — but an intraday lane is no longer gated by a $25k floor if we
    later want it.
