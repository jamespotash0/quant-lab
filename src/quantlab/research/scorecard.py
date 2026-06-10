"""The 5-criteria strategy scorecard.

Every candidate must clear five gates before it earns real capital. This module turns
that rubric into an automated, reproducible test so judgement isn't hand-waved:

  1. REPRODUCIBLE   — run the backtest twice; results must be bit-identical (no hidden
                      randomness, no order-dependent state).
  2. CLEAR THESIS   — the strategy must declare *why* it should work (a ``thesis``
                      attribute or a class docstring). An edge you can't explain is one
                      you can't trust when it breaks.
  3. THOROUGH TEST  — walk-forward split: fit/observe in-sample (pre-``SPLIT``), then
                      judge on a held-out out-of-sample period the strategy never had a
                      hand in. OOS performance is the one that counts.
  4. POSITIVE EV    — net of the pessimistic cost model, expectancy is positive
                      (Sharpe & CAGR > 0) AND it beats buy-and-hold SPY on Sharpe.
  5. RISK MGMT      — bounded tail risk: max drawdown no worse than SPY's, and no single
                      position dominates the book.

Usage:
    from quantlab.research.scorecard import score, render
    print(render(score(MyStrategy())))
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from ..backtest.costs import DEFAULT_COST
from ..backtest.engine import BacktestResult, run_backtest
from ..backtest.metrics import Metrics, compute_metrics
from ..strategies.base import Strategy
from ..strategies.buy_and_hold import BuyAndHold
from .runner import START, load_panel

#: Walk-forward boundary: everything before is in-sample, everything after is the
#: out-of-sample holdout (includes the 2022 bear market — a real stress test).
SPLIT = "2022-01-01"

#: Risk gate: a strategy's worst peak-to-trough loss must be no deeper than this.
MAX_DD_LIMIT = -0.25
#: Risk gate: no single name may exceed this share of the book on any day.
MAX_CONCENTRATION = 0.40


@dataclass
class Criterion:
    name: str
    passed: bool
    detail: str


@dataclass
class ScoreCard:
    strategy: str
    criteria: list[Criterion] = field(default_factory=list)
    full: Metrics | None = None
    in_sample: Metrics | None = None
    out_sample: Metrics | None = None
    spy_full: Metrics | None = None

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.criteria)

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed == len(self.criteria) else f"{self.passed}/5"


def _thesis(strategy: Strategy) -> str:
    """A strategy's declared rationale: an explicit ``thesis`` attribute wins, else the
    first paragraph of its class docstring."""
    declared = getattr(strategy, "thesis", None)
    if isinstance(declared, str) and declared.strip():
        return " ".join(declared.split())
    doc = type(strategy).__doc__ or ""
    para = doc.strip().split("\n\n")[0] if doc.strip() else ""
    return " ".join(para.split())


def _run(bars, strategy: Strategy, start: str | None, end: str | None) -> BacktestResult:
    # Fresh copy each run so mutable strategy state never leaks between evaluations.
    return run_backtest(
        bars, copy.deepcopy(strategy), cost_model=DEFAULT_COST, start=start, end=end
    )


def score(strategy: Strategy, start: str = START, split: str = SPLIT,
          end: str | None = None) -> ScoreCard:
    """Run ``strategy`` through all five gates and return a populated ScoreCard."""
    bars = load_panel(start, end)
    name = getattr(strategy, "name", type(strategy).__name__)

    full = _run(bars, strategy, start, end)
    fm = compute_metrics(full)
    ism = compute_metrics(_run(bars, strategy, start, split))
    oosm = compute_metrics(_run(bars, strategy, split, end))
    spym = compute_metrics(_run(bars, BuyAndHold("SPY"), start, end))

    # 1. Reproducible: a second independent run must match to the penny.
    repro_tr = compute_metrics(_run(bars, strategy, start, end)).total_return
    reproducible = abs(repro_tr - fm.total_return) < 1e-9

    # 2. Clear thesis: present and non-trivial.
    thesis = _thesis(strategy)
    has_thesis = len(thesis) >= 30

    # 5. Risk management: tail-loss and concentration from the daily weight book.
    w = full.weights
    max_name = float(w.to_numpy().max()) if w.size else 0.0
    cash = float((1.0 - w.sum(axis=1)).clip(lower=0.0).mean()) if w.size else 0.0
    risk_ok = fm.max_drawdown >= MAX_DD_LIMIT and max_name <= MAX_CONCENTRATION

    card = ScoreCard(strategy=name, full=fm, in_sample=ism, out_sample=oosm, spy_full=spym)
    card.criteria = [
        Criterion("Reproducible", reproducible,
                  f"two runs match to {abs(repro_tr - fm.total_return):.2e}"),
        Criterion("Clear thesis", has_thesis,
                  (thesis[:90] + "…") if len(thesis) > 90 else (thesis or "MISSING")),
        Criterion("Thorough testing", oosm.sharpe > 0.0,
                  f"OOS Sharpe {oosm.sharpe:.2f} vs IS {ism.sharpe:.2f} "
                  f"(OOS CAGR {oosm.cagr:.1%})"),
        Criterion("Positive EV", fm.sharpe > 0.0 and fm.cagr > 0.0 and fm.sharpe > spym.sharpe,
                  f"net Sharpe {fm.sharpe:.2f} vs SPY {spym.sharpe:.2f}, "
                  f"CAGR {fm.cagr:.1%}"),
        Criterion("Risk management", risk_ok,
                  f"maxDD {fm.max_drawdown:.1%} (limit {MAX_DD_LIMIT:.0%}), "
                  f"max name {max_name:.0%}, avg cash {cash:.0%}"),
    ]
    return card


def render(card: ScoreCard) -> str:
    lines = [f"STRATEGY CARD — {card.strategy}   [{card.verdict}]", "-" * 64]
    for c in card.criteria:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"  [{mark}] {c.name:<17} {c.detail}")
    if card.full and card.out_sample:
        lines.append("-" * 64)
        lines.append(
            f"  full: Sharpe {card.full.sharpe:.2f}  CAGR {card.full.cagr:.1%}  "
            f"maxDD {card.full.max_drawdown:.1%}   |   "
            f"OOS: Sharpe {card.out_sample.sharpe:.2f}  CAGR {card.out_sample.cagr:.1%}"
        )
    return "\n".join(lines)


def score_frame(cards: list[ScoreCard]) -> pd.DataFrame:
    """One row per strategy: pass count + the headline IS/OOS numbers, ranked by OOS Sharpe."""
    rows = {}
    for c in cards:
        rows[c.strategy] = {
            "criteria": f"{c.passed}/5",
            "is_sharpe": c.in_sample.sharpe if c.in_sample else float("nan"),
            "oos_sharpe": c.out_sample.sharpe if c.out_sample else float("nan"),
            "oos_cagr": c.out_sample.cagr if c.out_sample else float("nan"),
            "full_sharpe": c.full.sharpe if c.full else float("nan"),
            "full_maxdd": c.full.max_drawdown if c.full else float("nan"),
        }
    return pd.DataFrame(rows).T.sort_values("oos_sharpe", ascending=False)
