"""Main loop / orchestration: run the full platform once and emit a system snapshot.

On-demand (run it daily, manually or from a scheduler). One pass:

  1. backtest the :class:`StrategyOrchestrator` (regime engine -> vol-rank routing -> blend)
     under the circuit breaker over the full history up to the latest close,
  2. read the resulting current regime / confidence / allocation and the equity curve,
  3. write ``data/system_state.json`` in the schema the Streamlit dashboard consumes, and
  4. print a human summary.

    python -m quantlab.platform            # compute + write state + print summary
    python -m quantlab.dashboard           # (separately) view it: streamlit run .../dashboard/app.py

This does NOT place orders — use ``quantlab.live`` for the broker. It produces the decision
state and the dashboard feed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import pandas as pd

from .backtest.costs import DEFAULT_COST
from .backtest.engine import run_backtest
from .backtest.risk import CircuitBreaker
from .config import DATA_DIR
from .research.regime import Regime
from .research.runner import START, load_panel

STATE_PATH = DATA_DIR / "system_state.json"


def build_state(start: str = START, equity_tail: int = 252) -> dict:
    """Run the orchestrator over history and assemble the dashboard system-state dict."""
    from .strategies.research.orchestrator import StrategyOrchestrator

    bars = load_panel(start)
    orch = StrategyOrchestrator()
    breaker = CircuitBreaker()
    res = run_backtest(bars, orch, cost_model=DEFAULT_COST, risk_overlay=breaker)
    eng = orch.engine
    st = dict(orch.last_state)

    # Regime posterior over the full 5-point scale.
    proba = {r.label: 0.0 for r in Regime}
    for r, p in eng.predict_regime_proba().items():
        proba[r.label] = round(float(p), 4)

    tm = eng.get_transition_matrix()
    transition = (
        {"index": list(tm.index), "columns": list(tm.columns),
         "data": [[round(float(v), 4) for v in row] for row in tm.to_numpy()]}
        if not tm.empty else {"index": [], "columns": [], "data": []}
    )

    weights = res.weights.iloc[-1]
    target_weights = {s: round(float(w), 4) for s, w in weights.items() if w > 1e-6}

    eq = res.equity
    tail = eq.iloc[-equity_tail:]
    equity_curve = [{"date": str(pd.Timestamp(cast(Any, d)).date()), "equity": round(float(v), 2)}
                    for d, v in tail.items()]

    alerts = _alerts(st, breaker, res)

    return {
        "as_of": str(pd.Timestamp(eq.index.max()).date()),
        "regime": st.get("regime", "Neutral"),
        "regime_proba": proba,
        "confidence": st.get("confidence", 1.0),
        "uncertainty": st.get("uncertainty", 0.0),
        "regime_stability": st.get("regime_stability", 1.0),
        "flicker_rate": st.get("flicker_rate", 0.0),
        "is_flickering": st.get("is_flickering", False),
        "vol_rank": st.get("vol_rank", 0.5),
        "vol_bucket": st.get("vol_bucket", "mid"),
        "active_strategy": st.get("active_strategy", "MidVolCautiousStrategy"),
        "blend": st.get("strategy_blend", {}),
        "gross_exposure": st.get("gross_exposure", 0.0),
        "target_weights": target_weights,
        "transition_matrix": transition,
        "regime_metadata": [
            {"regime": m.regime.label, "mean_return": round(m.mean_return, 4),
             "mean_vol": round(m.mean_vol, 4), "persistence": round(m.persistence, 3),
             "frequency": round(m.frequency, 3)}
            for m in eng.regime_metadata()
        ],
        "equity_curve": equity_curve,
        "alerts": alerts,
    }


def _alerts(state: dict, breaker: CircuitBreaker, res) -> list[dict]:
    out: list[dict] = []
    if breaker.trips > 0:
        out.append({"level": "error",
                    "msg": f"Circuit breaker fired {breaker.trips}x over the backtest."})
    if state.get("is_flickering"):
        out.append({"level": "warn", "msg": "Regime signal is flickering — treat with caution."})
    if state.get("regime") in ("Crash", "Bear"):
        out.append({"level": "warn", "msg": f"Risk-off regime: {state.get('regime')}."})
    if state.get("confidence", 1.0) < 0.5:
        out.append({"level": "warn", "msg": "Low regime confidence."})
    if not out:
        out.append({"level": "info", "msg": "All systems nominal."})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the platform main loop and write system state.")
    p.add_argument("--start", default=START)
    args = p.parse_args(argv)

    state = build_state(args.start)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))

    # Also refresh the live paper-account snapshot for the dashboard (no-op without keys).
    from .broker import write_snapshot
    acct = write_snapshot()
    if acct is not None:
        print(f"  live account   equity ${acct['equity']:,.0f}, "
              f"P&L ${acct['total_pl']:+,.0f}, {acct['n_positions']} positions")

    print(f"System state written to {STATE_PATH}")
    print(f"  as-of           {state['as_of']}")
    print(f"  regime          {state['regime']}  (confidence {state['confidence']:.0%}, "
          f"stability {state['regime_stability']:.0%})")
    print(f"  vol             rank {state['vol_rank']:.0%} -> {state['vol_bucket']} bucket")
    print(f"  active strategy {state['active_strategy']}")
    print(f"  blend           " + ", ".join(f"{k.replace('Strategy','')} {v:.0%}"
                                             for k, v in state["blend"].items()))
    print(f"  gross exposure  {state['gross_exposure']:.0%}")
    print(f"  top holdings    " + ", ".join(
        f"{s} {w:.0%}" for s, w in sorted(state["target_weights"].items(),
                                          key=lambda kv: -kv[1])[:6]))
    for a in state["alerts"]:
        print(f"  [{a['level'].upper()}] {a['msg']}")
    print("\nView the dashboard:  .venv/bin/python -m streamlit run "
          "src/quantlab/dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
