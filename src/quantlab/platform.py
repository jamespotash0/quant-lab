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


def build_state(start: str = START) -> dict:
    """Run the orchestrator over history and assemble the dashboard system-state dict."""
    from .data_pipeline.loaders import to_panel
    from .research.regime_engine import RegimeEngine
    from .research.vol_rank import vol_bucket, vol_rank
    from .strategies.base import History
    from .strategies.research.dual_momentum import DualMomentum
    from .strategies.research.recommended import SwingMomentumV3
    from .strategies.research.swing_breakout import SwingPivotBreakout
    from .strategies.research.vol_managed import VolatilityManaged

    bars = load_panel(start)

    # --- The strategy we actually trade: SwingMomentumV3 (under the circuit breaker). ---
    breaker = CircuitBreaker()
    res = run_backtest(bars, SwingMomentumV3(), cost_model=DEFAULT_COST, risk_overlay=breaker)
    eq = res.equity

    # --- Sleeve decomposition: where each weight comes from (DM 0.5 / Vol 0.25 / Swing 0.25). ---
    close = to_panel(bars, "close")
    open_ = to_panel(bars, "open")
    asof = close.index.max()
    h = History(as_of=asof, close=close, open=open_)
    dm = DualMomentum().target_weights(h)
    vm = VolatilityManaged().target_weights(h)
    sw = SwingPivotBreakout().target_weights(h)
    MIX = {"DualMomentum": 0.5, "VolatilityManaged": 0.25, "SwingPivotBreakout": 0.25}
    syms = set(dm) | set(vm) | set(sw)
    raw = {s: 0.5 * dm.get(s, 0.0) + 0.25 * vm.get(s, 0.0) + 0.25 * sw.get(s, 0.0) for s in syms}
    gross = sum(raw.values())
    norm = (1.0 / gross) if gross > 1.0 else 1.0
    sleeve_decomposition = [
        {"symbol": s,
         "DualMomentum": round(0.5 * dm.get(s, 0.0) * norm, 4),
         "VolatilityManaged": round(0.25 * vm.get(s, 0.0) * norm, 4),
         "SwingPivotBreakout": round(0.25 * sw.get(s, 0.0) * norm, 4),
         "total": round(raw[s] * norm, 4)}
        for s in sorted(syms, key=lambda x: -raw[x]) if raw[s] * norm > 1e-6
    ]
    # The book we'd actually trade today = the fresh blend (matches quantlab.live),
    # not the backtest's last *held* book (which may be mid-rebalance-cycle).
    target_weights = {s: round(raw[s] * norm, 4) for s in syms if raw[s] * norm > 1e-6}

    # --- Regime context (HMM brain — informational, NOT what's traded). Walk no-lookahead. ---
    eng = RegimeEngine()
    for d in close.index:
        eng.update(History(as_of=d, close=cast(Any, close.loc[:d]), open=cast(Any, open_.loc[:d])))
    proba = {r.label: 0.0 for r in Regime}
    for r, p in eng.predict_regime_proba().items():
        proba[r.label] = round(float(p), 4)
    tm = eng.get_transition_matrix()
    transition = (
        {"index": list(tm.index), "columns": list(tm.columns),
         "data": [[round(float(v), 4) for v in row] for row in tm.to_numpy()]}
        if not tm.empty else {"index": [], "columns": [], "data": []}
    )
    spy = close["SPY"].dropna() if "SPY" in close.columns else None
    rank = vol_rank(spy) if spy is not None else 0.5

    st = {
        "regime": eng.current_regime().label,
        "is_flickering": eng.is_flickering(),
        "confidence": round(eng.confidence(), 3),
    }
    # Full backtest equity curve from the $100k start, downsampled to ~weekly to keep the
    # payload light (the very first point is the $100k initial capital).
    step = max(1, len(eq) // 400)
    sampled = eq.iloc[::step]
    if len(eq) and sampled.index[-1] != eq.index[-1]:
        sampled = pd.concat([sampled, eq.iloc[-1:]])
    equity_curve = [{"date": str(pd.Timestamp(cast(Any, d)).date()), "equity": round(float(v), 2)}
                    for d, v in sampled.items()]
    alerts = _alerts(st, breaker, res)

    return {
        "as_of": str(pd.Timestamp(eq.index.max()).date()),
        "strategy": "SwingMomentumV3",
        "regime": st["regime"],
        "regime_proba": proba,
        "confidence": st["confidence"],
        "uncertainty": round(eng.uncertainty(), 3),
        "regime_stability": round(eng.get_regime_stability(), 3),
        "flicker_rate": round(eng.get_regime_flicker_rate(), 3),
        "is_flickering": st["is_flickering"],
        "vol_rank": round(rank, 3),
        "vol_bucket": vol_bucket(rank),
        "active_strategy": "SwingMomentumV3",
        "blend": MIX,
        "gross_exposure": round(sum(target_weights.values()), 3),
        "target_weights": target_weights,
        "sleeve_decomposition": sleeve_decomposition,
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
