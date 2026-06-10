"""The strategy orchestrator: route capital across the three vol-regime strategies.

This is the brain-plus-allocator of the platform. Each decision day it:

  1. asks the HMM :class:`RegimeEngine` for the current regime, its confidence, and whether
     the signal is flickering;
  2. ranks the benchmark's volatility into a low/mid/high percentile with soft membership;
  3. blends the three vol-regime strategies by that membership — so calm markets lean on the
     aggressive low-vol-bull book and turbulent markets lean on the defensive book, which is
     where the "less capital when turbulent" behaviour comes from;
  4. modulates toward defense when the regime is bad OR the engine is uncertain/flickering
     (uncertainty -> caution); and
  5. rebalances on a slow cadence to keep turnover down.

Because the defensive strategy runs at low gross and the bull strategy at full gross, the
blend's gross exposure naturally scales down as volatility rises — no separate exposure dial
needed. The hard circuit breaker (:mod:`quantlab.backtest.risk`) still sits underneath as an
independent kill switch.
"""

from __future__ import annotations

from ...research.regime import Regime
from ...research.regime_engine import RegimeEngine
from ...research.vol_rank import bucket_membership, vol_rank
from ..base import History, Strategy
from .vol_highdefensive import HighVolDefensiveStrategy
from .vol_lowbull import LowVolBullStrategy
from .vol_midcautious import MidVolCautiousStrategy


class StrategyOrchestrator(Strategy):
    """Vol-regime-routed blend of LowVolBull / MidVolCautious / HighVolDefensive, modulated
    by HMM regime confidence. Exposes ``last_state`` for monitoring/dashboards."""

    thesis = (
        "Different volatility regimes reward different postures: ride momentum when calm, "
        "diversify when normal, defend when turbulent. Route capital across three "
        "regime-specialised books by a volatility-rank, leaning defensive when the regime "
        "model is uncertain, so exposure falls automatically as risk rises."
    )

    def __init__(
        self,
        engine: RegimeEngine | None = None,
        regime_symbol: str = "SPY",
        vol_lookback: int = 20,
        rank_window: int = 252,
        rebalance_days: int = 5,
        defensive_tilt: float = 0.30,
    ) -> None:
        self.engine = engine if engine is not None else RegimeEngine(regime_symbol=regime_symbol)
        self.regime_symbol = regime_symbol
        self.vol_lookback = vol_lookback
        self.rank_window = rank_window
        self.rebalance_days = rebalance_days
        self.defensive_tilt = defensive_tilt
        self.low = LowVolBullStrategy()
        self.mid = MidVolCautiousStrategy()
        self.high = HighVolDefensiveStrategy()
        self.warmup = max(
            self.engine.min_train + self.engine.vol_window,
            self.low.warmup, self.mid.warmup, self.high.warmup,
        )
        self._held: dict[str, float] = {}
        self._age = 0
        self.last_state: dict = {}

    @property
    def name(self) -> str:
        return "StrategyOrchestrator"

    def _membership(self, history: History) -> tuple[dict[str, float], dict]:
        """Soft {low, mid, high} weights from vol-rank, tilted defensive when the regime is
        bad or the engine is uncertain. Also returns a state dict for monitoring."""
        regime = self.engine.update(history)
        conf = self.engine.confidence()
        flick = self.engine.is_flickering()
        prices = history.close[self.regime_symbol].dropna()
        rank = vol_rank(prices, self.vol_lookback, self.rank_window)
        mem = bucket_membership(rank)

        # Uncertainty -> caution: shift mass toward the defensive (high-vol) book when the
        # regime is risk-off, the model is unsure, or the signal is flickering.
        risk_off = regime in (Regime.CRASH, Regime.BEAR)
        tilt = self.defensive_tilt * ((1.0 - conf) + (0.5 if risk_off else 0.0) + (0.5 if flick else 0.0))
        tilt = min(tilt, 0.8)
        if tilt > 0:
            mem = {k: v * (1.0 - tilt) for k, v in mem.items()}
            mem["high"] += tilt

        state = {
            "regime": regime.label,
            "confidence": round(conf, 3),
            "uncertainty": round(self.engine.uncertainty(), 3),
            "regime_stability": round(self.engine.get_regime_stability(), 3),
            "flicker_rate": round(self.engine.get_regime_flicker_rate(), 3),
            "is_flickering": flick,
            "vol_rank": round(rank, 3),
            "vol_bucket": max(mem, key=lambda k: mem[k]),
            "blend": {k: round(v, 3) for k, v in mem.items()},
        }
        return mem, state

    def target_weights(self, history: History) -> dict[str, float]:
        if len(history.close) < self.warmup:
            # still advance the engine so its history stays continuous
            self.engine.update(history)
            return {}

        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)
        self._age = 1

        mem, state = self._membership(history)
        books = {
            "low": self.low.target_weights(history),
            "mid": self.mid.target_weights(history),
            "high": self.high.target_weights(history),
        }
        blended: dict[str, float] = {}
        for bucket, w in mem.items():
            if w <= 0:
                continue
            for sym, sw in books[bucket].items():
                if sw > 0:
                    blended[sym] = blended.get(sym, 0.0) + w * sw

        self._held = blended
        state["gross_exposure"] = round(sum(blended.values()), 3)
        # Map bucket membership to the named strategies for the dashboard.
        state["strategy_blend"] = {
            "LowVolBullStrategy": mem["low"],
            "MidVolCautiousStrategy": mem["mid"],
            "HighVolDefensiveStrategy": mem["high"],
        }
        state["active_strategy"] = {
            "low": "LowVolBullStrategy", "mid": "MidVolCautiousStrategy",
            "high": "HighVolDefensiveStrategy",
        }[state["vol_bucket"]]
        self.last_state = state
        return dict(blended)
