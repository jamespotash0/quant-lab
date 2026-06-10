"""The allocation layer: let the brain decide how much capital is at risk.

Wraps any base strategy and scales its gross exposure by the current market regime (from
:class:`quantlab.research.regime.RegimeBrain`). The base strategy still chooses *what* to
hold; the regime decides *how much*. Calm, trending regimes get near-full capital; turbulent
regimes get scaled down toward cash; euphoria is trimmed because low-vol melt-ups tend to
precede reversals.

This is deliberately a soft, continuous risk dial — distinct from the hard circuit breaker
(:mod:`quantlab.backtest.risk`), which is a separate, model-independent kill switch.
"""

from __future__ import annotations

from ...research.regime import Regime, RegimeBrain
from ..base import History, Strategy

#: How much of the base book to hold in each regime (fraction of gross). Calm = more.
DEFAULT_EXPOSURE: dict[Regime, float] = {
    Regime.CRASH: 0.0,
    Regime.BEAR: 0.40,
    Regime.NEUTRAL: 0.70,
    Regime.BULL: 1.0,
    Regime.EUPHORIA: 0.85,
}


class RegimeAllocator(Strategy):
    """Scale a base strategy's weights by the HMM-classified regime's exposure budget."""

    def __init__(
        self,
        base: Strategy,
        brain: RegimeBrain | None = None,
        exposure: dict[Regime, float] | None = None,
    ) -> None:
        self.base = base
        self.brain = brain if brain is not None else RegimeBrain()
        self.exposure = exposure if exposure is not None else dict(DEFAULT_EXPOSURE)
        self.warmup = max(getattr(base, "warmup", 0), self.brain.min_train + self.brain.vol_window)
        self._last_regime = Regime.NEUTRAL

    @property
    def name(self) -> str:
        return f"Regime({type(self.base).__name__})"

    def target_weights(self, history: History) -> dict[str, float]:
        regime = self.brain.classify(history)
        self._last_regime = regime
        scale = self.exposure.get(regime, 0.70)
        if scale <= 0.0:
            return {}
        base_w = self.base.target_weights(history)
        return {s: w * scale for s, w in base_w.items() if w != 0.0}
