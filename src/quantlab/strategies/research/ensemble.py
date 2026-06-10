"""Signal-ensemble meta-strategy (round-2 tool).

Blends the target books of several sub-strategies into one. The thesis behind ensembling
weak-but-uncorrelated signals is the only "free lunch" in investing: if each component
has a low-but-positive information ratio and their errors are imperfectly correlated, the
blend's Sharpe exceeds any single component's. We mix at the *weight* level (average the
target portfolios), which keeps the result long-only and gross-bounded by construction.
"""

from __future__ import annotations

from ..base import History, Strategy


class EnsembleStrategy(Strategy):
    """Combine sub-strategies by averaging their target weights.

    ``components`` is a list of (Strategy, mix) pairs; each sub-strategy's target book is
    scaled by its mix weight and the books are summed, then the whole is renormalized so
    gross does not exceed ``target_gross``. Cash (un-allocated weight) is preserved: if
    the blended gross is below target we do NOT lever up, we just hold the cash.
    """

    def __init__(
        self,
        components: list[tuple[Strategy, float]],
        target_gross: float = 1.0,
    ) -> None:
        if not components:
            raise ValueError("EnsembleStrategy needs at least one component")
        self.components = components
        self.target_gross = target_gross
        self.warmup = max(getattr(s, "warmup", 0) for s, _ in components)
        total = sum(m for _, m in components)
        self._mix = [(s, m / total) for s, m in components] if total > 0 else components

    @property
    def name(self) -> str:
        return "Ensemble(" + "+".join(type(s).__name__ for s, _ in self.components) + ")"

    def target_weights(self, history: History) -> dict[str, float]:
        blended: dict[str, float] = {}
        for strat, mix in self._mix:
            for sym, w in strat.target_weights(history).items():
                if w > 0:
                    blended[sym] = blended.get(sym, 0.0) + mix * w
        gross = sum(blended.values())
        # Only scale DOWN to respect the gross cap; never lever the cash up.
        if gross > self.target_gross and gross > 0:
            blended = {s: w * self.target_gross / gross for s, w in blended.items()}
        return {s: w for s, w in blended.items() if w > 0}
