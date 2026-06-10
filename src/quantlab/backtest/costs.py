"""Transaction-cost model.

Costs are the silent killer (docs/00-PLAN.md section 0). We model, per dollar traded:
  - commission: Alpaca US equities is $0, but we keep the knob.
  - half-spread: you cross half the bid/ask on average.
  - slippage: market impact / not getting the price you saw. We deliberately set this
    *worse* than feels fair. If a strategy is only profitable at zero cost, it isn't.

Cost on a rebalance is charged on the *traded notional* (sum of |position changes| in
dollars), times the per-dollar rate below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.0   # per-dollar commission, basis points
    half_spread_bps: float = 2.0  # half the bid/ask spread, bps (liquid ETFs ~1-3)
    slippage_bps: float = 3.0     # deliberately pessimistic, bps

    @property
    def per_dollar_rate(self) -> float:
        """Total cost as a fraction of traded notional."""
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 1e4

    def cost(self, traded_notional: float) -> float:
        return abs(traded_notional) * self.per_dollar_rate


#: A frictionless model, used only to validate engine accounting (buy-and-hold == SPY).
ZERO_COST = CostModel(commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0)

#: The default we hold strategies to: pessimistic on purpose.
DEFAULT_COST = CostModel(commission_bps=0.0, half_spread_bps=2.0, slippage_bps=3.0)
