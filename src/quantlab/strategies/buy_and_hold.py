"""Buy-and-hold: the engine-validation strategy and a sanity baseline.

Holding 100% of one instrument through the engine must reproduce that instrument's own
return (net of costs/entry lag). If it doesn't, the engine's accounting is wrong and
nothing downstream can be trusted. See tests/test_engine_validation.py.
"""

from __future__ import annotations

from .base import History, Strategy


class BuyAndHold(Strategy):
    def __init__(self, symbol: str = "SPY") -> None:
        self.symbol = symbol

    @property
    def name(self) -> str:
        return f"BuyAndHold({self.symbol})"

    def target_weights(self, history: History) -> dict[str, float]:
        return {self.symbol: 1.0}
