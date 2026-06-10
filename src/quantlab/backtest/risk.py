"""The safety net: hard circuit breakers that act on equity, not on signals.

This layer is deliberately dumb and independent of the brain and the strategy. It watches
realized account equity and, if losses breach a hard limit, flattens the book to cash —
regardless of what the model "thinks". Two independent triggers:

  * DRAWDOWN breaker: peak-to-current loss exceeds ``max_drawdown`` → flat until equity
    recovers to within ``reentry_drawdown`` of the prior peak (hysteresis stops it
    flip-flopping at the threshold).
  * DAILY-LOSS breaker: a single-day loss worse than ``max_daily_loss`` → flat (then the
    drawdown rule governs re-entry).

A real desk keeps risk limits in a separate system from the alpha model for exactly this
reason: when the model is wrong in a way it can't see, the limit still fires. The engine
applies the breaker to the *next* order, so a breach today means we are flat at tomorrow's
open.
"""

from __future__ import annotations


class CircuitBreaker:
    """Stateful equity kill-switch. Call :meth:`apply` with the pending target weights and
    the equity history; it returns either the weights unchanged or ``{}`` (go to cash)."""

    def __init__(
        self,
        max_drawdown: float = 0.15,
        reentry_drawdown: float = 0.07,
        max_daily_loss: float = 0.05,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.reentry_drawdown = reentry_drawdown
        self.max_daily_loss = max_daily_loss
        self._tripped = False
        self._peak = float("-inf")
        self.trips = 0  # diagnostic: how many times it fired

    def apply(self, pending: dict[str, float], equity: list[float]) -> dict[str, float]:
        if not equity:
            return pending
        current = equity[-1]
        self._peak = max(self._peak, current)

        if self._tripped:
            # Re-enter only once we've climbed back to within reentry_drawdown of the peak.
            if current >= self._peak * (1.0 - self.reentry_drawdown):
                self._tripped = False
            else:
                return {}

        drawdown = current / self._peak - 1.0 if self._peak > 0 else 0.0
        daily = (current / equity[-2] - 1.0) if len(equity) >= 2 and equity[-2] > 0 else 0.0

        if drawdown <= -self.max_drawdown or daily <= -self.max_daily_loss:
            self._tripped = True
            self.trips += 1
            return {}
        return pending
