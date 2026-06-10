"""The safety net: hard circuit breakers that act on equity, not on signals.

This layer is deliberately dumb and independent of the brain and the strategy. It watches
realized account equity and, if losses breach a hard limit, flattens the book to cash —
regardless of what the model "thinks". Two independent triggers:

  * DRAWDOWN breaker: peak-to-current loss exceeds ``max_drawdown`` → flat for
    ``cooldown_days`` sessions, then re-enter with a reset high-water mark (a cooldown,
    not an equity-recovery trigger — in cash the equity is frozen, so a recovery trigger
    could never fire and would stick the book in cash forever).
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
        cooldown_days: int = 21,
        max_daily_loss: float = 0.05,
    ) -> None:
        self.max_drawdown = max_drawdown
        self.cooldown_days = cooldown_days
        self.max_daily_loss = max_daily_loss
        self._tripped = False
        self._cooldown = 0
        self._peak = float("-inf")
        self.trips = 0  # diagnostic: how many times it fired

    def apply(self, pending: dict[str, float], equity: list[float]) -> dict[str, float]:
        if not equity:
            return pending
        current = equity[-1]
        self._peak = max(self._peak, current)

        if self._tripped:
            # Re-enter after a fixed cooldown, NOT when equity recovers — in cash the equity
            # is frozen, so a recovery-based trigger could never fire (it would stick in cash
            # forever). On re-entry we reset the high-water mark to the realized level, so we
            # don't immediately re-trip on the stale pre-crash peak.
            self._cooldown -= 1
            if self._cooldown <= 0:
                self._tripped = False
                self._peak = current
            else:
                return {}

        drawdown = current / self._peak - 1.0 if self._peak > 0 else 0.0
        daily = (current / equity[-2] - 1.0) if len(equity) >= 2 and equity[-2] > 0 else 0.0

        if drawdown <= -self.max_drawdown or daily <= -self.max_daily_loss:
            self._tripped = True
            self._cooldown = self.cooldown_days
            self.trips += 1
            return {}
        return pending
