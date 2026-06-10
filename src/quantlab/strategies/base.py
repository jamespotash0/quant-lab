"""The Strategy interface.

A Strategy maps *information available at a decision date* to *target portfolio
weights*. It never sees the future: the engine hands it a History object whose price
panels end at (and include) the decision date and nothing after.

Weights are fractions of total equity. Long-only to start (weights >= 0); the sum may
be < 1 (the remainder is cash) but the engine clamps gross exposure via the risk layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class History:
    """A point-in-time view of market data, truncated to end at ``as_of``.

    ``close``/``open`` are wide panels (index=date, columns=symbol). By construction
    they contain no rows after ``as_of``, so a strategy *cannot* read the future.
    """

    as_of: pd.Timestamp
    close: pd.DataFrame
    open: pd.DataFrame

    def __post_init__(self) -> None:
        if not self.close.empty and self.close.index.max() > self.as_of:
            raise ValueError(
                f"History.close contains data after as_of={self.as_of} — lookahead!"
            )

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


class Strategy:
    """Base class. Subclasses implement :meth:`target_weights`."""

    #: Minimum number of daily bars of history required before the strategy will trade.
    warmup: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    def target_weights(self, history: History) -> dict[str, float]:
        """Return desired weights {symbol: fraction_of_equity} as of the decision date.

        Called once per decision day with history through that day's close. The engine
        executes the implied rebalance at the next session's open.
        """
        raise NotImplementedError
