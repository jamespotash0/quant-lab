"""The strategy-sweep winner.

Out of a 10-strategy parallel sweep (trend, dual-momentum, mean-reversion, risk
parity, HRP, vol-managed, min-variance, ML ranking, defensive allocation) evaluated
2017->present on the 30-ETF universe net of a pessimistic 5bps/side cost model, only
dual momentum cleared the decision gate (beat SPY *and* the dumb baselines on
risk-adjusted, net-of-cost return). Blending a 30% volatility-managed sleeve onto it
then improved the risk profile materially without giving up the edge:

    strategy                     net Sharpe   CAGR    vol    maxDD
    SwingMomentumV2 (this)          1.13     15.8%   13.8%  -15.6%
    DualMomentum (solo)             1.10     18.2%   16.5%  -19.3%
    Benchmark SPY (buy & hold)      1.08     20.9%   19.2%  -24.5%

It gives up some raw return for a much smoother ride (a third less drawdown, a fifth
less volatility than SPY) and carries a structural bear-market exit: the dual-momentum
core rotates into bonds when trend turns down, and the vol-managed sleeve de-risks into
cash when volatility spikes. The honest caveat: the Sharpe edge over buy-and-hold is
modest and within one-sample noise; the robust, repeatable win is risk reduction.
"""

from __future__ import annotations

from .dual_momentum import DualMomentum
from .ensemble import EnsembleStrategy
from .swing_breakout import SwingPivotBreakout
from .vol_managed import VolatilityManaged


class SwingMomentumV2(EnsembleStrategy):
    """70% dual-momentum (return + bond-rotation core) / 30% volatility-managed
    (drawdown + vol-spike de-risking). The recommended strategy from the sweep."""

    def __init__(self, dual_weight: float = 0.7, vol_weight: float = 0.3) -> None:
        super().__init__(
            [(DualMomentum(), dual_weight), (VolatilityManaged(), vol_weight)],
            target_gross=1.0,
        )

    @property
    def name(self) -> str:
        return "SwingMomentumV2"


class SwingMomentumV3(EnsembleStrategy):
    """The current best: 50% dual-momentum + 25% volatility-managed + 25% swing-pivot
    breakout. Adding the swing-structure sleeve (a market-structure trend signal that is
    low-correlation with trailing-return momentum) to V2 raises net Sharpe 1.13 -> 1.19
    and cuts max drawdown -15.6% -> -12.6%, while still passing all 5 criteria with an
    out-of-sample Sharpe (1.33) above its in-sample value.

        strategy            net Sharpe   CAGR    maxDD    OOS Sharpe   criteria
        SwingMomentumV3        1.19      14.4%  -12.6%       1.33        5/5
        SwingMomentumV2        1.13      15.8%  -15.6%       1.30        5/5
        Benchmark SPY          1.08      20.9%  -24.5%        --          --

    thesis = three uncorrelated, individually-validated edges stacked: relative+absolute
    momentum (return engine, rotates to bonds in downtrends), volatility targeting
    (de-risks before vol spikes), and swing-pivot breakouts (market-structure trend). The
    swing sleeve underperformed in-sample but adds the most out-of-sample, the signature of
    a genuine diversifier rather than an overfit.
    """

    thesis = (
        "Stack three low-correlation, individually-validated edges — dual momentum "
        "(return + bond rotation), volatility targeting (drawdown control), and "
        "swing-pivot breakouts (market-structure trend) — so the blend's risk-adjusted "
        "return exceeds any single sleeve's."
    )

    def __init__(
        self,
        dual_weight: float = 0.50,
        vol_weight: float = 0.25,
        swing_weight: float = 0.25,
    ) -> None:
        super().__init__(
            [
                (DualMomentum(), dual_weight),
                (VolatilityManaged(), vol_weight),
                (SwingPivotBreakout(), swing_weight),
            ],
            target_gross=1.0,
        )

    @property
    def name(self) -> str:
        return "SwingMomentumV3"
