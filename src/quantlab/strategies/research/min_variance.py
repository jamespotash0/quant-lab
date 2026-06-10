"""Global minimum-variance portfolio (long-only).

The low-volatility anomaly says that, empirically, low-risk assets earn returns at
least as high as high-risk ones on a risk-adjusted basis — so a portfolio built to
*minimize* variance (rather than chase return) tends to beat the cap-weighted index on
Sharpe. We estimate the covariance of trailing daily returns, shrink it toward a
well-conditioned target (Ledoit-Wolf) for stability, then solve the constrained QP

    min   wᵀ Σ w     s.t.   Σ w = 1,   w >= 0

with SLSQP. The book is recomputed only every ``rebalance_days`` so day-to-day turnover
stays low; in between we re-return the held weights verbatim (zero turnover).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from ...data_pipeline.features import daily_returns
from ..base import History, Strategy


class MinimumVariance(Strategy):
    """Long-only global minimum-variance portfolio over the ETF universe.

    Reuses ``daily_returns`` over a trailing window, shrinks the sample covariance with
    Ledoit-Wolf, and solves the long-only min-variance QP via SLSQP. Rebalances every
    ``rebalance_days`` to keep turnover low.
    """

    def __init__(
        self,
        lookback: int = 189,
        rebalance_days: int = 21,
        target_gross: float = 1.0,
        min_names: int = 10,
    ) -> None:
        self.lookback = lookback
        self.rebalance_days = rebalance_days
        self.target_gross = target_gross
        self.min_names = min_names
        # Need a full lookback window of returns (+1 row for pct_change's leading NaN).
        self.warmup = lookback + 1
        self._held: dict[str, float] = {}
        self._age = 0

    def _solve_min_variance(self, cov: np.ndarray) -> np.ndarray:
        """Solve min wᵀΣw s.t. sum(w)=1, w>=0. Falls back to equal weight on failure."""
        n = cov.shape[0]
        w0 = np.full(n, 1.0 / n)
        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        def jac(w: np.ndarray) -> np.ndarray:
            return 2.0 * cov @ w

        res = minimize(
            objective,
            w0,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if not res.success or not np.all(np.isfinite(res.x)):
            return w0
        w = np.clip(res.x, 0.0, None)
        total = w.sum()
        if total <= 0 or not np.isfinite(total):
            return w0
        return w / total

    def target_weights(self, history: History) -> dict[str, float]:
        if len(history.close) < self.warmup:
            return {}

        # Hold the existing book between scheduled rebalances => no turnover.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)

        # Use only fully-priced names over the trailing window (drop leading-NaN names).
        window = history.close.iloc[-(self.lookback + 1):]
        rets = daily_returns(window).iloc[1:]
        rets = rets.dropna(axis=1, how="any")
        if rets.shape[1] < self.min_names:
            # Not enough clean names yet — hold whatever we had (or cash).
            self._age += 1
            return dict(self._held)

        symbols = list(rets.columns)
        try:
            cov = LedoitWolf().fit(rets.values).covariance_
        except Exception:
            cov = np.cov(rets.values, rowvar=False)
        # Symmetrize and nudge the diagonal for numerical positive-definiteness.
        cov = 0.5 * (cov + cov.T)
        cov = cov + np.eye(cov.shape[0]) * 1e-8
        if not np.all(np.isfinite(cov)):
            self._age += 1
            return dict(self._held)

        w = self._solve_min_variance(cov)
        weights = {
            s: float(wi * self.target_gross)
            for s, wi in zip(symbols, w)
            if wi > 1e-4
        }
        if not weights:
            self._age += 1
            return dict(self._held)

        self._held = weights
        self._age = 1
        return dict(weights)
