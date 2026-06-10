"""Hierarchical Risk Parity (Lopez de Prado, 2016).

A risk-allocation strategy that builds a long-only, fully-invested book without ever
inverting the covariance matrix — the step that makes classic mean-variance / risk
parity numerically fragile on near-singular correlation panels (and ETF universes are
exactly that: sectors and broad indices are all heavily correlated).

The recipe, per rebalance, on a trailing ~252-day window of daily returns:
  1. Correlation matrix -> distance matrix  d_ij = sqrt((1 - corr_ij) / 2).
  2. Hierarchically cluster the assets (scipy single-linkage on that distance).
  3. *Quasi-diagonalize*: reorder assets by the dendrogram so similar assets sit
     adjacent, concentrating correlation mass near the diagonal.
  4. *Recursive bisection*: split the ordered list in two, size each half by its
     inverse-variance portfolio variance, and scale the two halves so the lower-variance
     cluster gets more weight. Recurse until singletons remain.

Within a cluster, allocation is inverse-variance (the diagonal of the covariance),
so we never touch off-diagonal inverses — the source of risk-parity's instability.

Turnover control: clustering is expensive and the resulting weights are stable, so we
recompute only every ``rebalance_days`` and otherwise return the cached book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ...data_pipeline.features import daily_returns
from ..base import History, Strategy


def _inverse_variance_weights(cov: pd.DataFrame) -> np.ndarray:
    """Inverse-variance portfolio: weight each asset by 1/var, normalized to sum 1."""
    ivp = 1.0 / np.diag(cov.values)
    ivp /= ivp.sum()
    return ivp


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Variance of the inverse-variance portfolio formed from ``items`` only."""
    sub = cov.loc[items, items]
    w = _inverse_variance_weights(sub).reshape(-1, 1)
    return float((w.T @ sub.values @ w).item())


def _quasi_diagonal_order(link: np.ndarray, n_items: int) -> list[int]:
    """Walk the linkage tree to recover the leaf order that puts correlated assets
    adjacent (the classic Lopez de Prado seriation, done without recursion)."""
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n_items:
        new_order: list[int] = []
        for idx in order:
            if idx < n_items:
                new_order.append(idx)
            else:  # it's a merged cluster -> expand into its two children
                row = link[idx - n_items]
                new_order.append(int(row[0]))
                new_order.append(int(row[1]))
        order = new_order
    return order


def _recursive_bisection(cov: pd.DataFrame, ordered: list[str]) -> pd.Series:
    """Allocate weights by recursively splitting the quasi-diagonalized list and
    sizing each half inversely to its cluster variance."""
    weights = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        # bisect every cluster with more than one member; singletons are terminal
        clusters = [
            c[half:i]
            for c in clusters
            if len(c) > 1
            for half, i in ((0, len(c) // 2), (len(c) // 2, len(c)))
        ]
        # clusters now come in (left, right) pairs from each split
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            denom = var_left + var_right
            alpha = 0.5 if denom <= 0.0 else 1.0 - var_left / denom
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
    return weights


class HierarchicalRiskParity(Strategy):
    """HRP risk-allocation book over the ETF universe (Lopez de Prado, 2016).

    Long-only, weights sum to ~1.0. Recomputed every ``rebalance_days`` from a trailing
    ``lookback``-day return window; cached in between so day-to-day turnover stays low.
    """

    def __init__(self, lookback: int = 252, rebalance_days: int = 21) -> None:
        self.lookback = lookback
        self.rebalance_days = rebalance_days
        #: Need a full lookback window of returns (+1 row to difference into returns).
        self.warmup = lookback + 1
        self._weights: dict[str, float] = {}
        self._age = 0

    def _compute_weights(self, history: History) -> dict[str, float]:
        close = history.close
        # Only assets fully priced across the whole trailing window are clusterable;
        # newly-listed names (leading NaNs) would poison the correlation matrix.
        window = close.iloc[-(self.lookback + 1):]
        rets = daily_returns(window).iloc[1:]
        rets = rets.dropna(axis=1, how="any")
        if rets.shape[1] < 2:
            return {}

        # Drop any zero-variance column (a flat series breaks the correlation/distance).
        std = rets.std()
        rets = rets.loc[:, std > 0.0]
        if rets.shape[1] < 2:
            return {}

        cov = rets.cov()
        corr = rets.corr()

        # Correlation -> distance, then condensed form for scipy linkage.
        dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, None))
        np.fill_diagonal(dist, 0.0)
        dist = (dist + dist.T) / 2.0  # enforce exact symmetry for squareform
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")

        order = _quasi_diagonal_order(link, n_items=corr.shape[0])
        ordered_syms = [corr.index[i] for i in order]

        w = _recursive_bisection(cov, ordered_syms)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            return {}
        w = w / total
        return {str(sym): float(val) for sym, val in w.items() if val > 0.0}

    def target_weights(self, history: History) -> dict[str, float]:
        if len(history.close) < self.warmup:
            return {}

        # Only re-cluster every rebalance_days; otherwise reuse the cached book so the
        # engine sees near-identical targets and turnover stays minimal.
        if not self._weights or self._age >= self.rebalance_days:
            new_weights = self._compute_weights(history)
            if new_weights:
                self._weights = new_weights
                self._age = 0
        self._age += 1
        return dict(self._weights)
