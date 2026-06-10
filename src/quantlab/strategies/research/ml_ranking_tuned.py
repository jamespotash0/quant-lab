"""Tunable gradient-boosted cross-sectional ranker (handoff surface for a quant).

This is a refactor of :mod:`quantlab.strategies.research.ml_ranking` whose sole job is
to be a clean, grid-searchable handoff surface. The economics are identical: a nonlinear
ensemble (``HistGradientBoostingRegressor``) learns how a panel of *standard,
point-in-time* features — multi-horizon momentum, realized vol, short-term reversal, and
distance-from-moving-average — maps to forward returns across the universe. The bet is
that the boosting ensemble captures *interactions* among these features (e.g. "high
momentum only pays when vol is low") that a single-factor linear strategy cannot.

No-lookahead label construction is preserved verbatim from the original: a row dated
``t`` is labelled with its forward ``horizon``-day return (close[t+h]/close[t]-1), and any
row whose forward window ``t+h`` would peek past today is dropped from training. We train
on an **expanding window of the strict past**, retrain only every ``rebalance_days`` days
(NOT daily — both for compute and to keep turnover low), and persist the fitted model and
the current book on ``self``. On a rebalance day we predict today's cross-section, go long
the top ``k`` predicted names sized inverse-vol with a per-name cap, and hold that book
unchanged until the next retrain (so day-to-day turnover is near zero).

Every model hyperparameter is exposed as an ``__init__`` kwarg with a sensible default so
a quant can grid-search them via :mod:`quantlab.research.sweep`. ``random_state`` is fixed
so two runs are bit-identical (criterion 1: reproducible).
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from ...data_pipeline.features import (
    momentum,
    realized_vol,
    short_term_reversal,
)
from ..base import History, Strategy


class MLRankingTuned(Strategy):
    """Long the top-``k`` names ranked by a *tunable* gradient-boosted forward-return
    model, inverse-vol sized with a per-name cap, retrained every ``rebalance_days``.

    Features per (date, symbol) are all trailing/point-in-time: momentum at 21/63/126/
    252, realized vol at 21/63, 5-day short-term reversal, and distance from the 50/200
    day moving averages. The label for a row dated ``t`` is the ``horizon``-day forward
    return; rows whose ``t+h`` extends past today are dropped, so labels never use the
    future. The model and book persist on ``self`` between rebalances => near-zero
    day-to-day turnover. All boosting hyperparameters are ``__init__`` kwargs for tuning.
    """

    #: Economic rationale, surfaced to the scorecard's "clear thesis" gate.
    thesis = (
        "A nonlinear gradient-boosted ensemble can capture interactions among standard "
        "point-in-time features (multi-horizon momentum, realized vol, short-term "
        "reversal, distance-from-moving-average) that single-factor linear strategies "
        "miss, and rank the cross-section of ETFs by forward return more richly."
    )

    #: Momentum lookbacks (days). Skip the most recent 21d on the longer ones.
    _MOM_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
    #: Realized-vol lookbacks (days).
    _VOL_LOOKBACKS: tuple[int, ...] = (21, 63)
    #: Moving-average windows for the distance-from-MA feature.
    _MA_WINDOWS: tuple[int, ...] = (50, 200)

    def __init__(
        self,
        # ---- signal / portfolio knobs --------------------------------------
        horizon: int = 10,
        k: int = 4,
        rebalance_days: int = 21,
        vol_lookback: int = 21,
        per_name_cap: float = 0.30,
        target_gross: float = 1.0,
        # ---- HistGradientBoostingRegressor hyperparameters (tune these) ----
        learning_rate: float = 0.05,
        max_depth: int = 2,
        max_leaf_nodes: int | None = 31,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        max_iter: int = 200,
        # ---- training-data knobs -------------------------------------------
        min_train_rows: int = 250,
        sample_stride: int = 5,
        random_state: int = 0,
    ) -> None:
        # Signal / portfolio.
        self.horizon = horizon
        self.k = k
        self.rebalance_days = rebalance_days
        self.vol_lookback = vol_lookback
        self.per_name_cap = per_name_cap
        self.target_gross = target_gross

        # Boosting hyperparameters (the handoff surface).
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_iter = max_iter

        # Training data.
        # Minimum number of (row) training samples before we trust the model.
        self.min_train_rows = min_train_rows
        # Subsample the training panel in time to keep retrains fast and decorrelated.
        self.sample_stride = sample_stride
        # FIXED seed => reproducible (criterion 1).
        self.random_state = random_state

        # Warmup: longest feature window (252) + a horizon + buffer so the very first
        # rebalance already has labelled rows whose forward window has closed.
        self.warmup = max(self._MA_WINDOWS + self._MOM_LOOKBACKS) + horizon + 30

        # Persisted across calls: the fitted model, current book, and rebalance age.
        self._model: HistGradientBoostingRegressor | None = None
        self._held: dict[str, float] = {}
        self._age = 0

    # ------------------------------------------------------------------ features
    def _feature_panels(self, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Build the named feature panels (each wide: index=date, columns=symbol)."""
        feats: dict[str, pd.DataFrame] = {}
        for lb in self._MOM_LOOKBACKS:
            # Skip the last 21d on the longer windows (classic), none on the 21d one.
            skip = 21 if lb > 21 else 0
            feats[f"mom_{lb}"] = momentum(close, lookback=lb, skip=skip)
        for lb in self._VOL_LOOKBACKS:
            feats[f"vol_{lb}"] = realized_vol(close, lookback=lb)
        feats["rev_5"] = short_term_reversal(close, lookback=5)
        for w in self._MA_WINDOWS:
            ma = close.rolling(w).mean()
            # Distance from the moving average, as a fraction. Row t uses closes <= t.
            feats[f"dma_{w}"] = close / ma - 1.0
        return feats

    def _stack_rows(
        self,
        feats: dict[str, pd.DataFrame],
        dates: pd.DatetimeIndex,
        label: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray | None, list[tuple[pd.Timestamp, str]]]:
        """Flatten the feature panels into an (n_samples, n_features) design matrix over
        the given ``dates``. If ``label`` is provided, also return the aligned y vector.
        Rows with any NaN feature (or NaN label, when labelling) are dropped."""
        names = list(feats.keys())
        cols = list(next(iter(feats.values())).columns)

        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        idx: list[tuple[pd.Timestamp, str]] = []
        # Pre-grab per-feature frames restricted to the requested dates.
        sub = {n: feats[n].reindex(dates) for n in names}
        lab = label.reindex(dates) if label is not None else None

        for d in dates:
            for s in cols:
                # .at returns a broadly-typed Scalar; these frames are float, so narrow.
                row = [cast(float, sub[n].at[d, s]) for n in names]
                if any(not np.isfinite(v) for v in row):
                    continue
                if lab is not None:
                    yv = cast(float, lab.at[d, s])
                    if not np.isfinite(yv):
                        continue
                    y_rows.append(float(yv))
                x_rows.append([float(v) for v in row])
                idx.append((d, s))

        x = np.asarray(x_rows, dtype=float) if x_rows else np.empty((0, len(names)))
        y = np.asarray(y_rows, dtype=float) if label is not None else None
        return x, y, idx

    # ------------------------------------------------------------------ training
    def _fit(self, close: pd.DataFrame, feats: dict[str, pd.DataFrame]) -> bool:
        """Retrain the model on the strict past. Returns True if a model was fit.

        Labels are forward ``horizon``-day returns; a row dated ``t`` is only usable if
        ``t + horizon`` exists at or before today (the last row), so the label never
        reads beyond the panel we were handed. We subsample dates by ``sample_stride``.
        """
        # Forward return aligned back to the decision date t: fwd[t] = close[t+h]/close[t]-1.
        fwd = close.shift(-self.horizon) / close - 1.0

        n = len(close)
        # The last date whose full forward window has closed within this panel.
        last_label_pos = n - 1 - self.horizon
        if last_label_pos < self.warmup:
            return False

        # Train dates: strided, from warmup up to (and including) the last labelled date.
        train_positions = range(self.warmup, last_label_pos + 1, self.sample_stride)
        train_dates = close.index[list(train_positions)]
        if len(train_dates) == 0:
            return False

        x, y, _ = self._stack_rows(feats, train_dates, label=fwd)  # type: ignore[arg-type]
        if y is None or len(y) < self.min_train_rows:
            return False

        model = HistGradientBoostingRegressor(
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            # None is valid at runtime (= unlimited leaves) but the stub types it int.
            max_leaf_nodes=cast(int, self.max_leaf_nodes),
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(x, y)
        self._model = model
        return True

    # ------------------------------------------------------------------ predict + size
    def _predict_today(
        self, feats: dict[str, pd.DataFrame], today: pd.Timestamp
    ) -> pd.Series:
        """Model prediction for every priced symbol on ``today`` (NaN-feature names
        are simply absent from the returned Series)."""
        assert self._model is not None
        x, _, idx = self._stack_rows(feats, pd.DatetimeIndex([today]), label=None)
        if x.shape[0] == 0:
            return pd.Series(dtype=float)
        preds = self._model.predict(x)
        return pd.Series(preds, index=[s for _, s in idx])

    def _size(self, winners: list[str], close: pd.DataFrame) -> dict[str, float]:
        """Inverse-realized-vol weights among ``winners``, capped + renormalized."""
        vol = realized_vol(close, lookback=self.vol_lookback).iloc[-1]
        inv_vol: dict[str, float] = {}
        for s in winners:
            v = vol.get(s, np.nan)
            inv_vol[s] = 1.0 / v if (np.isfinite(v) and v > 0) else np.nan

        finite = [x for x in inv_vol.values() if np.isfinite(x)]
        if not finite:
            raw = {s: 1.0 for s in winners}
        else:
            med = float(np.median(finite))
            raw = {s: (x if np.isfinite(x) else med) for s, x in inv_vol.items()}

        total = sum(raw.values())
        if total <= 0:
            return {}
        weights = {s: self.target_gross * x / total for s, x in raw.items()}
        weights = {s: min(w, self.per_name_cap) for s, w in weights.items()}
        capped_total = sum(weights.values())
        if capped_total <= 0:
            return {}
        scale = min(1.0, self.target_gross / capped_total)
        return {s: w * scale for s, w in weights.items()}

    # ------------------------------------------------------------------ interface
    def target_weights(self, history: History) -> dict[str, float]:
        close = history.close
        if len(close) < self.warmup:
            return {}

        # Hold the existing book between rebalances => near-zero turnover.
        if self._held and self._age < self.rebalance_days:
            self._age += 1
            return dict(self._held)

        feats = self._feature_panels(close)

        # Retrain on the strict past; if we can't (too little labelled data), keep any
        # existing book, else stay flat.
        if not self._fit(close, feats):
            self._age += 1
            return dict(self._held)

        today = close.index[-1]
        preds = self._predict_today(feats, today)
        last_close = close.iloc[-1]
        preds = preds[preds.index.map(lambda s: np.isfinite(last_close.get(s, np.nan)))]
        if preds.empty:
            self._age += 1
            return dict(self._held)

        winners = list(preds.sort_values(ascending=False).index[: self.k])
        weights = self._size(winners, close)
        if not weights:
            self._age += 1
            return dict(self._held)

        self._held = weights
        self._age = 1
        return dict(weights)
