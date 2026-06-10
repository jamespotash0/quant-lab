"""HMM regime-detection engine with full observability.

An upgrade of :class:`quantlab.research.regime.RegimeBrain` into a production-grade engine:

  * Gaussian HMM with AUTOMATIC MODEL SELECTION — refits candidate models across a range
    of state counts and picks the one minimising BIC, so the number of regimes is learned
    from the data rather than hard-coded.
  * NO LOOKAHEAD — refit only on the strict past; "today's" regime is the filtered estimate
    (forward pass over data up to today only). ``random_state`` fixed for reproducibility.
  * REGIME STABILITY FILTER — a candidate new regime must persist for ``confirm_bars``
    observations before it's accepted, which debounces one-bar flicker.
  * OBSERVABILITY API — predict_regime_proba, get_transition_matrix, get_regime_stability,
    detect_regime_change, get_regime_flicker_rate, is_flickering, confidence/uncertainty,
    and structured regime metadata.

Regimes are mapped to a fixed risk-ordered scale (crash..euphoria) by each fitted model's
emission means, so labels are stable across refits even though the HMM's internal state
numbering is arbitrary.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from .regime import Regime, _features

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


@dataclass
class RegimeMetadata:
    """Descriptive stats for one learned regime."""

    regime: Regime
    mean_return: float   # annualized, from the return feature
    mean_vol: float      # annualized realized vol
    persistence: float   # P(stay) — the transition-matrix diagonal for this regime
    frequency: float     # share of the recent classified history in this regime


class RegimeEngine:
    """Self-selecting Gaussian-HMM regime detector with a stability filter and a rich
    observability API. Stateful: refits every ``refit_days`` calls and remembers the
    recent regime path so it can report flicker/stability."""

    def __init__(
        self,
        regime_symbol: str = "SPY",
        state_range: tuple[int, ...] = (2, 3, 4, 5),
        ret_window: int = 10,
        vol_window: int = 20,
        refit_days: int = 63,
        min_train: int = 504,
        predict_window: int = 252,
        confirm_bars: int = 3,
        flicker_window: int = 21,
        random_state: int = 42,
    ) -> None:
        self.regime_symbol = regime_symbol
        self.state_range = state_range
        self.ret_window = ret_window
        self.vol_window = vol_window
        self.refit_days = refit_days
        self.min_train = min_train
        self.predict_window = predict_window
        self.confirm_bars = confirm_bars
        self.flicker_window = flicker_window
        self.random_state = random_state

        self._model = None
        self._n_states = 0
        self._state_to_regime: dict[int, Regime] = {}
        self._since_fit = 10**9
        # Confirmed regime path + raw (pre-filter) path, for stability/flicker metrics.
        self._confirmed: deque[Regime] = deque(maxlen=max(flicker_window, 64))
        self._raw_history: deque[Regime] = deque(maxlen=max(flicker_window, 64))
        self._current = Regime.NEUTRAL
        self._candidate: Regime | None = None
        self._candidate_count = 0
        self._last_proba: np.ndarray | None = None
        self._changed = False

    # ------------------------------------------------------------------ fitting
    def _select_and_fit(self, x: np.ndarray) -> None:
        """Fit a Gaussian HMM for each candidate state count and keep the best by BIC."""
        from hmmlearn.hmm import GaussianHMM

        best, best_bic = None, np.inf
        for k in self.state_range:
            if len(x) < k * 20:  # need enough data to estimate k states
                continue
            try:
                m = GaussianHMM(n_components=k, covariance_type="diag",
                                n_iter=100, random_state=self.random_state)
                m.fit(x)
                bic = m.bic(x)
            except Exception:
                continue
            if np.isfinite(bic) and bic < best_bic:
                best, best_bic = m, bic
        if best is None:
            return
        self._model = best
        self._n_states = best.n_components
        # Map raw states -> risk-ordered Regime by ascending mean of the return feature.
        order = np.argsort(best.means_[:, 0])
        scale = self._regime_scale(best.n_components)
        self._state_to_regime = {int(s): scale[i] for i, s in enumerate(order)}

    @staticmethod
    def _regime_scale(n: int) -> list[Regime]:
        """Map ``n`` learned states onto the 5-point risk scale (fewer states collapse
        toward the middle of the scale)."""
        full = [Regime.CRASH, Regime.BEAR, Regime.NEUTRAL, Regime.BULL, Regime.EUPHORIA]
        if n >= 5:
            return full
        # pick n evenly-spaced labels spanning crash..euphoria
        idx = np.linspace(0, 4, n).round().astype(int)
        return [full[i] for i in idx]

    # ------------------------------------------------------------------ core step
    def _feature_matrix(self, history) -> np.ndarray | None:
        if self.regime_symbol not in history.close.columns:
            return None
        prices = history.close[self.regime_symbol].dropna()
        feats = _features(prices, self.ret_window, self.vol_window)
        if len(feats) < self.min_train:
            return None
        return feats.to_numpy()

    def update(self, history) -> Regime:
        """Advance the engine by one observation (the last row of ``history``) and return
        the stability-filtered current regime. Call once per decision day. No lookahead."""
        x_all = self._feature_matrix(history)
        if x_all is None:
            return self._current

        if self._model is None or self._since_fit >= self.refit_days:
            try:
                self._select_and_fit(x_all)
                self._since_fit = 0
            except Exception:
                self._since_fit = 0
        else:
            self._since_fit += 1
        if self._model is None:
            return self._current

        window = x_all[-self.predict_window:]
        try:
            proba = cast(np.ndarray, self._model.predict_proba(window))[-1]
        except Exception:
            return self._current
        self._last_proba = proba
        raw = self._state_to_regime.get(int(proba.argmax()), Regime.NEUTRAL)
        self._raw_history.append(raw)

        # Stability filter: a new regime must persist confirm_bars before it's confirmed.
        self._changed = False
        if raw == self._current:
            self._candidate, self._candidate_count = None, 0
        else:
            if raw == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate, self._candidate_count = raw, 1
            if self._candidate_count >= self.confirm_bars:
                self._current = raw
                self._candidate, self._candidate_count = None, 0
                self._changed = True
        self._confirmed.append(self._current)
        return self._current

    # ------------------------------------------------------------------ observability
    def current_regime(self) -> Regime:
        return self._current

    def predict_regime_proba(self) -> dict[Regime, float]:
        """Filtered posterior over regimes for the latest observation."""
        if self._last_proba is None:
            return {self._current: 1.0}
        out: dict[Regime, float] = {}
        for state, p in enumerate(self._last_proba):
            r = self._state_to_regime.get(state, Regime.NEUTRAL)
            out[r] = out.get(r, 0.0) + float(p)
        return out

    def confidence(self) -> float:
        """1 - normalized entropy of the posterior: 1.0 = certain, 0.0 = uniform."""
        if self._last_proba is None or len(self._last_proba) <= 1:
            return 1.0
        p = self._last_proba[self._last_proba > 1e-12]
        ent = -np.sum(p * np.log(p))
        return float(1.0 - ent / np.log(len(self._last_proba)))

    def uncertainty(self) -> float:
        return 1.0 - self.confidence()

    def get_transition_matrix(self) -> pd.DataFrame:
        """Transition probabilities relabeled to the Regime scale (rows=from, cols=to)."""
        if self._model is None:
            return pd.DataFrame()
        labels = [self._state_to_regime.get(s, Regime.NEUTRAL).label
                  for s in range(self._n_states)]
        return pd.DataFrame(self._model.transmat_, index=labels, columns=labels)

    def get_regime_stability(self) -> float:
        """Persistence of the current regime = P(stay) from the transition matrix."""
        if self._model is None:
            return 1.0
        for state, reg in self._state_to_regime.items():
            if reg == self._current:
                return float(self._model.transmat_[state, state])
        return 1.0

    def detect_regime_change(self) -> bool:
        """True if the last :meth:`update` confirmed a regime switch."""
        return self._changed

    def get_regime_flicker_rate(self) -> float:
        """Fraction of recent steps (raw, pre-filter) that switched regime — a measure of
        how unstable the signal currently is."""
        hist = list(self._raw_history)[-self.flicker_window:]
        if len(hist) < 2:
            return 0.0
        switches = sum(1 for a, b in zip(hist, hist[1:]) if a != b)
        return switches / (len(hist) - 1)

    def is_flickering(self, threshold: float = 0.30) -> bool:
        """True when the raw regime signal is switching too often to trust a hard regime."""
        return self.get_regime_flicker_rate() > threshold

    def regime_metadata(self) -> list[RegimeMetadata]:
        """Per-regime descriptive stats from the fitted model + recent history."""
        if self._model is None:
            return []
        hist = list(self._confirmed)
        n = len(hist) or 1
        out = []
        for state in range(self._n_states):
            reg = self._state_to_regime.get(state, Regime.NEUTRAL)
            out.append(RegimeMetadata(
                regime=reg,
                mean_return=float(self._model.means_[state, 0]) * 252 / self.ret_window,
                mean_vol=float(self._model.means_[state, 1]),
                persistence=float(self._model.transmat_[state, state]),
                frequency=hist.count(reg) / n,
            ))
        return out
