"""The brain: a Hidden Markov Model that classifies the market regime.

A Gaussian HMM is fit on two features of the benchmark (SPY) — short-horizon return and
realized volatility — and learns a small number of hidden states. Markets genuinely behave
like a regime-switching process: long calm uptrends punctuated by short violent selloffs,
with persistent states (today's regime predicts tomorrow's). An HMM is the textbook model
for exactly that structure, and unlike a fixed moving-average rule it adapts its own
thresholds to the data.

The five learned states are labelled by their return/vol signature:

    CRASH   — sharply negative returns, very high volatility
    BEAR    — negative drift, elevated volatility
    NEUTRAL — flat, average volatility
    BULL    — positive drift, contained volatility
    EUPHORIA— strong positive drift, low volatility (often pre-reversal froth)

NO LOOKAHEAD — the hard part. The model is refit only on data up to the decision date, and
"today's" regime is the *filtered* state estimate: we run the forward pass over observations
up to and including today only, so no future bar can influence the classification. Refitting
every ``refit_days`` keeps it tractable; ``random_state`` is fixed so runs are reproducible.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from typing import cast

import numpy as np
import pandas as pd

# The EM fit routinely stops at the iteration cap rather than the convergence tolerance;
# that's fine for our purposes and the per-fit warnings are pure noise here.
logging.getLogger("hmmlearn").setLevel(logging.ERROR)


class Regime(IntEnum):
    CRASH = 0
    BEAR = 1
    NEUTRAL = 2
    BULL = 3
    EUPHORIA = 4

    @property
    def label(self) -> str:
        return self.name.capitalize()


def _features(prices: pd.Series, ret_window: int, vol_window: int) -> pd.DataFrame:
    """Two point-in-time features of the benchmark: trailing return and realized vol.
    Row t uses only prices up to t."""
    logp = cast(pd.Series, np.log(prices.astype(float)))
    ret = logp.diff(ret_window)
    daily = logp.diff()
    vol = daily.rolling(vol_window).std() * np.sqrt(252)
    return pd.concat([ret.rename("ret"), vol.rename("vol")], axis=1).dropna()


class RegimeBrain:
    """Fit a 5-state Gaussian HMM on the benchmark and classify the current regime with no
    lookahead. Stateful: caches the fitted model and refits every ``refit_days`` calls."""

    def __init__(
        self,
        regime_symbol: str = "SPY",
        n_states: int = 5,
        ret_window: int = 10,
        vol_window: int = 20,
        refit_days: int = 63,
        min_train: int = 504,
        predict_window: int = 252,
        random_state: int = 42,
    ) -> None:
        self.regime_symbol = regime_symbol
        self.n_states = n_states
        self.ret_window = ret_window
        self.vol_window = vol_window
        self.refit_days = refit_days
        self.min_train = min_train
        self.predict_window = predict_window
        self.random_state = random_state
        self._model = None
        self._state_to_regime: dict[int, Regime] = {}
        self._since_fit = 10**9  # force a fit on first eligible call

    def _fit(self, x: np.ndarray) -> None:
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=self.random_state,
        )
        model.fit(x)
        # Label states by mean return (annualized via the return feature): the lowest-return
        # state is CRASH, the highest is EUPHORIA. This makes state identity stable across
        # refits even though the HMM's internal state numbering is arbitrary.
        order = np.argsort(model.means_[:, 0])  # ascending by return-feature mean
        regimes = list(Regime)
        self._state_to_regime = {int(s): regimes[i] for i, s in enumerate(order)}
        self._model = model

    def classify(self, history) -> Regime:
        """Return the filtered regime as of the last row of ``history`` (no lookahead)."""
        prices = history.close[self.regime_symbol].dropna() \
            if self.regime_symbol in history.close.columns else pd.Series(dtype=float)
        feats = _features(prices, self.ret_window, self.vol_window)
        if len(feats) < self.min_train:
            return Regime.NEUTRAL

        x_all = feats.to_numpy()
        # Refit periodically on everything available up to today; reuse the model otherwise.
        if self._model is None or self._since_fit >= self.refit_days:
            try:
                self._fit(x_all)
                self._since_fit = 0
            except Exception:
                self._since_fit = 0
                if self._model is None:
                    return Regime.NEUTRAL
        else:
            self._since_fit += 1

        model = self._model
        if model is None:
            return Regime.NEUTRAL
        # Filtered estimate for today: run the model over a trailing window ending at today;
        # the LAST row's posterior uses no observations after today.
        window = x_all[-self.predict_window :]
        try:
            state = int(model.predict_proba(window)[-1].argmax())
        except Exception:
            return Regime.NEUTRAL
        return self._state_to_regime.get(state, Regime.NEUTRAL)
