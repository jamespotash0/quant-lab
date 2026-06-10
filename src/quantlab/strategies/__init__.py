"""Strategies. The same Strategy object is consumed by the backtest engine and (later)
the live runner — there is no separate "backtest version" of the logic."""

from .base import History, Strategy
from .buy_and_hold import BuyAndHold
from .momentum import MomentumReversal

__all__ = ["Strategy", "History", "BuyAndHold", "MomentumReversal"]
