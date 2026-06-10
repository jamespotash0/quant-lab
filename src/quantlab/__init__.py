"""quant-lab: a research harness for backtesting a systematic daily-swing trading bot.

The design rule that matters most: the same Strategy object is consumed by both the
backtest engine and (eventually) the live runner, and every feature is computed
point-in-time so a backtest cannot see the future.
"""

__version__ = "0.1.0"
