"""Paper-trade executor: turn a strategy's target book into Alpaca orders.

This is the bridge from research to a live (paper) account. It does NOT run continuously —
a daily-rebalanced strategy only needs to act once a day. You run it once (manually or from
a scheduler) after the close; it:

  1. loads recent bars and computes the strategy's target weights as of the latest close,
  2. reads your Alpaca account equity and current positions,
  3. diffs target vs current to produce a rebalance plan (what to buy / sell), and
  4. prints that plan — and, only with ``--submit``, sends the orders to your PAPER account.

Orders are notional market orders (fractional shares) with time-in-force DAY, so if you run
this after the close they queue and fill at the next open — matching the backtest's
"decide on the close, trade at the next open" assumption.

    python -m quantlab.live                 # dry run, recommended strategy (V3)
    python -m quantlab.live --strategy v2    # a different strategy
    python -m quantlab.live --submit         # actually place the orders (paper account)

Long-only strategies only: this executor does not place short orders.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, cast

import pandas as pd

from .config import DATA_DIR, alpaca_config
from .data_pipeline.loaders import load_bars, to_panel
from .rationale import TRADE_LOG_PATH, explain_book, log_rationale
from .strategies.base import History, Strategy
from .strategies.research.dual_momentum import DualMomentum
from .strategies.research.recommended import SwingMomentumV2, SwingMomentumV3
from .universe import BENCHMARK, universe

#: Selectable strategies for live trading (long-only only).
STRATEGIES: dict[str, Strategy] = {
    "v3": SwingMomentumV3(),
    "v2": SwingMomentumV2(),
    "dm": DualMomentum(),
}

#: Skip rebalancing trades smaller than this (dollars) — avoids dust orders.
MIN_TRADE_USD = 25.0

#: Where the broker-level safety net persists its equity high-water mark between runs.
_STATE_PATH = DATA_DIR / "live_state.json"


def check_safety_net(equity: float, max_drawdown: float) -> tuple[bool, float, float]:
    """Independent, model-blind kill switch run at the broker. Tracks the account's
    all-time-high equity in a state file; if the current drawdown breaches ``max_drawdown``
    it reports tripped=True so the caller flattens to cash. Returns (tripped, peak, dd)."""
    peak = equity
    if _STATE_PATH.exists():
        try:
            peak = max(equity, float(json.loads(_STATE_PATH.read_text()).get("peak", equity)))
        except (ValueError, OSError):
            peak = equity
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps({"peak": peak}))
    drawdown = equity / peak - 1.0 if peak > 0 else 0.0
    return (max_drawdown > 0 and drawdown <= -max_drawdown), peak, drawdown


def target_book(strategy: Strategy) -> tuple[dict[str, float], str, pd.DataFrame]:
    """Compute the strategy's target weights as of the most recent available close.
    Returns (weights, as_of_date, close_panel). Pulls ~3y of history for warmup."""
    start = (pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    syms = sorted(set(universe()) | {BENCHMARK})
    bars = load_bars(syms, start, use_cache=True)
    close = to_panel(bars, "close")
    open_ = to_panel(bars, "open")
    as_of = close.index.max()
    weights = strategy.target_weights(History(as_of=as_of, close=close, open=open_))
    return weights, str(as_of.date()), close


def latest_prices(symbols: list[str]) -> dict[str, float]:
    """Most recent trade price per symbol, including pre-market/extended hours. Returns
    {symbol: price}; missing symbols are omitted. Used to measure the overnight gap vs the
    last close so a rebalance can see how the market moved before the open."""
    cfg = alpaca_config()
    if not cfg.is_configured or not symbols:
        return {}
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestTradeRequest

        client = StockHistoricalDataClient(cfg.api_key, cfg.secret_key)
        trades = client.get_stock_latest_trade(
            StockLatestTradeRequest(symbol_or_symbols=symbols))
        return {s: float(t.price) for s, t in trades.items()}  # type: ignore[union-attr]
    except Exception:
        return {}


def _client():
    cfg = alpaca_config()
    if not cfg.is_configured:
        raise RuntimeError("Alpaca keys not configured — fill in .env (see .env.example).")
    from alpaca.trading.client import TradingClient

    return TradingClient(cfg.api_key, cfg.secret_key, paper=True)


def rebalance_plan(weights: dict[str, float], equity: float,
                   current_value: dict[str, float]) -> list[dict]:
    """Diff target dollar exposure against current holdings. Returns a list of trades
    {symbol, side, notional, target_$, current_$}."""
    symbols = set(weights) | set(current_value)
    plan = []
    for sym in sorted(symbols):
        tgt = max(0.0, weights.get(sym, 0.0)) * equity
        cur = current_value.get(sym, 0.0)
        delta = tgt - cur
        if abs(delta) < MIN_TRADE_USD:
            continue
        plan.append({
            "symbol": sym,
            "side": "buy" if delta > 0 else "sell",
            "notional": round(abs(delta), 2),
            "target": round(tgt, 2),
            "current": round(cur, 2),
        })
    return plan


def submit(client, plan: list[dict]) -> None:
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    for t in plan:
        side = OrderSide.BUY if t["side"] == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=t["symbol"], notional=t["notional"], side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        print(f"  submitted {t['side']:4} ${t['notional']:>9,.2f} {t['symbol']:5} "
              f"(id {str(order.id)[:8]})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Paper-trade a strategy's target book on Alpaca.")
    p.add_argument("--strategy", default="v3", choices=sorted(STRATEGIES), help="which strategy")
    p.add_argument("--submit", action="store_true", help="actually place orders (paper account)")
    p.add_argument("--max-drawdown", type=float, default=0.15,
                   help="broker safety net: flatten to cash if equity drawdown exceeds this "
                        "(0 disables). Independent of the strategy.")
    p.add_argument("--gap-guard", type=float, default=0.0,
                   help="skip BUYS on names that gapped up more than this fraction overnight "
                        "(e.g. 0.04 = 4%%) to avoid chasing a gap. 0 disables.")
    args = p.parse_args(argv)

    strat = STRATEGIES[args.strategy]
    weights, as_of, close = target_book(strat)

    client = _client()
    # Alpaca SDK returns broad union types (RawData | model); narrow for the type checker.
    acct = cast(Any, client.get_account())
    equity = float(acct.equity)
    positions = cast(list, client.get_all_positions())
    current_value = {pos.symbol: float(pos.market_value) for pos in positions}

    # Safety net runs FIRST and overrides the strategy: a breached drawdown limit flattens
    # the book regardless of what the model wants.
    tripped, peak, dd = check_safety_net(equity, args.max_drawdown)
    if tripped:
        weights = {}

    # Per-trade rationale (signals behind each holding) — printed and logged for audit.
    book = explain_book(weights, close, benchmark=BENCHMARK)
    log_rationale(book, as_of, strat.name)

    # Pre-market / overnight: compare the latest (extended-hours) price to the last close
    # so we see how the market moved before we trade. Optionally guard against chasing gaps.
    last_close = close.iloc[-1]
    latest = latest_prices([r["symbol"] for r in book])
    gaps = {s: (latest[s] / float(last_close.get(s, latest[s])) - 1.0)
            for s in latest if s in last_close.index}
    guarded: set[str] = set()
    if args.gap_guard > 0:
        for s, g in gaps.items():
            if g > args.gap_guard and weights.get(s, 0) > current_value.get(s, 0) / equity:
                guarded.add(s)
        for s in guarded:
            weights[s] = current_value.get(s, 0.0) / equity  # hold current, don't add

    plan = rebalance_plan(weights, equity, current_value)

    print(f"Strategy: {strat.name}   as-of close: {as_of}")
    print(f"Account:  equity ${equity:,.2f}   positions held: {len(positions)}")
    print(f"Safety net: peak ${peak:,.0f}  drawdown {dd:+.1%}  limit -{args.max_drawdown:.0%}  "
          f"-> {'TRIPPED — flattening to cash' if tripped else 'ok'}\n")

    print("TARGET BOOK & RATIONALE:")
    for r in book:
        s = r["symbol"]
        gap = gaps.get(s)
        gap_str = f"  | overnight {gap:+.1%}{' [GAP-GUARD: not adding]' if s in guarded else ''}" \
            if gap is not None else ""
        print(f"  {s:5} {r['weight']:6.1%}  (${r['weight']*equity:,.0f}){gap_str}")
        print(f"        ↳ {r['reason']}")

    print(f"\nREBALANCE PLAN ({len(plan)} orders, min ${MIN_TRADE_USD:.0f}):")
    if not plan:
        print("  (already aligned — nothing to do)")
    for t in plan:
        print(f"  {t['side']:4} ${t['notional']:>9,.2f} {t['symbol']:5}  "
              f"(${t['current']:,.0f} -> ${t['target']:,.0f})")
    print(f"\n(rationale logged to {TRADE_LOG_PATH})")

    if args.submit:
        if not plan:
            return 0
        print("\nSubmitting to PAPER account...")
        submit(client, plan)
        print("Done. Market orders queue for the next open.")
    else:
        print("\nDRY RUN — re-run with --submit to place these orders on your paper account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
