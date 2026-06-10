"""Live (paper) account snapshot — the *real* broker state, for monitoring.

Distinct from the strategy backtest: this reads your actual Alpaca paper account and
reports real equity, profit/loss, open positions, and the trades that were actually
filled. The dashboard renders this alongside (and clearly separated from) the simulated
backtest, so "what the strategy did in history" is never confused with "what my account
is actually worth right now".

    python -m quantlab.broker        # fetch + write data/account_state.json + print summary
"""

from __future__ import annotations

import json
from typing import Any, cast

from .config import DATA_DIR, alpaca_config

#: Assumed starting capital for total-P&L reporting (Alpaca paper default).
STARTING_CAPITAL = 100_000.0

ACCOUNT_STATE_PATH = DATA_DIR / "account_state.json"


def is_trading_day() -> bool:
    """True if today is a US market trading day (per Alpaca's calendar). Used by the daily
    autonomous routine to skip weekends/holidays. Defaults to True if keys aren't set."""
    cfg = alpaca_config()
    if not cfg.is_configured:
        return True
    from datetime import datetime, timezone

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    client = TradingClient(cfg.api_key, cfg.secret_key, paper=True)
    today = datetime.now(timezone.utc).date()
    try:
        cal = cast(list, client.get_calendar(GetCalendarRequest(start=today, end=today)))
        return len(cal) > 0
    except Exception:
        return True


def account_snapshot() -> dict | None:
    """Pull the live paper account state from Alpaca. Returns None if keys aren't set."""
    cfg = alpaca_config()
    if not cfg.is_configured:
        return None

    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest, GetPortfolioHistoryRequest

    client = TradingClient(cfg.api_key, cfg.secret_key, paper=True)
    acct = cast(Any, client.get_account())
    equity = float(acct.equity)
    last_equity = float(acct.last_equity)
    cash = float(acct.cash)

    positions = []
    unrealized = 0.0
    for p in cast(list, client.get_all_positions()):
        upl = float(p.unrealized_pl)
        unrealized += upl
        positions.append({
            "symbol": p.symbol,
            "qty": round(float(p.qty), 4),
            "avg_entry": round(float(p.avg_entry_price), 2),
            "current_price": round(float(p.current_price), 2),
            "market_value": round(float(p.market_value), 2),
            "unrealized_pl": round(upl, 2),
            "unrealized_plpc": round(float(p.unrealized_plpc), 4),
        })
    positions.sort(key=lambda r: -r["market_value"])

    trades = []
    try:
        orders = cast(list, client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100)))
        for o in orders:
            if o.filled_at is None:
                continue
            trades.append({
                "time": str(o.filled_at)[:19],
                "side": o.side.value,
                "symbol": o.symbol,
                "qty": round(float(o.filled_qty), 4) if o.filled_qty else 0.0,
                "price": round(float(o.filled_avg_price), 2) if o.filled_avg_price else None,
                "notional": round(float(o.filled_qty or 0) * float(o.filled_avg_price or 0), 2),
            })
    except Exception:
        pass

    equity_curve = []
    try:
        ph = cast(Any, client.get_portfolio_history(
            GetPortfolioHistoryRequest(period="3M", timeframe="1D")))
        import datetime as _dt
        for ts, eq in zip(ph.timestamp or [], ph.equity or []):
            if eq:
                d = _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).date()
                equity_curve.append({"date": str(d), "equity": round(float(eq), 2)})
    except Exception:
        pass

    return {
        "equity": round(equity, 2),
        "cash": round(cash, 2),
        "last_equity": round(last_equity, 2),
        "starting_capital": STARTING_CAPITAL,
        "total_pl": round(equity - STARTING_CAPITAL, 2),
        "total_pl_pct": round(equity / STARTING_CAPITAL - 1.0, 4),
        "today_pl": round(equity - last_equity, 2),
        "today_pl_pct": round(equity / last_equity - 1.0, 4) if last_equity else 0.0,
        "unrealized_pl": round(unrealized, 2),
        "n_positions": len(positions),
        "positions": positions,
        "trades": trades,
        "equity_curve": equity_curve,
    }


def write_snapshot() -> dict | None:
    snap = account_snapshot()
    if snap is not None:
        ACCOUNT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ACCOUNT_STATE_PATH.write_text(json.dumps(snap, indent=2))
    return snap


def main() -> int:
    snap = write_snapshot()
    if snap is None:
        print("Alpaca keys not configured — cannot read account.")
        return 1
    print(f"Live paper account  ->  {ACCOUNT_STATE_PATH}")
    print(f"  equity        ${snap['equity']:,.2f}")
    print(f"  total P&L     ${snap['total_pl']:+,.2f} ({snap['total_pl_pct']:+.2%})")
    print(f"  today P&L     ${snap['today_pl']:+,.2f}")
    print(f"  cash          ${snap['cash']:,.2f}")
    print(f"  positions     {snap['n_positions']}  (unrealized P&L ${snap['unrealized_pl']:+,.2f})")
    print(f"  trades logged {len(snap['trades'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
