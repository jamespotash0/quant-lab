#!/usr/bin/env bash
#
# Autonomous daily trading routine (PAPER). Run once per trading day, after the close.
# It: (1) skips non-trading days, (2) computes the regime/allocation, (3) submits the
# rebalance to the Alpaca PAPER account, (4) refreshes the dashboard's account snapshot.
# Everything is logged to data/logs/daily-YYYY-MM-DD.log.
#
# Manual run:   ./scripts/run_daily.sh
# Scheduled:    see scripts/com.quantlab.daily.plist (launchd) or the cron line in docs/05-AUTONOMOUS.md
#
# Strategy is chosen via STRATEGY env var (default v3). Set DRY_RUN=1 to NOT submit orders.

set -uo pipefail

# Resolve repo root (this script lives in <repo>/scripts).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="$REPO/.venv/bin/python"
STRATEGY="${STRATEGY:-v3}"
LOG_DIR="$REPO/data/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/daily-$(date +%Y-%m-%d).log"

{
  echo "==================== $(date) ===================="

  # 1) Trading-day guard — skip weekends/holidays.
  if ! "$PY" -c "import sys; from quantlab.broker import is_trading_day; sys.exit(0 if is_trading_day() else 1)"; then
    echo "Not a trading day — skipping."
    exit 0
  fi

  # 2) Compute regime + allocation, write dashboard state.
  echo "--- compute state ---"
  "$PY" -m quantlab.platform || { echo "platform step FAILED"; exit 1; }

  # 3) Execute the rebalance on the paper account (unless DRY_RUN=1).
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "--- DRY RUN: showing plan, not submitting ---"
    "$PY" -m quantlab.live --strategy "$STRATEGY"
  else
    echo "--- execute trades (paper, --submit) ---"
    "$PY" -m quantlab.live --strategy "$STRATEGY" --submit || { echo "live step FAILED"; exit 1; }
  fi

  # 4) Refresh the live account snapshot for the dashboard.
  echo "--- refresh account snapshot ---"
  "$PY" -m quantlab.broker || echo "broker snapshot failed (non-fatal)"

  echo "Done $(date)."
} >> "$LOG" 2>&1

# Echo a one-liner to stdout too (useful when run manually).
tail -n 1 "$LOG"
