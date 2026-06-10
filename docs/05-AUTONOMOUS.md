# Running autonomously (paper) for a month

The system is designed to run **once per trading day, after the close** — it is not a
continuous daemon (a daily-rebalanced strategy doesn't need one). One daily run does the
whole loop: detect regime → allocate → submit the rebalance to your **paper** account →
refresh the dashboard's account snapshot. Schedule that, leave it for a month, and check
the dashboard whenever you like.

## The daily routine

[`scripts/run_daily.sh`](../scripts/run_daily.sh) is the unit of work:

1. **Skips non-trading days** (weekends/holidays, via Alpaca's calendar).
2. `python -m quantlab.platform` — compute regime + allocation, write `data/system_state.json`.
3. `python -m quantlab.live --strategy v3 --submit` — submit the rebalance to the **paper** account.
4. `python -m quantlab.broker` — refresh `data/account_state.json` (real equity, positions, trades).

Everything is logged to `data/logs/daily-YYYY-MM-DD.log`. Env knobs: `STRATEGY=v3` (default),
`DRY_RUN=1` (compute + show the plan but **don't** submit — recommended for the first few days).

Try it once by hand first:

```bash
DRY_RUN=1 ./scripts/run_daily.sh      # full routine, no orders submitted
cat data/logs/daily-$(date +%Y-%m-%d).log
```

## Schedule it (macOS, launchd)

[`scripts/com.quantlab.daily.plist`](../scripts/com.quantlab.daily.plist) runs the routine
**every weekday at 16:15 local time**. Market-DAY orders submitted ~15 min after the US close
queue to fill at the **next** open — matching the backtest's "decide on close, trade at next
open" rule. (If you're not in US-Eastern, set the Hour to your local equivalent of ~4:15pm ET.)

```bash
cp scripts/com.quantlab.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.quantlab.daily.plist     # start
launchctl list | grep quantlab                                     # verify it's registered
launchctl unload ~/Library/LaunchAgents/com.quantlab.daily.plist   # stop
```

If your Mac is asleep at 16:15, launchd runs the job the next time it wakes. For a machine
that's often off, run it on an always-on box or skip days (the strategy is slow; a missed day
just means the rebalance happens the next run).

Prefer cron? `crontab -e` and add (adjust path/time):

```
15 16 * * 1-5  cd /Users/james/Desktop/Coding/AI_apps/quant-lab && DRY_RUN=0 ./scripts/run_daily.sh
```

## Safety net while unattended

Every run applies the **circuit breaker** ([`quantlab.live --max-drawdown 0.15`](../src/quantlab/live.py)):
if account equity ever falls 15% below its high-water mark, the routine flattens to cash on the
next run, independent of the model. It persists the high-water mark in `data/live_state.json`.

## Watch it

```bash
python -m streamlit run src/quantlab/dashboard/app.py
```

The **Live Paper Account** panel shows real equity, total/today P&L, open positions with
unrealized P&L, and the full trade history — refreshed every time the routine runs. The
backtest curve below it is clearly labelled as *simulated*, separate from your real account.

## Honest expectations for the month

- It's **paper money** — this is a plumbing and discipline test, not a profit test. A month is
  far too short to say anything statistical about the edge (you'd need years).
- Expect **index-like-or-slightly-below** returns with a smoother ride. Do not extrapolate a good
  or bad month either way — it's noise.
- Use the month to confirm the *operations* work: orders fill where expected, costs are close to
  model, the dashboard reflects reality, and you're comfortable with the drawdowns.
