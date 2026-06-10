# Runbook — how to run the system from scratch

Everything below assumes you are **in the project folder** and using the **project's venv**
(there is no bare `python` on this machine — use `.venv/bin/python`, or `activate` first).

```bash
cd /Users/james/Desktop/Coding/AI_apps/quant-lab
```

## 0. One-time setup

**Python**: use 3.11 or **3.12** (3.13 was removed from this machine). Build the venv and
install everything (core libs + dev tools + dashboard):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev,dashboard]"
```

**Alpaca keys** (paper): already configured in `.env`. If you ever need to redo it:

```bash
cp .env.example .env          # then paste your paper Key ID + Secret into .env
```

**Verify the install:**

```bash
.venv/bin/python -m pytest -q                       # should print "6 passed"
.venv/bin/python -m pyright --pythonpath .venv/bin/python src tests   # "0 errors"
```

Optional convenience — activate the venv so plain `python` works for the session:

```bash
source .venv/bin/activate      # prompt shows (.venv); type `deactivate` when done
```

## 1. The daily loop (run these in order)

| Step | Command | What it does |
|---|---|---|
| **Decide** | `.venv/bin/python -m quantlab.platform` | computes regime + the V3 book + sleeve decomposition, writes `data/system_state.json` and refreshes the account snapshot. (~2–3 min — it walks the HMM.) |
| **Watch** | `.venv/bin/python -m streamlit run src/quantlab/dashboard/app.py` | opens the dashboard in your browser: live account (equity, P&L, trades), the V3 allocation, *why these weights* (sleeve decomposition), and the regime context. |
| **Preview trades** | `.venv/bin/python -m quantlab.live --strategy v3` | dry run — prints the exact orders + per-trade rationale + overnight gaps. No orders placed. |
| **Trade** | `.venv/bin/python -m quantlab.live --strategy v3 --submit` | places the rebalance on your **paper** account. |
| **Refresh account** | `.venv/bin/python -m quantlab.broker` | updates the dashboard's live-account panel (equity, positions, fills). |

Start by running **Decide → Watch → Preview** for a few days *without* `--submit`, so you can
see what it wants to do before any orders go in.

## 2. Run it autonomously for a month

One command does the whole loop (skip-if-closed → decide → submit → refresh), logging to
`data/logs/`:

```bash
DRY_RUN=1 ./scripts/run_daily.sh     # full routine, NO orders (recommended first)
./scripts/run_daily.sh               # the real thing (submits to paper)
```

Schedule it every weekday at 4:15pm ET (after the close; orders queue for the next open):

```bash
cp scripts/com.quantlab.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.quantlab.daily.plist     # start
launchctl list | grep quantlab                                     # verify
launchctl unload ~/Library/LaunchAgents/com.quantlab.daily.plist   # stop
```

Your Mac must be **awake at 4:15pm ET** on weekdays for it to fire (it catches up on the next
wake if asleep). Full detail: [docs/05-AUTONOMOUS.md](05-AUTONOMOUS.md).

## 3. Knobs

- **Strategy**: `--strategy v3` (recommended) / `v2` / `dm`.
- **Safety net**: `--max-drawdown 0.15` — flatten to cash if equity falls 15% below its peak (0 disables).
- **Gap guard**: `--gap-guard 0.03` — skip *adding* to a name that gapped up >3% overnight (default off; backtested as immaterial for V3).
- **Universe**: trade stocks instead of ETFs via `export QUANTLAB_UNIVERSE="AAPL,MSFT,NVDA"` or a `data/universe.txt` file (re-validate first — strategies were tuned on ETFs).

## 4. If something breaks

- **"command not found: python"** → use `.venv/bin/python` (no bare `python` exists here).
- **"no such file or directory: .venv/bin/python"** → the venv's interpreter was removed (e.g.
  Homebrew upgraded Python). Rebuild: `python3.12 -m venv .venv --clear && .venv/bin/python -m pip install -e ".[dev,dashboard]"`.
- **"Alpaca keys not configured"** → fill `.env` (see step 0).
- **Dashboard shows old numbers** → re-run `quantlab.platform` (or `quantlab.broker`) to refresh the state files, then hit "Reload state" in the sidebar.

## 5. Honest reminder
This is **paper money** and a **plumbing test**, not a profit test. Expect index-like-or-below
returns with a smoother ride; a month is far too short to judge the edge. See
[docs/01-STRATEGY-SWEEP.md](01-STRATEGY-SWEEP.md) and [docs/04-PLATFORM.md](04-PLATFORM.md) for the
full results and caveats.
