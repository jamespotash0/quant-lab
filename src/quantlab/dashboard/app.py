"""QuantLab live system-state dashboard.

Reads the system-state JSON written by the platform's main loop and renders a
single-page Streamlit dashboard. Run with::

    .venv/bin/python -m streamlit run src/quantlab/dashboard/app.py

The expected JSON lives at ``data/system_state.json`` (relative to the repo
root). If it is missing, the dashboard degrades gracefully and offers a button
to write a demoable sample produced by :func:`make_sample_state`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import altair as alt
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# app.py -> dashboard -> quantlab -> src -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = REPO_ROOT / "data" / "system_state.json"


# --------------------------------------------------------------------------- #
# Sample state
# --------------------------------------------------------------------------- #
def make_sample_state() -> dict[str, Any]:
    """Return a valid sample system-state dict matching the consumed schema."""
    equity_curve = []
    base = 100_000.0
    for i in range(60):
        # Deterministic gentle wiggle so the demo curve looks plausible.
        base *= 1.0 + (((i * 37) % 11) - 5) / 1000.0
        day = pd.Timestamp("2026-01-02") + pd.Timedelta(days=i)
        equity_curve.append({"date": day.strftime("%Y-%m-%d"), "equity": round(base, 2)})

    return {
        "as_of": "2026-06-09",
        "regime": "Bear",
        "regime_proba": {
            "Crash": 0.0,
            "Bear": 0.99,
            "Neutral": 0.0,
            "Bull": 0.01,
            "Euphoria": 0.0,
        },
        "confidence": 0.99,
        "uncertainty": 0.01,
        "regime_stability": 0.945,
        "flicker_rate": 0.05,
        "is_flickering": False,
        "vol_rank": 0.42,
        "vol_bucket": "mid",
        "active_strategy": "MidVolCautiousStrategy",
        "blend": {
            "LowVolBullStrategy": 0.0,
            "MidVolCautiousStrategy": 1.0,
            "HighVolDefensiveStrategy": 0.0,
        },
        "gross_exposure": 0.85,
        "target_weights": {
            "XLK": 0.10,
            "EEM": 0.08,
            "TLT": 0.12,
            "GLD": 0.07,
            "XLP": 0.05,
        },
        "transition_matrix": {
            "index": ["Crash", "Bear", "Neutral", "Bull", "Euphoria"],
            "columns": ["Crash", "Bear", "Neutral", "Bull", "Euphoria"],
            "data": [
                [0.70, 0.25, 0.04, 0.01, 0.00],
                [0.05, 0.90, 0.04, 0.01, 0.00],
                [0.01, 0.10, 0.78, 0.10, 0.01],
                [0.00, 0.02, 0.05, 0.90, 0.03],
                [0.00, 0.00, 0.02, 0.18, 0.80],
            ],
        },
        "regime_metadata": [
            {"regime": "Crash", "mean_return": -0.45, "mean_vol": 0.55, "persistence": 0.70, "frequency": 0.04},
            {"regime": "Bear", "mean_return": -0.12, "mean_vol": 0.30, "persistence": 0.90, "frequency": 0.20},
            {"regime": "Neutral", "mean_return": 0.04, "mean_vol": 0.16, "persistence": 0.78, "frequency": 0.26},
            {"regime": "Bull", "mean_return": 0.17, "mean_vol": 0.13, "persistence": 0.90, "frequency": 0.40},
            {"regime": "Euphoria", "mean_return": 0.28, "mean_vol": 0.22, "persistence": 0.80, "frequency": 0.10},
        ],
        "equity_curve": equity_curve,
        "alerts": [
            {"level": "warn", "msg": "Regime changed to Bear"},
            {"level": "info", "msg": "Rebalanced into MidVolCautiousStrategy"},
        ],
    }


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_state(path: Path) -> dict[str, Any] | None:
    """Load the system-state JSON, or ``None`` if missing/unreadable."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:  # pragma: no cover - UI path
        st.error(f"Failed to parse {path}: {exc}")
        return None


def write_sample(path: Path) -> None:
    """Write the sample state to ``path`` (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(make_sample_state(), fh, indent=2)


# --------------------------------------------------------------------------- #
# Render helpers
# --------------------------------------------------------------------------- #
_ALERT_RENDERERS = {
    "error": st.error,
    "warn": st.warning,
    "warning": st.warning,
    "info": st.info,
}


def render_header(state: dict[str, Any]) -> None:
    st.subheader(f"As of {state.get('as_of', 'n/a')}")

    regime = state.get("regime", "Unknown")
    st.markdown(
        f"<div style='font-size:3rem;font-weight:700;line-height:1.1'>"
        f"Regime: {regime}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Confidence", f"{state.get('confidence', 0):.0%}")
    c2.metric("Uncertainty", f"{state.get('uncertainty', 0):.0%}")
    c3.metric("Vol bucket", str(state.get("vol_bucket", "n/a")))
    c4.metric("Vol rank", f"{state.get('vol_rank', 0):.2f}")
    c5.metric("Gross exposure", f"{state.get('gross_exposure', 0):.0%}")


def render_alerts(state: dict[str, Any]) -> None:
    alerts = state.get("alerts") or []
    st.subheader("Alerts")
    if not alerts:
        st.success("No active alerts.")
        return
    for alert in alerts:
        level = str(alert.get("level", "info")).lower()
        msg = alert.get("msg", "")
        renderer = _ALERT_RENDERERS.get(level, st.info)
        renderer(f"[{level.upper()}] {msg}")


def render_regime_panel(state: dict[str, Any]) -> None:
    st.subheader("Regime")
    proba = state.get("regime_proba") or {}
    if proba:
        proba_df = pd.DataFrame(
            {"probability": list(proba.values())}, index=list(proba.keys())
        )
        st.bar_chart(proba_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Regime stability", f"{state.get('regime_stability', 0):.2f}")
    c2.metric("Flicker rate", f"{state.get('flicker_rate', 0):.2f}")
    if state.get("is_flickering"):
        c3.error("FLICKERING")
    else:
        c3.success("STABLE")


def render_allocation_panel(state: dict[str, Any]) -> None:
    st.subheader("Allocation")
    left, right = st.columns(2)

    with left:
        st.caption("Strategy blend")
        blend = state.get("blend") or {}
        if blend:
            blend_df = pd.DataFrame(
                {"weight": list(blend.values())}, index=list(blend.keys())
            )
            st.bar_chart(blend_df)
        st.caption(f"Active strategy: **{state.get('active_strategy', 'n/a')}**")

    with right:
        st.caption("Target weights")
        weights = state.get("target_weights") or {}
        if weights:
            weights_df = (
                pd.DataFrame({"weight": weights})
                .sort_values("weight", ascending=False)
            )
            st.dataframe(
                weights_df.style.format({"weight": "{:.2%}"}),
                width='stretch',
            )
        else:
            st.info("No target weights.")


def render_sleeve_decomposition(state: dict[str, Any]) -> None:
    """Show where each position's weight comes from across the three sub-strategy sleeves."""
    decomp = state.get("sleeve_decomposition") or []
    if not decomp:
        return
    st.subheader("Why these weights — sleeve decomposition")
    st.caption("Each holding's weight = the sum of its contributions from the three sleeves "
               "(DualMomentum ×0.50, VolatilityManaged ×0.25, SwingPivotBreakout ×0.25). "
               "A name held by more than one sleeve accumulates weight.")
    df = pd.DataFrame(decomp).set_index("symbol")
    pct = {c: "{:.1%}" for c in df.columns}
    st.dataframe(df.style.format(cast(Any, pct)).background_gradient(
        cmap="Greens", subset=["total"]), width="stretch")


def _heat_color(value: float, vmin: float, vmax: float) -> str:
    """Map a value to a CSS background color (light -> deep blue).

    Pure-Python gradient so the heatmap works without matplotlib.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    span = (vmax - vmin) or 1.0
    t = max(0.0, min(1.0, (v - vmin) / span))
    # Interpolate from near-white (245,247,251) to deep blue (8,48,107).
    r = int(245 + (8 - 245) * t)
    g = int(247 + (48 - 247) * t)
    b = int(251 + (107 - 251) * t)
    text = "white" if t > 0.55 else "black"
    return f"background-color: rgb({r},{g},{b}); color: {text};"


def render_transition_matrix(state: dict[str, Any]) -> None:
    st.subheader("Transition matrix (from -> to)")
    tm = state.get("transition_matrix")
    if not tm:
        st.info("No transition matrix available.")
        return
    matrix = pd.DataFrame(
        tm.get("data", []),
        index=tm.get("index", []),
        columns=tm.get("columns", []),
    )
    try:
        numeric = matrix.apply(pd.to_numeric, errors="coerce")
        vmin = float(numeric.min().min())
        vmax = float(numeric.max().max())
        styled = matrix.style.map(
            lambda v: _heat_color(cast(float, v), vmin, vmax)
        ).format("{:.2f}")
        st.dataframe(styled, width='stretch')
    except Exception:  # pragma: no cover - styling fallback
        st.dataframe(matrix, width='stretch')


def render_regime_metadata(state: dict[str, Any]) -> None:
    st.subheader("Regime metadata")
    meta = state.get("regime_metadata") or []
    if not meta:
        st.info("No regime metadata available.")
        return
    meta_df = pd.DataFrame(meta)
    fmt = {
        col: "{:.2%}"
        for col in ("mean_return", "mean_vol", "persistence", "frequency")
        if col in meta_df.columns
    }
    st.dataframe(meta_df.style.format(cast(Any, fmt)), width='stretch', hide_index=True)


def render_equity_curve(state: dict[str, Any]) -> None:
    st.subheader("Strategy backtest — simulated equity curve")
    st.caption("Simulated history of $100,000 invested in the strategy from the start of the "
               "backtest (2017) to today — NOT your live account. Your real account is in the "
               "Live Paper Account section above.")
    curve = state.get("equity_curve") or []
    if not curve:
        st.info("No equity curve available.")
        return
    curve_df = pd.DataFrame(curve)
    if "date" in curve_df.columns:
        curve_df["date"] = pd.to_datetime(curve_df["date"])
    curve_df = curve_df.rename(columns={"equity": "Strategy", "sp500": "S&P 500"})
    st.altair_chart(_equity_chart(curve_df), use_container_width=True)


def _equity_chart(curve_df: pd.DataFrame) -> "alt.LayerChart":
    """Multi-line equity chart with a shared crosshair tooltip that shows every
    series' value at the hovered date (Strategy + S&P 500)."""
    series = [c for c in curve_df.columns if c != "date"]
    long = curve_df.melt("date", value_vars=series, var_name="series", value_name="value")

    base = alt.Chart(long).encode(
        x=alt.X("date:T", title=None),
        color=alt.Color("series:N", title=None,
                        legend=alt.Legend(orient="top-left")),
    )
    lines = base.mark_line().encode(
        y=alt.Y("value:Q", title="Equity ($)", axis=alt.Axis(format="$,.0s")),
    )
    # Invisible vertical selector tracking the nearest date under the cursor.
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=["date"], empty=False)
    selectors = base.mark_point().encode(opacity=alt.value(0)).add_params(nearest)
    points = lines.mark_point(size=55).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
    )
    # One rule at the hovered date; pivot so a single tooltip lists both series.
    rule = (
        alt.Chart(long)
        .transform_pivot("series", value="value", groupby=["date"])
        .mark_rule(color="gray")
        .encode(
            x="date:T",
            opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Date")]
            + [alt.Tooltip(f"{s}:Q", title=s, format="$,.0f") for s in series],
        )
    )
    return cast(Any, alt.layer(lines, selectors, points, rule).interactive(bind_y=False))


# --------------------------------------------------------------------------- #
# Live (paper) account — REAL trades and P&L
# --------------------------------------------------------------------------- #
ACCOUNT_PATH = REPO_ROOT / "data" / "account_state.json"


def render_account() -> None:
    st.header("💼 Live Paper Account")
    acct = load_state(ACCOUNT_PATH)
    if acct is None:
        st.info("No live account snapshot yet. Run `python -m quantlab.broker` "
                "(or the daily routine) to fetch real equity, positions, and trades.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"${acct.get('equity', 0):,.2f}",
              f"{acct.get('today_pl', 0):+,.2f} today")
    c2.metric("Total P&L", f"${acct.get('total_pl', 0):+,.2f}",
              f"{acct.get('total_pl_pct', 0) * 100:+.2f}%")
    c3.metric("Cash", f"${acct.get('cash', 0):,.2f}")
    c4.metric("Open positions", str(acct.get("n_positions", 0)),
              f"${acct.get('unrealized_pl', 0):+,.2f} unrealized")

    positions = acct.get("positions") or []
    st.subheader("Open positions")
    if positions:
        pdf = pd.DataFrame(positions)
        pdf = pdf.rename(columns={"unrealized_pl": "P&L $", "unrealized_plpc": "P&L %"})
        if "P&L %" in pdf:
            pdf["P&L %"] = (pdf["P&L %"] * 100).round(2)
        st.dataframe(pdf, width="stretch", hide_index=True)
    else:
        st.caption("Flat — no open positions.")

    trades = acct.get("trades") or []
    st.subheader(f"Trade history ({len(trades)} fills)")
    if trades:
        st.dataframe(pd.DataFrame(trades), width="stretch", hide_index=True)
    else:
        st.caption("No trades filled yet. Run `python -m quantlab.live --submit` to trade.")

    curve = acct.get("equity_curve") or []
    if curve:
        st.subheader("Account equity (real)")
        cdf = pd.DataFrame(curve)
        cdf["date"] = pd.to_datetime(cdf["date"])
        st.line_chart(cdf.set_index("date"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_missing_state() -> None:
    st.warning(
        f"No system-state file found at `{STATE_PATH}`. "
        "The platform's main loop has not written one yet."
    )
    st.write(
        "You can write a sample state to make the dashboard demoable, then reload."
    )
    if st.button("Write sample system_state.json", type="primary"):
        write_sample(STATE_PATH)
        st.success(f"Wrote sample state to {STATE_PATH}. Reloading...")
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="QuantLab System State", layout="wide")
    st.title("QuantLab System State")

    if st.sidebar.button("Reload state"):
        st.rerun()
    st.sidebar.caption(f"State file:\n`{STATE_PATH}`")

    state = load_state(STATE_PATH)
    if state is None:
        render_missing_state()
        return

    render_header(state)
    st.divider()
    render_account()
    st.divider()
    render_alerts(state)
    st.divider()
    render_regime_panel(state)
    st.divider()
    render_allocation_panel(state)
    st.divider()
    render_sleeve_decomposition(state)
    st.divider()
    render_transition_matrix(state)
    st.divider()
    render_regime_metadata(state)
    st.divider()
    render_equity_curve(state)


if __name__ == "__main__":
    main()
