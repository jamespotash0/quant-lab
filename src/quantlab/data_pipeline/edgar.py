"""Point-in-time fundamentals from SEC EDGAR (free).

The single hardest part of backtesting fundamental factors is avoiding lookahead: a
company's fiscal-year-2024 earnings are NOT knowable on the 2024 fiscal year-end — they're
only knowable once the 10-K is *filed*, typically 4–8 weeks later. EDGAR's company-facts API
gives every reported figure together with the date it was ``filed``, so we can ask "what was
known about this company *as of* date D" without cheating.

This loader maps tickers to CIKs, pulls a company's XBRL facts (cached to disk), and exposes
an annual fundamentals table keyed by **filing date**. Combined with prices (yfinance), that
lets us build value/quality factors with correct point-in-time alignment.

Cache: data/edgar/<CIK>.json (raw company facts) + data/edgar/cik_map.json.
"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

from ..config import DATA_DIR

EDGAR_DIR = DATA_DIR / "edgar"
#: SEC asks for a descriptive User-Agent with contact info and caps requests at ~10/sec.
_UA = {"User-Agent": "quant-lab research james.potash0@gmail.com"}
_LAST_REQ = [0.0]


def _get(url: str) -> dict:
    # Be polite to SEC: keep under ~10 req/s.
    dt = time.monotonic() - _LAST_REQ[0]
    if dt < 0.12:
        time.sleep(0.12 - dt)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    _LAST_REQ[0] = time.monotonic()
    return data


def cik_map() -> dict[str, str]:
    """Ticker -> zero-padded 10-digit CIK, cached to disk."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    path = EDGAR_DIR / "cik_map.json"
    if path.exists():
        return json.loads(path.read_text())
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    out = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in raw.values()}
    path.write_text(json.dumps(out))
    return out


def company_facts(cik: str) -> dict:
    """Raw XBRL company facts for a CIK, cached to disk."""
    EDGAR_DIR.mkdir(parents=True, exist_ok=True)
    path = EDGAR_DIR / f"{cik}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
    path.write_text(json.dumps(data))
    return data


def _concept(facts: dict, names: list[str], unit: str = "USD") -> pd.DataFrame:
    """Pull the first available us-gaap/dei concept among ``names`` as a tidy frame with
    columns [end, filed, val]. Returns empty if none present."""
    for ns in ("us-gaap", "dei"):
        block = facts.get("facts", {}).get(ns, {})
        for name in names:
            if name in block:
                units = block[name].get("units", {})
                rows = units.get(unit) or next(iter(units.values()), [])
                df = pd.DataFrame(rows)
                if not df.empty and {"end", "filed", "val"} <= set(df.columns):
                    return df[["end", "filed", "val"] + (["form", "fp"] if "form" in df else [])]
    return pd.DataFrame(columns=["end", "filed", "val"])


def annual_fundamentals(ticker: str) -> pd.DataFrame:
    """Point-in-time annual fundamentals for ``ticker``: one row per 10-K figure, indexed by
    the date it was FILED (so a row is only usable from that date forward). Columns:
    net_income, equity, revenue, assets, shares. Returns empty if the company isn't found."""
    cmap = cik_map()
    cik = cmap.get(ticker.upper())
    if cik is None:
        return pd.DataFrame()
    facts = company_facts(cik)

    def annual(names: list[str], unit: str = "USD") -> pd.Series:
        df = _concept(facts, names, unit)
        if df.empty:
            return pd.Series(dtype=float)
        if "form" in df.columns:
            df = df[df["form"] == "10-K"]
        df = df.dropna(subset=["filed", "val"]).copy()
        df["filed"] = pd.to_datetime(df["filed"]).dt.tz_localize("UTC")
        # Keep the latest value reported on each filing date.
        return df.sort_values("filed").groupby("filed")["val"].last()

    cols = {
        "net_income": annual(["NetIncomeLoss"]),
        "equity": annual(["StockholdersEquity"]),
        "revenue": annual(["Revenues",
                           "RevenueFromContractWithCustomerExcludingAssessedTax"]),
        "assets": annual(["Assets"]),
        "shares": annual(["EntityCommonStockSharesOutstanding",
                          "CommonStockSharesOutstanding"], unit="shares"),
    }
    out = pd.DataFrame(cols).sort_index()
    out.index.name = "filed"
    return out.dropna(how="all")
