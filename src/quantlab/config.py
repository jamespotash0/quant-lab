"""Central config: paths and secrets. Secrets come from a local .env (never committed)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BARS_CACHE_DIR = DATA_DIR / "bars"

# Load .env from the project root if present. Existing env vars win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str
    secret_key: str
    data_feed: str = "iex"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)


def alpaca_config() -> AlpacaConfig:
    return AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        data_feed=os.getenv("ALPACA_DATA_FEED", "iex"),
    )
