"""Runtime settings, loaded from environment / .env. No secrets are hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
POLICY_DIR = DATA_DIR / "policies"
DB_PATH = DATA_DIR / "shop.db"

load_dotenv(ROOT / ".env")


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    extra_body: str = field(default_factory=lambda: os.getenv("DEEPSEEK_EXTRA_BODY", ""))
    max_steps: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_STEPS", 8)))
    timeout_s: float = field(default_factory=lambda: _f("AGENT_TIMEOUT_S", 90))
    today: date = field(default_factory=lambda: date.fromisoformat(os.getenv("SIM_TODAY", "2026-11-18")))
    # CNY per 1M tokens
    price_in_hit: float = field(default_factory=lambda: _f("PRICE_INPUT_CACHE_HIT", 0.04))
    price_in_miss: float = field(default_factory=lambda: _f("PRICE_INPUT_CACHE_MISS", 2.0))
    price_out: float = field(default_factory=lambda: _f("PRICE_OUTPUT", 8.0))

    def cost_cny(self, prompt_tokens: int, completion_tokens: int, cache_hit_tokens: int = 0) -> float:
        miss = max(prompt_tokens - cache_hit_tokens, 0)
        return (cache_hit_tokens * self.price_in_hit + miss * self.price_in_miss
                + completion_tokens * self.price_out) / 1_000_000


def get_settings() -> Settings:
    return Settings()
