"""Central configuration: paths, seeds, and model settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SQL_DIR = ROOT / "sql"
WAREHOUSE_DIR = ROOT / "warehouse"
DOCS_DIR = ROOT / "docs"
ARTIFACTS_DIR = WAREHOUSE_DIR / "artifacts"

SEED = int(os.getenv("CIQ_SEED", "42"))

# Synthetic data window
START_DATE = "2025-01-06"  # a Monday
N_WEEKS = 91               # ~21 months of history
POSTS_PER_WEEK = 64

# LLM settings (optional). Without a key every AI step falls back to a
# deterministic local implementation, so the project always runs end to end.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
LLM_MODEL = os.getenv("CIQ_LLM_MODEL", "claude-sonnet-5-5")
LLM_BATCH_SIZE = 25

# A post "breaks out" when its 14-day views reach this multiple of the
# brand x platform trailing average (previous 30 posts).
BREAKOUT_MULTIPLE = 2.0


def ensure_dirs() -> None:
    for d in (RAW_DIR, WAREHOUSE_DIR, DOCS_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
