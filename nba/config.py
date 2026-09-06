"""
Central config for the entire project.
Import this in any module instead of repeating magic strings.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_SAVED  = ROOT / "models" / "saved"

# ── Scraping ──────────────────────────────────────────────────────────────────
BBREF_BASE    = "https://www.basketball-reference.com"
SCRAPE_DELAY  = 3.5          # seconds between requests (be polite to BBRef)

# ── Model ──────────────────────────────────────────────────────────────────────
TARGETS       = ["pts", "reb", "ast", "stl", "blk", "minutes"]
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

# ── Fantasy score formula ─────────────────────────────────────────────────────
FANTASY_WEIGHTS = {
    "pts": 1.0,
    "reb": 1.2,
    "ast": 1.5,
    "stl": 3.0,
    "blk": 3.0,
    "tov": -1.0,   # turnovers (negative)
}

def fantasy_score(pts, reb, ast, stl, blk, tov=0.0) -> float:
    """Compute DraftKings-style fantasy score from predicted stat lines."""
    return (
        pts  * FANTASY_WEIGHTS["pts"]
        + reb  * FANTASY_WEIGHTS["reb"]
        + ast  * FANTASY_WEIGHTS["ast"]
        + stl  * FANTASY_WEIGHTS["stl"]
        + blk  * FANTASY_WEIGHTS["blk"]
        + tov  * FANTASY_WEIGHTS["tov"]
    )
