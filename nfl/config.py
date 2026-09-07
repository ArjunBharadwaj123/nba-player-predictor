"""
Central configuration for the NFL predictor package.
=====================================================
Kept deliberately isolated from the NBA project's ``config.py`` so the two
sports never import each other. Anything sport-wide (paths, seasons, supported
positions, canonical targets, position normalization) lives here.

Environment overrides
---------------------
- ``NFL_SEASONS``   : comma list, e.g. "2021,2022,2023,2024" (overrides range)
- ``NFL_DEV_MODE``  : "1" to use the smaller DEV_SEASONS range by default
- ``NFL_START_SEASON`` / ``NFL_END_SEASON`` : integer season bounds
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).parent            # .../nfl
DATA_RAW       = ROOT / "data" / "raw"
DATA_CACHE     = ROOT / "data" / "cache"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_SAVED   = ROOT / "models" / "saved"

for _p in (DATA_RAW, DATA_CACHE, DATA_PROCESSED, MODELS_SAVED):
    _p.mkdir(parents=True, exist_ok=True)

# ── Seasons ────────────────────────────────────────────────────────────────────
# Production default: 2018 through the latest completed season. The exact
# "latest" is resolved at runtime from the schedule, but we keep an upper bound
# here so a bad env var can't ask for the future.
PROD_START_SEASON = 2018
LATEST_SEASON     = 2024          # bump as new seasons complete / are cached

# Development mode trains on fewer seasons so the full workflow runs quickly and
# within container memory. Artifacts built this way are clearly labelled
# (see models/train.py -> training metadata "mode": "dev").
DEV_SEASONS = [2022, 2023, 2024]


def _env_seasons() -> list[int] | None:
    raw = os.environ.get("NFL_SEASONS", "").strip()
    if not raw:
        return None
    try:
        return sorted({int(s) for s in raw.split(",") if s.strip()})
    except ValueError:
        return None


def is_dev_mode() -> bool:
    return os.environ.get("NFL_DEV_MODE", "0") == "1"


def default_seasons() -> list[int]:
    """Resolve the season range to use.

    Priority: explicit NFL_SEASONS env > dev mode > start/end env > prod range.
    """
    env = _env_seasons()
    if env:
        return env
    if is_dev_mode():
        return list(DEV_SEASONS)
    start = int(os.environ.get("NFL_START_SEASON", PROD_START_SEASON))
    end   = int(os.environ.get("NFL_END_SEASON", LATEST_SEASON))
    return list(range(start, end + 1))


# ── Supported positions ────────────────────────────────────────────────────────
SUPPORTED_POSITIONS = ["QB", "RB", "WR", "TE", "K"]

# Explicit normalization for unusual / historical labels nflverse emits. We map
# to a supported position ONLY when the mapping is unambiguous by role — never
# guess from a player's statline. Anything not covered returns None -> unsupported.
POSITION_NORMALIZATION = {
    "QB": "QB",
    "RB": "RB", "HB": "RB", "FB": "RB", "TB": "RB",   # halfback/fullback -> RB group
    "WR": "WR", "SE": "WR", "FL": "WR",               # split end / flanker -> WR
    "TE": "TE",
    "K": "K", "PK": "K",                              # placekicker -> K
    # position_group values (nflverse) also normalize cleanly:
    "QUARTERBACK": "QB", "RUNNING_BACK": "RB",
    "WIDE_RECEIVER": "WR", "TIGHT_END": "TE",
    "SPECIALIST": None,        # ambiguous (K/P/LS) — resolve via `position`, not group
}

# Positions we explicitly do NOT support in v1 (documented, returns clear message).
UNSUPPORTED_EXPLICIT = {
    "P": "Punters are not supported.",
    "LS": "Long snappers are not supported.",
    "OL": "Offensive linemen are not supported.",
    "C": "Offensive linemen are not supported.",
    "G": "Offensive linemen are not supported.",
    "T": "Offensive linemen are not supported.",
    "OT": "Offensive linemen are not supported.",
    "OG": "Offensive linemen are not supported.",
    "DL": "Individual defensive players are not supported.",
    "DE": "Individual defensive players are not supported.",
    "DT": "Individual defensive players are not supported.",
    "LB": "Individual defensive players are not supported.",
    "ILB": "Individual defensive players are not supported.",
    "OLB": "Individual defensive players are not supported.",
    "MLB": "Individual defensive players are not supported.",
    "CB": "Individual defensive players are not supported.",
    "S": "Individual defensive players are not supported.",
    "SS": "Individual defensive players are not supported.",
    "FS": "Individual defensive players are not supported.",
    "DB": "Individual defensive players are not supported.",
    "DST": "Team defense / special teams is not supported as a player.",
    "DEF": "Team defense / special teams is not supported as a player.",
}


def normalize_position(position: str | None,
                       position_group: str | None = None) -> str | None:
    """Return a supported position (QB/RB/WR/TE/K) or None.

    Uses the canonical nflverse ``position`` first, falling back to
    ``position_group``. Never infers a position from statistics.
    """
    for candidate in (position, position_group):
        if not candidate:
            continue
        key = str(candidate).strip().upper()
        if key in POSITION_NORMALIZATION:
            mapped = POSITION_NORMALIZATION[key]
            if mapped is not None:
                return mapped
    return None


def position_support_message(position: str | None,
                             position_group: str | None = None) -> str:
    """Human-readable reason a position isn't supported (for API 4xx messages)."""
    key = str(position or "").strip().upper()
    if key in UNSUPPORTED_EXPLICIT:
        return UNSUPPORTED_EXPLICIT[key]
    return (
        f"Position '{position or 'unknown'}' is not supported. "
        f"Supported positions: {', '.join(SUPPORTED_POSITIONS)}."
    )


# ── Canonical prediction targets per position ──────────────────────────────────
# These are the canonical names the API and models use. Where nflverse uses a
# different field name, feature_config/build_dataset maps to these (documented in
# COLUMN_ALIASES below).
POSITION_TARGETS: dict[str, list[str]] = {
    "QB": [
        "attempts", "completions", "passing_yards", "passing_tds",
        "passing_interceptions", "rushing_yards", "rushing_tds",
    ],
    "RB": [
        "carries", "rushing_yards", "rushing_tds",
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ],
    "WR": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "rushing_yards",
    ],
    "TE": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
    ],
    "K": [
        "field_goals_made", "extra_points_made", "kicking_points",
    ],
}

# Canonical target name -> nflverse source column in load_player_stats().
# When the two names match, the entry is identity (kept explicit for clarity /
# schema validation). Targets NOT here are derived (see build_dataset.py):
#   - kicking_points        : derived from distance-bucketed FG + PAT (scoring.py)
#   - field_goals_made      : nflverse "fg_made"
#   - extra_points_made     : nflverse "pat_made"
COLUMN_ALIASES: dict[str, str] = {
    "attempts": "attempts",
    "completions": "completions",
    "passing_yards": "passing_yards",
    "passing_tds": "passing_tds",
    "passing_interceptions": "passing_interceptions",
    "rushing_yards": "rushing_yards",
    "rushing_tds": "rushing_tds",
    "carries": "carries",
    "targets": "targets",
    "receptions": "receptions",
    "receiving_yards": "receiving_yards",
    "receiving_tds": "receiving_tds",
    "field_goals_made": "fg_made",
    "extra_points_made": "pat_made",
    # kicking_points has no direct source column -> derived in build_dataset.
    "kicking_points": None,
    "fumbles_lost": "fumbles_lost_total",
}

# Non-negative COUNT targets get a Poisson objective; continuous yardage gets
# squared-error. (evaluate/train read this.)
COUNT_TARGETS = {
    "attempts", "completions", "passing_tds", "passing_interceptions",
    "rushing_tds", "carries", "targets", "receptions", "receiving_tds",
    "field_goals_made", "extra_points_made",
}
# Targets that must never go negative when clamping predictions. (All NFL box-
# score volume/production stats qualify — none can be negative.)
NONNEGATIVE_TARGETS = set().union(
    COUNT_TARGETS,
    {"passing_yards", "rushing_yards", "receiving_yards", "kicking_points"},
)

# Quantile levels for prediction intervals (~70% central band).
QUANTILE_LOW  = 0.15
QUANTILE_HIGH = 0.85

# Recency weighting half-life (days) for training rows. Newer games count more.
RECENCY_HALFLIFE_DAYS = 540

# Regular-season only for training.
REGULAR_SEASON_TYPE = "REG"


def all_targets() -> list[str]:
    seen: list[str] = []
    for targets in POSITION_TARGETS.values():
        for t in targets:
            if t not in seen:
                seen.append(t)
    return seen
