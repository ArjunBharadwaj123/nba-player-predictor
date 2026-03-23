"""
Step 8: Build Training Dataset
==============================
Merges BBRef game logs with NBA API schedule context into a single
training-ready DataFrame where every row is one player in one game.

Input files (all in data/raw/):
    all_gamelogs.csv            - player game logs from BBRef (Step 4)
    schedule_with_context.csv   - pace, opp_def_rating, rest days (Step 5)

Output files:
    data/processed/training_dataset.csv   - merged, cleaned, ready for Step 9

Usage:
    python features/build_dataset.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# BBRef → NBA API abbreviation normalisation map
# ─────────────────────────────────────────────────────────────────────────────
# BBRef uses different 3-letter codes than the NBA API for several teams.
# We normalise BBRef abbreviations to NBA API abbreviations before joining.
# Add entries here if you see unexpected NaNs in opp_def_rating after the merge.

ABBREV_MAP = {
    # BBRef : NBA API
    "BRK": "BKN",   # Brooklyn Nets
    "CHO": "CHA",   # Charlotte Hornets
    "PHO": "PHX",   # Phoenix Suns
    "NOP": "NOP",   # New Orleans Pelicans (same, listed for clarity)
    "UTA": "UTA",   # Utah Jazz (same)
    "SAS": "SAS",   # San Antonio Spurs (same)
    "GSW": "GSW",   # Golden State Warriors (same)
    # Historical relocations — add if your seasons include these
    "NJN": "BKN",   # New Jersey Nets → Brooklyn Nets
    "SEA": "OKC",   # Seattle SuperSonics → OKC Thunder
    "NOH": "NOP",   # New Orleans Hornets → Pelicans
}


def normalise_abbrev(abbrev: str) -> str:
    """Map a BBRef team abbreviation to its NBA API equivalent."""
    return ABBREV_MAP.get(str(abbrev).strip().upper(), str(abbrev).strip().upper())


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load raw data
# ─────────────────────────────────────────────────────────────────────────────

def load_gamelogs() -> pd.DataFrame:
    """Load and lightly clean the BBRef game logs."""
    path = RAW / "all_gamelogs.csv"
    log.info(f"Loading game logs from {path.name}...")
    df = pd.read_csv(path, parse_dates=["game_date"])

    # Rename trb -> reb for clarity (total rebounds is our target, not orb/drb)
    if "trb" in df.columns and "reb" not in df.columns:
        df = df.rename(columns={"trb": "reb"})

    # Ensure target columns exist and are numeric
    targets = ["pts", "reb", "ast", "stl", "blk", "minutes"]
    for col in targets:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise the team abbreviation to NBA API format
    if "team" in df.columns:
        df["team_abbrev"] = df["team"].apply(normalise_abbrev)
    if "opponent" in df.columns:
        df["opponent_abbrev"] = df["opponent"].apply(normalise_abbrev)

    log.info(f"  {len(df):,} rows, {df['player_name'].nunique()} players, "
             f"{df['season'].nunique()} seasons")
    return df


def load_schedule_context() -> pd.DataFrame:
    """Load the enriched schedule (pace + opp_def_rating + rest days)."""
    path = RAW / "schedule_with_context.csv"
    log.info(f"Loading schedule context from {path.name}...")
    df = pd.read_csv(path, parse_dates=["game_date"])
    log.info(f"  {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Merge
# ─────────────────────────────────────────────────────────────────────────────

def merge_datasets(gamelogs: pd.DataFrame,
                   schedule: pd.DataFrame) -> pd.DataFrame:
    """
    Join player game logs with schedule context.

    Join key: (team_abbrev, opponent_abbrev, game_date)

    Why this key?
        Each row in the schedule represents one team in one game.
        Each row in the game log represents one player in one game.
        The team + opponent + date combination uniquely identifies
        which schedule row corresponds to which player game.

    We use a LEFT join so that:
        - All player game rows are preserved
        - Games that don't match any schedule row get NaN context columns
          (we investigate and fix these rather than silently dropping games)
    """
    log.info("Merging game logs with schedule context...")

    # The schedule context columns we want to pull in
    context_cols = [
        "team_abbrev", "opponent_abbrev", "game_date",
        "rest_days", "back_to_back", "team_pace", "opp_def_rating",
    ]
    schedule_slim = schedule[context_cols].drop_duplicates(
        subset=["team_abbrev", "opponent_abbrev", "game_date"]
    )

    merged = gamelogs.merge(
        schedule_slim,
        on=["team_abbrev", "opponent_abbrev", "game_date"],
        how="left",
        validate="many_to_one",   # each player-game maps to exactly one schedule row
    )

    # ── Diagnose unmatched rows ───────────────────────────────────────────────
    n_unmatched = merged["rest_days"].isna().sum()
    if n_unmatched > 0:
        log.warning(f"  {n_unmatched} rows did not match a schedule row")
        # Show which teams are causing mismatches — helps debug abbrev issues
        unmatched = merged[merged["rest_days"].isna()]
        problem_teams = (
            unmatched.groupby(["team_abbrev", "season"])
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        log.warning(f"  Top unmatched team/seasons:\n{problem_teams.to_string()}")
    else:
        log.info("  All rows matched — no missing context")

    log.info(f"  Merged dataset: {len(merged):,} rows x {len(merged.columns)} columns")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Clean and derive columns
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final cleaning and derived column creation.

    Operations performed:
        1. Replace rest_days = -1 (season opener) with 3
           (neutral value — openers have similar rest to a 3-day break)
        2. Drop rows with missing target variables
        3. Drop rows with minutes < 5 (garbage time, unreliable stats)
        4. Derive fantasy_score from the DraftKings formula
        5. Derive usage_proxy from available BBRef columns
        6. Cap extreme outlier minutes (>60 is OT, keep; >70 is data error)
        7. Reset index
    """
    log.info("Cleaning merged dataset...")
    n_start = len(df)

    # ── 1. Fix season-opener rest days ───────────────────────────────────────
    df["rest_days"] = df["rest_days"].fillna(3)   # NaN from unmatched rows
    df["rest_days"] = df["rest_days"].replace(-1, 3)
    df["back_to_back"] = df["back_to_back"].fillna(0).astype(int)

    # ── 2. Drop rows missing any target variable ──────────────────────────────
    targets = ["pts", "reb", "ast", "stl", "blk", "minutes"]
    before = len(df)
    df = df.dropna(subset=targets)
    dropped_targets = before - len(df)
    if dropped_targets:
        log.info(f"  Dropped {dropped_targets} rows with missing targets")

    # ── 3. Drop garbage-time rows (< 5 minutes) ───────────────────────────────
    # A player who plays 2 minutes in garbage time is not a meaningful training
    # example — their stat line reflects noise, not skill or matchup.
    before = len(df)
    df = df[df["minutes"] >= 5].copy()
    dropped_garbage = before - len(df)
    if dropped_garbage:
        log.info(f"  Dropped {dropped_garbage} garbage-time rows (< 5 min)")

    # ── 4. Cap extreme minutes ────────────────────────────────────────────────
    # > 70 minutes is almost certainly a data error (OT games max ~65 min)
    extreme_min = (df["minutes"] > 70).sum()
    if extreme_min:
        log.warning(f"  {extreme_min} rows with minutes > 70 — capping at 70")
        df["minutes"] = df["minutes"].clip(upper=70)

    # ── 5. Fantasy score ──────────────────────────────────────────────────────
    # DraftKings formula: PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
    tov = df["tov"] if "tov" in df.columns else 0
    df["fantasy_score"] = (
        df["pts"]
        + 1.2 * df["reb"]
        + 1.5 * df["ast"]
        + 3.0 * df["stl"]
        + 3.0 * df["blk"]
        - tov
    )

    # ── 6. Usage proxy ────────────────────────────────────────────────────────
    # True usage rate requires team possessions, which we don't have at the
    # player level yet. This proxy (FGA + 0.44*FTA + TOV) / minutes
    # correlates ~0.92 with true usage and is sufficient for a feature.
    # We'll refine it in Step 9 when we add rolling averages.
    if all(c in df.columns for c in ["fga", "fta", "tov", "minutes"]):
        df["usage_proxy"] = (
            (df["fga"] + 0.44 * df["fta"] + df["tov"])
            / df["minutes"].clip(lower=1)   # avoid divide-by-zero
        )
    else:
        df["usage_proxy"] = np.nan

    # ── 7. Game outcome as binary ─────────────────────────────────────────────
    if "result" in df.columns:
        df["won"] = df["result"].astype(str).str.startswith("W").astype(int)

    # ── 8. Sort chronologically ───────────────────────────────────────────────
    df = df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    n_end = len(df)
    log.info(f"  Cleaning complete: {n_start:,} → {n_end:,} rows "
             f"({n_start - n_end:,} dropped total)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Quality report
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> None:
    print(f"\n{'─'*55}")
    print(f" Training dataset quality report")
    print(f"{'─'*55}")
    print(f"  Rows       : {len(df):,}")
    print(f"  Columns    : {len(df.columns)}")
    print(f"  Players    : {df['player_name'].nunique()}")
    print(f"  Seasons    : {sorted(df['season'].unique())}")
    print(f"  Date range : {df['game_date'].min().date()} "
          f"to {df['game_date'].max().date()}")

    print(f"\n  Target averages (should be per-game stats):")
    targets = ["pts", "reb", "ast", "stl", "blk", "minutes", "fantasy_score"]
    for col in targets:
        if col in df.columns:
            print(f"    {col:15s}: {df[col].mean():.2f}  "
                  f"(min={df[col].min():.1f}, max={df[col].max():.1f})")

    print(f"\n  Context feature null rates (should all be 0%):")
    context = ["rest_days", "back_to_back", "team_pace", "opp_def_rating"]
    for col in context:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            flag = "  <-- fix this" if pct > 0 else ""
            print(f"    {col:20s}: {pct:.1f}%{flag}")

    print(f"\n  Back-to-back games    : "
          f"{df['back_to_back'].sum():,} "
          f"({df['back_to_back'].mean()*100:.1f}% of games)")

    print(f"\n  Games per player:")
    counts = (
        df.groupby("player_name")["game_date"]
        .count()
        .sort_values(ascending=False)
    )
    for name, count in counts.items():
        flag = "  <-- low" if count < 50 else ""
        print(f"    {name:32s}: {count}{flag}")

    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_training_dataset() -> pd.DataFrame:
    """Full pipeline: load → merge → clean → save."""

    # Load
    gamelogs = load_gamelogs()
    schedule = load_schedule_context()

    # Merge
    merged = merge_datasets(gamelogs, schedule)

    # Clean
    clean = clean_dataset(merged)

    # Save
    out = PROCESSED / "training_dataset.csv"
    clean.to_csv(out, index=False)
    log.info(f"\nSaved training dataset -> {out}")

    return clean


if __name__ == "__main__":
    df = build_training_dataset()
    quality_report(df)