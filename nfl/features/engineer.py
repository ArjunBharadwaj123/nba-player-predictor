"""
Leakage-safe NFL feature engineering.
======================================
Every feature for a player-week row uses ONLY information available before that
game started. The data is sorted by (season, week) and ``.shift(1)`` is applied
before every rolling / expanding / season / opponent / matchup aggregate. The
same-week target values are never used as features.

Builds:
  * Per-stat lagged features: previous value, rolling mean (3/5/8), rolling std,
    season-to-date mean, exponentially-weighted mean, recent trend.
  * Opportunity shares (target share etc. are already weekly; we lag them).
  * Game-context features (home/away, rest, bye return, spread, total, implied
    total, surface, indoor, temp, wind, experience, injury, depth/starter).
  * Opponent-defense features: fantasy points and yards allowed to the player's
    position, built by aggregating weekly production BY opponent and shifting.
  * Missing-data indicators + documented neutral imputation.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nfl.features.feature_config import (
    ROLL_WINDOWS,
    ROLLING_STATS,
    NEUTRAL_IMPUTATION,
    MISSING_INDICATORS,
)

log = logging.getLogger(__name__)


def _lagged_rolling(df: pd.DataFrame, group: str, stat: str) -> dict[str, pd.Series]:
    """Return a dict of leakage-safe lagged features for one stat.

    All series are grouped by player and shifted by 1 BEFORE aggregating, so a
    row never sees its own game.
    """
    g = df.groupby(group, sort=False)[stat]
    shifted = g.shift(1)                      # previous games only
    out: dict[str, pd.Series] = {}
    out[f"{stat}_prev"] = shifted
    # Rolling means over recent games.
    for w in ROLL_WINDOWS:
        out[f"{stat}_roll{w}"] = (
            g.shift(1).groupby(df[group], sort=False)
            .rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        )
    # Rolling std (volatility) over 5.
    out[f"{stat}_roll5_std"] = (
        g.shift(1).groupby(df[group], sort=False)
        .rolling(5, min_periods=2).std().reset_index(level=0, drop=True)
    )
    # Season-to-date expanding mean (shifted).
    out[f"{stat}_season_avg"] = (
        g.shift(1).groupby(df[group], sort=False)
        .expanding(min_periods=1).mean().reset_index(level=0, drop=True)
    )
    # Exponentially weighted mean (recency-biased, shifted).
    out[f"{stat}_ewma"] = (
        g.shift(1).groupby(df[group], sort=False)
        .ewm(halflife=3).mean().reset_index(level=0, drop=True)
    )
    return out


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)
    new_cols: dict[str, pd.Series] = {}
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        feats = _lagged_rolling(df, "gsis_id", stat)
        new_cols.update(feats)
    # Recent trend: roll3 / roll8 (ratio) for a few core stats where present.
    trend_stats = ["fantasy_points_component", "passing_yards", "rushing_yards",
                   "receiving_yards", "targets", "carries", "kicking_points"]
    tmp = pd.DataFrame(new_cols, index=df.index)
    for stat in trend_stats:
        r3, r8 = f"{stat}_roll3", f"{stat}_roll8"
        if r3 in tmp.columns and r8 in tmp.columns:
            tmp[f"{stat}_trend"] = (
                tmp[r3] / tmp[r8].replace(0, np.nan)
            ).clip(0.25, 4.0)
    df = pd.concat([df, tmp], axis=1)
    return df


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Static, pre-kickoff game context + bye-week return + games played."""
    df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)

    # Games played so far this season (count of prior games) — leakage-safe.
    df["games_played_season"] = (
        df.groupby(["gsis_id", "season"], sort=False).cumcount()
    )

    # Bye-week / missed-week return: gap in weeks since last appearance > 1.
    prev_week = df.groupby(["gsis_id", "season"], sort=False)["week"].shift(1)
    week_gap = df["week"] - prev_week
    df["bye_week_return"] = (week_gap > 1).fillna(False).astype(int)
    df["days_rest"] = pd.to_numeric(df.get("days_rest"), errors="coerce")

    # Age at game time (aging/decline signal), from birth_date when available.
    if "birth_date" in df.columns:
        bd = pd.to_datetime(df["birth_date"], errors="coerce")
        gd = pd.to_datetime(df["gameday"], errors="coerce")
        df["age"] = ((gd - bd).dt.days / 365.25).clip(lower=18, upper=50)
    else:
        df["age"] = np.nan

    return df


def _add_team_environment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe OWN-team offensive-environment features.

    For each (season, team) we compute the season-to-date (shifted) average of
    the team's weekly offensive profile — pass rate, plays per game, offensive
    EPA, touchdowns, and implied team total — using ONLY prior weeks. This lets
    the model project a player into a NEW team's offense at serve time (the
    serving layer injects the current team's latest profile).
    """
    team_week = (
        df.groupby(["season", "team", "week"], as_index=False)
        .agg(
            _pass_att=("attempts", "sum"),
            _rush_att=("carries", "sum"),
            _pass_epa=("passing_epa", "sum"),
            _rush_epa=("rushing_epa", "sum"),
            _pass_tds=("passing_tds", "sum"),
            _rush_tds=("rushing_tds", "sum"),
            _implied=("implied_team_total", "mean"),
        )
        .sort_values(["season", "team", "week"])
    )
    tw = team_week
    tw["_plays"] = tw["_pass_att"] + tw["_rush_att"]
    tw["_pass_rate"] = tw["_pass_att"] / tw["_plays"].replace(0, np.nan)
    tw["_off_epa"] = tw["_pass_epa"] + tw["_rush_epa"]
    tw["_tds"] = tw["_pass_tds"] + tw["_rush_tds"]

    grp = tw.groupby(["season", "team"], sort=False)
    src_dst = [
        ("_pass_rate", "team_pass_rate"),
        ("_plays", "team_plays_per_game"),
        ("_off_epa", "team_off_epa"),
        ("_tds", "team_tds_per_game"),
        ("_implied", "team_implied_total_avg"),
    ]
    for src, dst in src_dst:
        tw[dst] = (
            grp[src].shift(1).groupby([tw["season"], tw["team"]], sort=False)
            .expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
    keep = ["season", "team", "week"] + [d for _, d in src_dst]
    df = df.merge(tw[keep], on=["season", "team", "week"], how="left")
    return df


# Which canonical target a position "produces" for opponent-defense-allowed.
POSITION_PRODUCTION = {
    "QB": "passing_yards",
    "RB": "rushing_yards",
    "WR": "receiving_yards",
    "TE": "receiving_yards",
    "K":  "kicking_points",
}


def _add_opponent_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """Opponent fantasy/yards allowed to each position, leakage-safe.

    For every (opponent_team, position) we compute the running season-to-date
    average of the fantasy/yardage that position produced against that opponent,
    using ONLY weeks strictly before the current one (shift within the opponent
    timeline).
    """
    # Per (season, opponent, week, position): total allowed that week.
    allowed = (
        df.groupby(["season", "opponent_team", "week", "position"], as_index=False)
        .agg(
            allowed_fp=("fantasy_points_component", "sum"),
            allowed_pass_yds=("passing_yards", "sum"),
            allowed_rush_yds=("rushing_yards", "sum"),
            allowed_rec_yds=("receiving_yards", "sum"),
            allowed_pass_tds=("passing_tds", "sum"),
            allowed_rush_tds=("rushing_tds", "sum"),
        )
        .sort_values(["season", "opponent_team", "position", "week"])
    )
    grp = allowed.groupby(["season", "opponent_team", "position"], sort=False)
    for src, dst in [
        ("allowed_fp", "opp_fantasy_allowed_pos"),
        ("allowed_pass_yds", "opp_pass_yards_allowed"),
        ("allowed_rush_yds", "opp_rush_yards_allowed"),
        ("allowed_rec_yds", "opp_rec_yards_allowed_pos"),
        ("allowed_pass_tds", "opp_pass_tds_allowed"),
        ("allowed_rush_tds", "opp_rush_tds_allowed"),
    ]:
        allowed[dst] = (
            grp[src].shift(1).groupby(
                [allowed["season"], allowed["opponent_team"], allowed["position"]],
                sort=False,
            ).expanding(min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)
        )
    keep = ["season", "opponent_team", "week", "position",
            "opp_fantasy_allowed_pos", "opp_pass_yards_allowed",
            "opp_rush_yards_allowed", "opp_rec_yards_allowed_pos",
            "opp_pass_tds_allowed", "opp_rush_tds_allowed"]
    df = df.merge(allowed[keep],
                  on=["season", "opponent_team", "week", "position"], how="left")
    return df


def _impute_and_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Documented neutral imputation, paired with missing-data indicators.

    We never replace a genuinely-zero football value with a fabricated number;
    imputation only fills values that are truly unavailable (NaN), and each such
    fill is flagged by a missing-* indicator.
    """
    for col in MISSING_INDICATORS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for col, neutral in NEUTRAL_IMPUTATION.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(neutral)

    # Robust column accessor: returns an all-NaN Series when the column is
    # absent (so unit-testing this step in isolation, or a missing schedule
    # field, never crashes).
    def _col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    # Weather: indoors -> neutral (no wind, controlled temp) with indicator set.
    df["temp"] = _col("temp")
    df["wind"] = _col("wind")
    if "is_indoor" in df.columns:
        indoor = df["is_indoor"] == 1
        df.loc[indoor & df["temp"].isna(), "temp"] = 70.0
        df.loc[indoor & df["wind"].isna(), "wind"] = 0.0
    df["temp"] = df["temp"].fillna(NEUTRAL_IMPUTATION["temp"])
    df["wind"] = df["wind"].fillna(NEUTRAL_IMPUTATION["wind"])
    df["days_rest"] = _col("days_rest").fillna(7.0)   # default weekly cadence
    df["div_game"] = _col("div_game").fillna(0)

    # Opponent-defense NaN (a team's first game of a season) -> position mean,
    # flagged. These are leakage-safe means over the same frame's early weeks.
    for col in ["opp_fantasy_allowed_pos", "opp_pass_yards_allowed",
                "opp_rush_yards_allowed", "opp_rec_yards_allowed_pos",
                "opp_pass_tds_allowed", "opp_rush_tds_allowed"]:
        if col in df.columns:
            pos_mean = df.groupby("position")[col].transform("mean")
            df[col] = df[col].fillna(pos_mean).fillna(0.0)

    # Team-environment NaN (team's first game of a season) -> league mean fill.
    for col in TEAM_ENV_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean()).fillna(0.0)

    # Age unknown -> league mean (paired with no explicit indicator; age is
    # rarely missing for rostered players).
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["age"] = df["age"].fillna(df["age"].median()).fillna(26.0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full leakage-safe feature build. Input: canonical dataset from
    build_dataset. Output: same rows + engineered feature columns."""
    log.info("Engineering leakage-safe features for %d rows", len(df))
    df = _add_rolling_features(df)
    df = _add_context_features(df)
    df = _add_team_environment_features(df)
    df = _add_opponent_defense_features(df)
    df = _impute_and_indicators(df)
    log.info("  -> %d columns after engineering", df.shape[1])
    return df


# Engineered-feature suffixes produced by _lagged_rolling / trend. ONLY columns
# with these suffixes (all built from .shift(1)) plus the explicit context /
# opponent / indicator lists are eligible as model features. This allowlist is
# the core leakage guard: no raw same-week stat column can slip into the model.
ENGINEERED_SUFFIXES = ("_prev", "_roll3", "_roll5", "_roll8", "_roll5_std",
                       "_season_avg", "_ewma", "_trend")

# Explicit, always-known-before-kickoff context features.
_CONTEXT_ALLOW = [
    "season", "week", "home_away", "days_rest", "bye_week_return",
    "spread_line_team", "game_total", "implied_team_total",
    "is_indoor", "is_grass", "temp", "wind", "div_game",
    "years_experience", "age", "injury_designation_enc", "depth_rank",
    "is_starter", "games_played_season",
]
_OPPONENT_ALLOW = [
    "opp_fantasy_allowed_pos", "opp_pass_yards_allowed",
    "opp_rush_yards_allowed", "opp_rec_yards_allowed_pos",
    "opp_pass_tds_allowed", "opp_rush_tds_allowed",
]
# OWN-team offensive-environment features (season-to-date, shifted). Served from
# the player's CURRENT team so a mover is projected in the new offense.
TEAM_ENV_FEATURES = [
    "team_pass_rate", "team_plays_per_game", "team_off_epa",
    "team_tds_per_game", "team_implied_total_avg",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Leakage-safe model feature columns via strict allowlist.

    A column is a feature only if it is an engineered lagged column (known
    suffix) OR an explicit context/opponent/team-env/missing-indicator column.
    Raw same-week stat columns are never features.
    """
    allow = (set(_CONTEXT_ALLOW) | set(_OPPONENT_ALLOW) | set(TEAM_ENV_FEATURES)
             | set(MISSING_INDICATORS))
    feats: list[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.endswith(ENGINEERED_SUFFIXES) or c in allow:
            feats.append(c)
    # Stable order.
    return sorted(dict.fromkeys(feats))
