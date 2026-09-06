"""
Build the canonical NFL player-week dataset.
============================================
Merges weekly player_stats with schedule context, injuries, snap counts, depth
charts, weekly rosters and canonical player metadata into ONE tidy pandas frame,
one row per (canonical player_id, game). Derives the canonical prediction targets
(mapping nflverse field names -> POSITION_TARGETS names) and records which
optional signals were available.

Everything here is *pre-engineering*: it assembles raw, same-week values and
static game context. All leakage-safe lagging happens later in ``engineer.py``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nfl.config import (
    SUPPORTED_POSITIONS,
    normalize_position,
)
from nfl.scraping.collect import DataBundle, to_pandas

log = logging.getLogger(__name__)

# Distance buckets (nflverse) -> canonical kicker tier columns.
FG_BUCKET_0_39  = ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"]
FG_BUCKET_40_49 = ["fg_made_40_49"]
FG_BUCKET_50P   = ["fg_made_50_59", "fg_made_60_"]

# Team-relocation / abbreviation standardization. nflverse `player_stats` uses
# the CURRENT franchise code (e.g. LV) while historical `schedules` use the
# era-accurate code (e.g. OAK). Left un-aligned, the schedule merge misses for
# those seasons and every game-context field (date, spread, total, weather) is
# lost. We map both sides to the current code before merging.
TEAM_STANDARDIZE = {
    "OAK": "LV",   # Raiders -> Las Vegas (2020)
    "SD":  "LAC",  # Chargers -> Los Angeles (2017)
    "STL": "LA",   # Rams -> Los Angeles (2016)
    "LAR": "LA",   # some feeds use LAR for the Rams
    "JAC": "JAX",  # Jacksonville abbreviation variant
    "WSH": "WAS",  # Washington abbreviation variant
}


def _standardize_teams(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(TEAM_STANDARDIZE)
    return df


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def _derive_targets(ps: pd.DataFrame) -> pd.DataFrame:
    """Add canonical target columns (rename + derive kicker fields)."""
    df = ps.copy()

    # Canonical renames where nflverse differs.
    df["field_goals_made"] = _num(df, "fg_made")
    df["extra_points_made"] = _num(df, "pat_made")
    df["fumbles_lost"] = _num(df, "fumbles_lost_total")

    # Distance-tiered FG buckets (used by kicker scoring for kicking_points).
    df["field_goals_made_0_39"]   = sum(_num(df, c) for c in FG_BUCKET_0_39)
    df["field_goals_made_40_49"]  = sum(_num(df, c) for c in FG_BUCKET_40_49)
    df["field_goals_made_50_plus"] = sum(_num(df, c) for c in FG_BUCKET_50P)

    # kicking_points derived from distance-tiered FGs + PATs (matches scoring.py
    # kicker constants: 3 / 4 / 5 per FG tier, 1 per XP).
    df["kicking_points"] = (
        df["field_goals_made_0_39"]   * 3.0
        + df["field_goals_made_40_49"] * 4.0
        + df["field_goals_made_50_plus"] * 5.0
        + df["extra_points_made"]      * 1.0
    )

    # A component-derived fantasy value (PPR) used only as a rolling *feature*
    # (never a training target). Skill positions only; kickers use kicking_points.
    df["fantasy_points_component"] = (
        _num(df, "passing_yards") * 0.04
        + _num(df, "passing_tds") * 4.0
        + _num(df, "passing_interceptions") * -2.0
        + _num(df, "rushing_yards") * 0.1
        + _num(df, "rushing_tds") * 6.0
        + _num(df, "receiving_yards") * 0.1
        + _num(df, "receiving_tds") * 6.0
        + _num(df, "receptions") * 1.0
        + df["fumbles_lost"] * -2.0
    )
    return df


def _schedule_context(schedules: pd.DataFrame) -> pd.DataFrame:
    """Explode schedule into one row per (game_id, team) with team-relative
    context: home/away, days rest, team spread, game total, implied team total,
    roof/surface/weather.

    nflverse convention: ``spread_line`` is the point spread from the HOME team's
    perspective (positive = home favored). We convert to the team's own spread
    (negative = favored) and implied points total.
    """
    s = schedules.copy()
    rows = []
    for _, g in s.iterrows():
        total = g.get("total_line")
        spread_home = g.get("spread_line")   # home perspective (+ => home fav)
        for side in ("home", "away"):
            team = g.get(f"{side}_team")
            opp = g.get("away_team" if side == "home" else "home_team")
            if not team:
                continue
            # Team spread: negative when favored. Home favored (spread_home>0)
            # means home's own spread is -spread_home.
            team_spread = None
            implied = None
            if pd.notna(spread_home):
                team_spread = -float(spread_home) if side == "home" else float(spread_home)
            if pd.notna(total) and team_spread is not None:
                # implied = total/2 - team_spread/2  (favored team scores more)
                implied = float(total) / 2.0 - team_spread / 2.0
            rows.append({
                "game_id": g.get("game_id"),
                "season": g.get("season"),
                "week": g.get("week"),
                "team": team,
                "opponent_team": opp,
                "gameday": g.get("gameday"),
                "gametime": g.get("gametime"),
                "home_away": 1 if side == "home" else 0,
                "days_rest": g.get(f"{side}_rest"),
                "game_total": float(total) if pd.notna(total) else np.nan,
                "spread_line_team": team_spread if team_spread is not None else np.nan,
                "implied_team_total": implied if implied is not None else np.nan,
                "roof": g.get("roof"),
                "surface": g.get("surface"),
                "temp": g.get("temp"),
                "wind": g.get("wind"),
                "div_game": g.get("div_game"),
            })
    ctx = pd.DataFrame(rows)
    # Indoor / grass indicators + weather availability.
    roof = ctx["roof"].astype("string").str.lower().fillna("")
    ctx["is_indoor"] = roof.isin(["dome", "closed"]).astype(int)
    surface = ctx["surface"].astype("string").str.lower().fillna("")
    ctx["is_grass"] = (surface == "grass").astype(int)
    # Weather only meaningful outdoors; indoors temp/wind are effectively neutral.
    ctx["missing_weather"] = (
        ctx["temp"].isna() | (ctx["is_indoor"] == 1)
    ).astype(int)
    ctx["missing_odds"] = ctx["game_total"].isna().astype(int)
    return ctx


INJURY_RANK = {
    "OUT": 4, "DOUBTFUL": 3, "QUESTIONABLE": 2, "PROBABLE": 1, "": 0, None: 0,
}


def build_dataset(bundle: DataBundle) -> pd.DataFrame:
    """Assemble the canonical player-week dataset (pandas)."""
    ps = to_pandas(bundle.player_stats)
    players = to_pandas(bundle.players)
    schedules = to_pandas(bundle.schedules)

    # Align team codes across sources so relocated franchises merge correctly
    # (e.g. player_stats "LV" vs 2018 schedule "OAK").
    ps = _standardize_teams(ps, ["team", "opponent_team"])
    schedules = _standardize_teams(schedules, ["home_team", "away_team"])

    log.info("Building dataset from %d player-week rows", len(ps))

    # ── Supported positions only ───────────────────────────────────────────────
    ps["position_norm"] = [
        normalize_position(p, g)
        for p, g in zip(ps.get("position"), ps.get("position_group"))
    ]
    ps = ps[ps["position_norm"].isin(SUPPORTED_POSITIONS)].copy()
    log.info("  %d rows after filtering to %s", len(ps), SUPPORTED_POSITIONS)

    # ── Canonical targets ──────────────────────────────────────────────────────
    ps = _derive_targets(ps)

    # Canonical id/name.
    ps = ps.rename(columns={"player_id": "gsis_id"})
    ps["player_name"] = ps.get("player_display_name").fillna(ps.get("player_name"))

    # ── Schedule context ───────────────────────────────────────────────────────
    ctx = _schedule_context(schedules)
    df = ps.merge(
        ctx.drop(columns=["opponent_team"], errors="ignore"),
        on=["game_id", "season", "week", "team"],
        how="left", suffixes=("", "_sched"),
    )

    # ── Injuries (optional) ────────────────────────────────────────────────────
    if bundle.injuries is not None:
        inj = to_pandas(bundle.injuries)[
            ["gsis_id", "season", "week", "team", "report_status", "practice_status"]
        ].copy()
        inj = inj.drop_duplicates(["gsis_id", "season", "week", "team"], keep="last")
        df = df.merge(inj, on=["gsis_id", "season", "week", "team"], how="left")
        df["missing_injury"] = df["report_status"].isna().astype(int)
        df["injury_designation"] = df["report_status"].fillna("")
    else:
        df["missing_injury"] = 1
        df["injury_designation"] = ""
    df["injury_designation_enc"] = (
        df["injury_designation"].astype("string").str.upper()
        .map(lambda x: INJURY_RANK.get(x, 0)).fillna(0).astype(float)
    )

    # ── Snap counts (optional) — join via players.pfr_id -> pfr_player_id ───────
    df["missing_snap_share"] = 1
    df["offense_pct"] = np.nan
    if bundle.snap_counts is not None and players is not None and "pfr_id" in players.columns:
        pfr_map = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
        snaps = to_pandas(bundle.snap_counts)[
            ["pfr_player_id", "season", "week", "offense_pct"]
        ].copy()
        snaps = snaps.drop_duplicates(["pfr_player_id", "season", "week"], keep="last")
        df = df.merge(pfr_map, on="gsis_id", how="left")
        df = df.merge(
            snaps.rename(columns={"pfr_player_id": "pfr_id",
                                  "offense_pct": "offense_pct_snap"}),
            on=["pfr_id", "season", "week"], how="left",
        )
        df["offense_pct"] = df["offense_pct_snap"]
        df["missing_snap_share"] = df["offense_pct"].isna().astype(int)

    # ── Depth charts (optional) ────────────────────────────────────────────────
    df["missing_depth"] = 1
    df["depth_rank"] = np.nan
    if bundle.depth_charts is not None:
        dc = to_pandas(bundle.depth_charts).copy()
        dc["depth_team"] = pd.to_numeric(dc["depth_team"], errors="coerce")
        dc = (dc.dropna(subset=["gsis_id"])
                .groupby(["gsis_id", "season", "week"], as_index=False)["depth_team"]
                .min())
        df = df.merge(dc.rename(columns={"depth_team": "depth_rank_dc"}),
                      on=["gsis_id", "season", "week"], how="left")
        df["depth_rank"] = df["depth_rank_dc"]
        df["missing_depth"] = df["depth_rank"].isna().astype(int)
    df["is_starter"] = (df["depth_rank"] == 1).astype(int)

    # ── Weekly rosters (experience / status / headshot) ────────────────────────
    if bundle.rosters_weekly is not None:
        rw = to_pandas(bundle.rosters_weekly)[
            ["gsis_id", "season", "week", "years_exp", "status", "headshot_url"]
        ].copy()
        rw = rw.drop_duplicates(["gsis_id", "season", "week"], keep="last")
        df = df.merge(rw, on=["gsis_id", "season", "week"], how="left")
        df["years_experience"] = pd.to_numeric(df.get("years_exp"), errors="coerce")
    else:
        df["years_experience"] = np.nan

    # ── Canonical player metadata (headshot / experience fallback) ─────────────
    if players is not None:
        pmeta = players[["gsis_id", "display_name", "position",
                         "years_of_experience"]].copy()
        if "headshot" in players.columns:
            pmeta["headshot"] = players["headshot"]
        pmeta = pmeta.drop_duplicates("gsis_id")
        df = df.merge(pmeta, on="gsis_id", how="left", suffixes=("", "_meta"))
        df["years_experience"] = df["years_experience"].fillna(
            pd.to_numeric(df.get("years_of_experience"), errors="coerce"))
        # Headshot: prefer weekly roster, then player_stats, then players meta.
        hs = df.get("headshot_url")
        if hs is None:
            hs = pd.Series(np.nan, index=df.index)
        df["headshot_url"] = (
            hs.fillna(df.get("headshot_url_meta"))
              .fillna(df.get("headshot"))
        )

    df["years_experience"] = df["years_experience"].fillna(0.0)

    # ── Canonical dedup: one row per (gsis_id, game_id) ────────────────────────
    df["gameday"] = pd.to_datetime(df["gameday"], errors="coerce")
    before = len(df)
    df = df.sort_values(["gsis_id", "season", "week"]).drop_duplicates(
        ["gsis_id", "game_id"], keep="last").reset_index(drop=True)
    if len(df) != before:
        log.info("  deduped %d -> %d rows (canonical player+game)", before, len(df))

    df["position"] = df["position_norm"]
    log.info("Dataset built: %d rows, %d columns", len(df), df.shape[1])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    from nfl.config import default_seasons
    from nfl.scraping.collect import collect
    bundle = collect(default_seasons())
    ds = build_dataset(bundle)
    print(ds[["player_name", "position", "season", "week", "team",
              "opponent_team", "home_away", "days_rest", "implied_team_total",
              "passing_yards", "receiving_yards", "kicking_points"]].head(12).to_string())
    print("positions:", ds["position"].value_counts().to_dict())
