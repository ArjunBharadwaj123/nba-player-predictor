"""
Serve-time helpers for the NFL API.
===================================
Loads the compact serving artifacts produced by ``nfl.pipeline.update`` and
builds a single leakage-safe feature row for a player's UPCOMING game: it starts
from the player's most recent engineered row (their form heading into the next
game) and overlays the upcoming game's context (opponent, venue, odds, weather,
rest, injury) — using only pre-kickoff information.

All artifacts are loaded lazily and cached so the deployed API stays light.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from nfl.config import DATA_PROCESSED
from nfl.features.feature_config import NEUTRAL_IMPUTATION
from nfl.explainability.explainer import load_feature_names

log = logging.getLogger(__name__)

SERVING_PARQUET  = DATA_PROCESSED / "features_serving.parquet"
PLAYERS_JSON     = DATA_PROCESSED / "players.json"
OPP_DEFENSE_JSON = DATA_PROCESSED / "opp_defense.json"
FRESHNESS_JSON   = DATA_PROCESSED / "freshness.json"


@lru_cache(maxsize=1)
def load_serving() -> pd.DataFrame:
    if not SERVING_PARQUET.exists():
        raise FileNotFoundError(
            f"{SERVING_PARQUET} not found — run `python -m nfl.pipeline.update`.")
    return pd.read_parquet(SERVING_PARQUET)


@lru_cache(maxsize=1)
def load_players() -> list[dict]:
    return json.loads(PLAYERS_JSON.read_text()) if PLAYERS_JSON.exists() else []


@lru_cache(maxsize=1)
def load_opp_defense() -> dict:
    return json.loads(OPP_DEFENSE_JSON.read_text()) if OPP_DEFENSE_JSON.exists() else {}


@lru_cache(maxsize=1)
def load_freshness() -> dict:
    return json.loads(FRESHNESS_JSON.read_text()) if FRESHNESS_JSON.exists() else {}


def find_player(player_id: str) -> dict | None:
    for p in load_players():
        if p["id"] == player_id:
            return p
    return None


def latest_player_row(player_id: str) -> pd.Series | None:
    df = load_serving()
    rows = df[df["gsis_id"] == player_id].sort_values(["season", "week"])
    if rows.empty:
        return None
    return rows.iloc[-1]


def recent_games(player_id: str, n: int = 6) -> list[dict]:
    df = load_serving()
    rows = df[df["gsis_id"] == player_id].sort_values(["season", "week"]).tail(n)
    cols = ["season", "week", "opponent_team", "home_away",
            "passing_yards", "rushing_yards", "receiving_yards",
            "receptions", "targets", "carries", "attempts", "completions",
            "passing_tds", "rushing_tds", "receiving_tds",
            "field_goals_made", "extra_points_made", "kicking_points",
            "fantasy_points_component"]
    cols = [c for c in cols if c in rows.columns]
    out = []
    for _, r in rows[cols].iterrows():
        out.append({c: _clean(r[c]) for c in cols})
    return out


def _clean(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not np.isfinite(f) else round(f, 3)
    return v


INJURY_RANK = {"OUT": 4, "DOUBTFUL": 3, "QUESTIONABLE": 2, "PROBABLE": 1}


def build_upcoming_row(player_id: str, position: str, context: dict) -> pd.DataFrame:
    """One-row feature frame for the player's next game.

    ``context`` may include: opponent_team, home_away, days_rest, spread_line_team,
    game_total, implied_team_total, is_indoor, is_grass, temp, wind, div_game,
    injury_designation, bye_week_return. Missing values fall back to documented
    neutral imputation with the matching missing-* indicator set.
    """
    base = latest_player_row(player_id)
    if base is None:
        raise KeyError(player_id)
    feats = load_feature_names()
    row = base.copy()

    # Advance games-played counter (one more game than last observed).
    if "games_played_season" in row.index:
        row["games_played_season"] = float(row.get("games_played_season", 0)) + 1

    def setf(col, val, neutral_key=None, indicator=None):
        if val is None:
            if indicator:
                row[indicator] = 1
            if neutral_key is not None:
                row[col] = NEUTRAL_IMPUTATION.get(neutral_key, 0.0)
            return
        row[col] = val
        if indicator:
            row[indicator] = 0

    setf("home_away", context.get("home_away"))
    if context.get("home_away") is None:
        row["home_away"] = 1  # default neutral: treat as home
    setf("days_rest", context.get("days_rest"))
    if context.get("days_rest") is None:
        row["days_rest"] = 7.0
    setf("spread_line_team", context.get("spread_line_team"),
         "spread_line_team", "missing_odds")
    setf("game_total", context.get("game_total"), "game_total", "missing_odds")
    setf("implied_team_total", context.get("implied_team_total"),
         "implied_team_total", "missing_odds")
    setf("temp", context.get("temp"), "temp", "missing_weather")
    setf("wind", context.get("wind"), "wind", "missing_weather")
    if context.get("is_indoor") is not None:
        row["is_indoor"] = int(context["is_indoor"])
    if context.get("is_grass") is not None:
        row["is_grass"] = int(context["is_grass"])
    if context.get("div_game") is not None:
        row["div_game"] = int(context["div_game"])
    row["bye_week_return"] = int(bool(context.get("bye_week_return", 0)))

    # Injury designation.
    des = (context.get("injury_designation") or "").upper()
    row["injury_designation_enc"] = float(INJURY_RANK.get(des, 0))
    row["missing_injury"] = 0 if des else int(row.get("missing_injury", 1))

    # Opponent-defense context for the upcoming opponent.
    opp = context.get("opponent_team")
    if opp:
        table = load_opp_defense().get(position, {}).get(opp)
        if table:
            for k, v in table.items():
                if v is not None:
                    row[k] = v

    frame = row.to_frame().T
    X = frame.reindex(columns=feats).apply(pd.to_numeric, errors="coerce")
    return X


def injury_from_row(player_id: str) -> str | None:
    """Latest injury designation string from the serving row, if present."""
    base = latest_player_row(player_id)
    if base is None:
        return None
    val = base.get("injury_designation")
    if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
        return None
    return str(val)
