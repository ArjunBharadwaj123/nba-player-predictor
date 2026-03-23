"""
FastAPI Prediction Server
==========================
Endpoints:
    GET  /health                          — server status
    GET  /next-game/{player_name}         — auto-fetch next game context
    POST /predict                         — run prediction
    POST /probability                     — over/under probability
    GET  /players                         — list available players

Usage:
    uvicorn api.main:app --reload
"""

import json
import logging
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import scipy.stats as scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from scraping.next_game import get_next_game_context
    _NEXT_GAME_OK = True
except Exception as _e:
    logging.warning("next_game module unavailable: %s", _e)
    _NEXT_GAME_OK = False
    def get_next_game_context(*a, **kw):
        return None

from explainability.shap_explainer import (
    explain_prediction,
    load_feature_names,
    load_models,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED  = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "saved"
TARGETS    = ["pts", "reb", "ast", "stl", "blk", "minutes"]

POSITION_MAP = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}

SEASON_STARTS = {
    2022: "2021-10-19", 2023: "2022-10-18",
    2024: "2023-10-24", 2025: "2024-10-22",
    2026: "2025-10-22",
}


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NBA Player Performance Predictor",
    description="Predicts per-game stats with SHAP explanations and injury awareness.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict       = {}
_feature_names: list = []
_features_df        = None
_live_cache: dict   = {}   # player_name.lower() -> fresh BBRef game log df


@app.on_event("startup")
async def startup():
    global _models, _feature_names, _features_df
    log.info("Loading models...")
    _models = load_models()
    log.info("  Loaded: %s", list(_models.keys()))

    log.info("Loading feature names...")
    _feature_names = load_feature_names()
    df = _get_df()
    _feature_names = [f for f in _feature_names if f in df.columns]
    log.info("  %d features ready", len(_feature_names))
    log.info("API ready.")


def _get_df() -> pd.DataFrame:
    global _features_df
    if _features_df is None:
        _features_df = pd.read_csv(
            PROCESSED / "features.csv", parse_dates=["game_date"]
        )
        # Ensure usage_rate exists
        if "usage_rate" not in _features_df.columns and "usage_proxy" in _features_df.columns:
            _features_df["usage_rate"] = _features_df["usage_proxy"]
        # Ensure minutes std exists
        if "rolling_last5_minutes_std" not in _features_df.columns:
            _features_df["rolling_last5_minutes_std"] = (
                _features_df.groupby("player_id")["minutes"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
                .fillna(3.5)
            )
    return _features_df


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    player_name:     str   = Field(..., example="LeBron James")
    player_id:       str   = Field(..., example="jamesle01")
    opponent_abbrev: str   = Field(..., example="GSW")
    home_game:       bool  = Field(True)
    opp_def_rating:  float = Field(..., example=112.4)
    team_pace:       float = Field(..., example=99.8)
    rest_days:       int   = Field(1)
    back_to_back:    bool  = Field(False)
    opp_def_rank:    float = Field(15.0)
    position:        str   = Field("SF")


class StatPrediction(BaseModel):
    pts: float; reb: float; ast: float
    stl: float; blk: float; minutes: float


class StatRange(BaseModel):
    pts: tuple[float, float]; reb: tuple[float, float]
    ast: tuple[float, float]; stl: tuple[float, float]
    blk: tuple[float, float]; minutes: tuple[float, float]


class PredictResponse(BaseModel):
    player_name:    str
    opponent:       str
    game_date:      str | None
    predictions:    StatPrediction
    ranges:         StatRange
    fantasy_score:  float
    reasoning:      str
    warnings:       list[str]
    injury_status:  str | None
    model_version:  str = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Feature row builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_rolling_from_live(fresh_df: pd.DataFrame, team_pace: float) -> pd.Series:
    """
    Given a freshly scraped current-season game log, compute rolling averages
    and return the most recent row as a Series ready for prediction.
    """
    import numpy as np

    df = fresh_df.copy()
    if "trb" in df.columns and "reb" not in df.columns:
        df = df.rename(columns={"trb": "reb"})
    df = df.sort_values("game_date").reset_index(drop=True)

    ROLL_STATS = ["pts", "reb", "ast", "stl", "blk", "minutes",
                  "fg_pct", "fg3_pct", "ft_pct", "tov"]

    for stat in ROLL_STATS:
        if stat not in df.columns:
            continue
        df[stat] = pd.to_numeric(df[stat], errors="coerce")
        for w in [3, 5, 10]:
            df[f"rolling_last{w}_{stat}"] = (
                df[stat].shift(1).rolling(w, min_periods=1).mean()
            )

    for stat in ["pts", "reb", "ast", "stl", "blk", "minutes"]:
        if stat in df.columns:
            df[f"season_avg_{stat}"] = (
                df[stat].shift(1).expanding(min_periods=1).mean()
            )

    if all(c in df.columns for c in ["fga", "fta", "tov", "minutes"]):
        ps = (df["fga"].fillna(0) + 0.44 * df["fta"].fillna(0)
              + df["tov"].fillna(0))
        tp = (df["minutes"].clip(lower=1) / 48) * (team_pace or 99.5)
        df["usage_rate"] = (ps / tp.clip(lower=0.1)).clip(0, 1)
        for w in [3, 5, 10]:
            df[f"rolling_last{w}_usage_rate"] = (
                df["usage_rate"].shift(1).rolling(w, min_periods=1).mean()
            )
        df["season_avg_usage_rate"] = (
            df["usage_rate"].shift(1).expanding(min_periods=1).mean()
        )

    for stat in ["pts", "reb", "ast"]:
        l3  = f"rolling_last3_{stat}"
        l10 = f"rolling_last10_{stat}"
        if l3 in df.columns and l10 in df.columns:
            df[f"trend_{stat}"] = (
                df[l3] / df[l10].replace(0, np.nan)
            ).clip(0.3, 3.0)

    df["games_played_season"] = range(len(df))

    row = df.iloc[-1].copy()
    log.info(
        "Live rolling built — last game %s | rolling_last5_pts=%.1f | season_avg_pts=%.1f",
        str(row.get("game_date", "?"))[:10],
        row.get("rolling_last5_pts", 0),
        row.get("season_avg_pts", 0),
    )
    return row


def build_feature_row(req: PredictRequest, df: pd.DataFrame) -> pd.DataFrame:
    player_rows = df[
        df["player_name"].str.lower() == req.player_name.lower()
    ].copy()

    if player_rows.empty:
        raise HTTPException(
            status_code=404,
            detail="Player '" + req.player_name + "' not found in features dataset. "
                   "Run the scraper for this player first.",
        )

    # Always start from the training-data row — it has ALL feature columns.
    # Then overlay live rolling averages on top where we have fresh data.
    # This avoids KeyErrors from columns like rolling_last3_fantasy_score
    # that _build_rolling_from_live does not compute.
    row = player_rows.sort_values("game_date").iloc[-1].copy()

    cache_key = req.player_name.lower()
    if cache_key in _live_cache:
        live_row = _build_rolling_from_live(_live_cache[cache_key], req.team_pace)
        # Overwrite only columns that exist in both row and live_row
        for col in live_row.index:
            if col in row.index and pd.notna(live_row.get(col)):
                row[col] = live_row[col]
        row["season"] = 2026

    # Override context with tonight's matchup
    row["home_game"]        = int(req.home_game)
    row["back_to_back"]     = int(req.back_to_back)
    row["rest_days_capped"] = min(req.rest_days, 7)
    row["opp_def_rating"]   = req.opp_def_rating
    row["opp_def_rank"]     = req.opp_def_rank
    row["team_pace"]        = req.team_pace
    row["opponent_abbrev"]  = req.opponent_abbrev
    row["position"]         = req.position.upper().strip()
    row["position_enc"]     = POSITION_MAP.get(req.position.upper().strip(), 3)

    # Usage rate — recalculate from last game's raw stats with new pace
    if all(c in row.index for c in ["fga", "fta", "tov", "minutes"]):
        player_share = (row["fga"] + 0.44 * float(row.get("fta", 0) or 0)
                        + float(row.get("tov", 0) or 0))
        team_poss = (max(float(row.get("minutes", 30)), 1) / 48) * req.team_pace
        row["usage_rate"] = min(player_share / max(team_poss, 0.1), 1.0)

    # Schedule features
    season = int(row.get("season", 2026))
    start_str = SEASON_STARTS.get(season, "2025-10-22")
    days_in = (pd.Timestamp("today") - pd.Timestamp(start_str)).days
    row["days_into_season"]     = max(0, days_in)
    row["games_in_last_7_days"] = 3   # neutral fallback
    row["home_court_factor"]    = 0.575

    # Season phase (1=early, 2=mid, 3=late, 4=end)
    games_so_far = max(0, days_in // 2)   # rough estimate
    if games_so_far <= 20:   row["season_phase"] = 1
    elif games_so_far <= 55: row["season_phase"] = 2
    elif games_so_far <= 72: row["season_phase"] = 3
    else:                    row["season_phase"] = 4
    row["is_late_season"] = int(games_so_far >= 56)

    # Positional defense — load from processed file if available
    pos_file = ROOT / "data" / "processed" / "opp_pos_defense.csv"
    if pos_file.exists():
        try:
            pos_df = pd.read_csv(pos_file)
            POS_NBA = {"PG":"Point Guard","SG":"Shooting Guard",
                       "SF":"Small Forward","PF":"Power Forward","C":"Center"}
            nba_pos = POS_NBA.get(req.position.upper(), "Small Forward")
            match = pos_df[
                (pos_df["opponent_abbrev"] == req.opponent_abbrev) &
                (pos_df["season"] == season) &
                (pos_df["position"] == nba_pos)
            ]
            if not match.empty:
                row["opp_pts_allowed_pos"]  = float(match.iloc[0]["pts_allowed_per_game"])
                row["opp_pos_defense_rank"] = float(match.iloc[0]["rank"])
            else:
                row["opp_pts_allowed_pos"]  = 0.0
                row["opp_pos_defense_rank"] = 15.0
        except Exception:
            row["opp_pts_allowed_pos"]  = 0.0
            row["opp_pos_defense_rank"] = 15.0
    else:
        row["opp_pts_allowed_pos"]  = 0.0
        row["opp_pos_defense_rank"] = 15.0

    return row.to_frame().T


# ─────────────────────────────────────────────────────────────────────────────
# Confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

MIN_STD = {"pts": 4.0, "reb": 2.0, "ast": 1.5,
           "stl": 0.5, "blk": 0.4, "minutes": 3.0}

def compute_ranges(player_name: str, predictions: dict,
                   df: pd.DataFrame, window: int = 15) -> dict:
    player_rows = df[
        df["player_name"].str.lower() == player_name.lower()
    ].sort_values("game_date").tail(window)

    ranges = {}
    for stat in TARGETS:
        pred = float(predictions.get(stat, 0))
        if stat in player_rows.columns and len(player_rows) >= 3:
            std = float(player_rows[stat].dropna().std())
        else:
            std = MIN_STD[stat]
        std = max(std, MIN_STD[stat])
        ranges[stat] = (round(max(0.0, pred - std), 1), round(pred + std, 1))
    return ranges


# ─────────────────────────────────────────────────────────────────────────────
# Warnings
# ─────────────────────────────────────────────────────────────────────────────

def build_warnings(req: PredictRequest, df: pd.DataFrame,
                   predictions: dict,
                   injury_warning: str | None = None) -> list[str]:
    warnings = []

    if injury_warning:
        warnings.append(injury_warning)

    player_rows = df[df["player_name"].str.lower() == req.player_name.lower()]
    if not player_rows.empty:
        avg_min  = player_rows["minutes"].mean()
        pred_min = predictions.get("minutes", avg_min)
        if abs(pred_min - avg_min) > 8:
            warnings.append(
                "Predicted minutes (" + str(round(pred_min)) + ") differs from "
                "season average (" + str(round(avg_min)) + "). "
                "Check for load management or injury news."
            )
        if len(player_rows) < 20:
            warnings.append(
                "Limited data (" + str(len(player_rows)) + " games). "
                "Prediction may be less reliable."
            )

    if req.back_to_back:
        warnings.append(
            "Back-to-back game. Verify the player is confirmed in the "
            "starting lineup before using this prediction."
        )

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "models_loaded": list(_models.keys()),
        "feature_count": len(_feature_names),
        "next_game_api": _NEXT_GAME_OK,
        "scipy":         _SCIPY,
    }


@app.get("/next-game/{player_name}")
def next_game(player_name: str, player_id: str, position: str = "SF"):
    """Auto-fetch next game context from the NBA API."""
    ctx = get_next_game_context(player_name, player_id, position)
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail="Could not find next game for '" + player_name + "'. "
                   "The player may have no upcoming games scheduled, or the NBA API may be unavailable.",
        )

    # Re-scrape current season so rolling averages are live
    try:
        from scraping.bbref_scraper import scrape_one, CURRENT_SEASON
        fresh = scrape_one(
            {"id": player_id, "name": player_name, "pos": position},
            CURRENT_SEASON, force_refresh=True,
        )
        if not fresh.empty:
            _live_cache[player_name.lower()] = fresh
            log.info("Cached %d live rows for %s", len(fresh), player_name)
        else:
            log.warning("Empty live scrape for %s", player_name)
    except Exception as exc:
        log.warning("Live scrape failed for %s: %s", player_name, exc)

    return ctx


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, game_date: str = "", injury_warning: str = ""):
    """Run stat predictions for a player's upcoming game."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded")

    df = _get_df()

    try:
        feature_row = build_feature_row(req, df)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Feature building failed: " + str(exc))

    try:
        result = explain_prediction(feature_row, _models, _feature_names)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed: " + str(exc))

    preds    = result["predictions"]
    ranges   = compute_ranges(req.player_name, preds, df)
    warnings = build_warnings(req, df, preds, injury_warning or None)

    return PredictResponse(
        player_name=req.player_name,
        opponent=req.opponent_abbrev,
        game_date=game_date or None,
        predictions=StatPrediction(
            pts=round(preds.get("pts", 0), 1),
            reb=round(preds.get("reb", 0), 1),
            ast=round(preds.get("ast", 0), 1),
            stl=round(preds.get("stl", 0), 1),
            blk=round(preds.get("blk", 0), 1),
            minutes=round(preds.get("minutes", 0), 1),
        ),
        ranges=StatRange(
            pts=ranges["pts"], reb=ranges["reb"], ast=ranges["ast"],
            stl=ranges["stl"], blk=ranges["blk"], minutes=ranges["minutes"],
        ),
        fantasy_score=result["fantasy_score"],
        reasoning=result["reasoning_text"],
        warnings=warnings,
        injury_status=None,
    )


@app.get("/probability")
def probability(
    stat: str,
    threshold: float,
    direction: str,
    player_name: str,
    window: int = 20,
):
    """
    Compute probability that a player goes over/under a stat threshold.

    Blends empirical hit rate from recent games with normal distribution
    probability for robustness with small samples.
    """
    VALID_STATS = {"pts", "reb", "ast", "stl", "blk", "minutes"}
    if stat not in VALID_STATS:
        raise HTTPException(status_code=422,
                            detail="stat must be one of: " + ", ".join(VALID_STATS))
    if direction not in ("over", "under"):
        raise HTTPException(status_code=422,
                            detail="direction must be 'over' or 'under'")

    # Prefer live current-season data if available (populated by /next-game)
    cache_key = player_name.lower()
    if cache_key in _live_cache:
        live_df = _live_cache[cache_key].copy()
        if "trb" in live_df.columns and "reb" not in live_df.columns:
            live_df = live_df.rename(columns={"trb": "reb"})
        live_df[stat] = pd.to_numeric(live_df.get(stat, pd.Series()), errors="coerce")
        values = live_df[stat].dropna().tail(window)
        source = "current season (" + str(len(values)) + " games)"
    else:
        df = _get_df()
        player_rows = df[
            df["player_name"].str.lower() == player_name.lower()
        ].sort_values("game_date").tail(window)

        if player_rows.empty or stat not in player_rows.columns:
            raise HTTPException(status_code=404,
                                detail="Player not found: " + player_name)

        values = player_rows[stat].dropna()
        source = "training data (" + str(len(values)) + " games)"

    if len(values) < 3:
        raise HTTPException(status_code=422,
                            detail="Not enough data to compute probability")

    mean = float(values.mean())
    std  = float(values.std())
    n    = int(len(values))

    # Empirical hit rate
    hits = int((values > threshold).sum() if direction == "over"
               else (values < threshold).sum())
    hit_rate = hits / n

    # Normal distribution probability
    if std > 0 and _SCIPY:
        z = (threshold - mean) / std
        normal_prob = float(
            scipy_stats.norm.cdf(z) if direction == "under"
            else 1 - scipy_stats.norm.cdf(z)
        )
    elif std > 0:
        # Fallback without scipy: use error function approximation
        z = (threshold - mean) / std
        erf_val = math.erf(z / math.sqrt(2))
        cdf = 0.5 * (1 + erf_val)
        normal_prob = float(cdf if direction == "under" else 1 - cdf)
    else:
        normal_prob = 1.0 if (direction == "over" and mean > threshold) else 0.0

    # Blend: weight empirical more as sample grows
    emp_weight  = min(n / 20, 1.0)
    norm_weight = 1 - emp_weight * 0.5
    total       = emp_weight + norm_weight
    blend       = (hit_rate * emp_weight + normal_prob * norm_weight) / total
    blend       = max(0.02, min(0.98, blend))

    return {
        "probability"  : round(blend, 3),
        "pct_display"  : str(round(blend * 100)) + "%",
        "hit_rate"     : round(hit_rate, 3),
        "normal_prob"  : round(normal_prob, 3),
        "sample_size"  : n,
        "stat_mean"    : round(mean, 1),
        "stat_std"     : round(std, 1),
        "direction"    : direction,
        "threshold"    : threshold,
        "stat"         : stat,
        "data_source"  : source,
    }


@app.get("/players")
def list_players():
    df = _get_df()
    players = (
        df.groupby("player_name")
        .agg(games=("game_date", "count"), last_game=("game_date", "max"))
        .reset_index()
        .sort_values("player_name")
    )
    return {"players": players.to_dict(orient="records"), "total": len(players)}


@app.get("/players/{player_name}/recent")
def player_recent(player_name: str, n: int = 5):
    df = _get_df()
    rows = df[df["player_name"].str.lower() == player_name.lower()]
    if rows.empty:
        raise HTTPException(status_code=404,
                            detail="Player not found: " + player_name)
    recent = rows.sort_values("game_date").tail(n)
    cols = ["game_date", "opponent_abbrev", "home_game", "minutes",
            "pts", "reb", "ast", "stl", "blk",
            "usage_rate", "rolling_last5_pts", "opp_def_rating", "rest_days_capped"]
    cols = [c for c in cols if c in recent.columns]
    return {"player": player_name, "recent_games": recent[cols].to_dict(orient="records")}