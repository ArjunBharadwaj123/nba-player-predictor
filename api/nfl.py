"""
NFL FastAPI router (mounted under /nfl).
========================================
Isolated from the NBA endpoints. Exposes:

    GET  /nfl/health
    GET  /nfl/players
    GET  /nfl/next-game/{player_id}
    POST /nfl/predict
    GET  /nfl/probability
    POST /nfl/score                    (lightweight PPR recompute; no ML rerun)
    GET  /nfl/players/{player_id}/recent

Models are loaded lazily by position with a bounded LRU cache — the deployed API
never loads every position at startup. All responses are JSON-safe (no NaN /
Infinity / NumPy scalars / pandas objects).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from nfl.config import (
    POSITION_TARGETS, SUPPORTED_POSITIONS, normalize_position,
    position_support_message,
)
from nfl import scoring
from nfl.serving import (
    find_player, load_players, load_freshness, build_upcoming_row,
    recent_games, injury_from_row, latest_player_row,
)
from nfl.explainability.explainer import predict_player, load_position_bundle

log = logging.getLogger(__name__)
router = APIRouter(prefix="/nfl", tags=["nfl"])


# ── JSON-safety helper ──────────────────────────────────────────────────────────
def _safe(obj):
    """Recursively coerce numpy/pandas scalars and non-finite floats to plain
    JSON-serializable values."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── Request / response models ───────────────────────────────────────────────────
class GameContext(BaseModel):
    opponent_team: str | None = None
    home_away: int | None = Field(None, ge=0, le=1)
    days_rest: float | None = None
    spread_line_team: float | None = None
    game_total: float | None = None
    implied_team_total: float | None = None
    is_indoor: int | None = Field(None, ge=0, le=1)
    is_grass: int | None = Field(None, ge=0, le=1)
    temp: float | None = None
    wind: float | None = None
    div_game: int | None = Field(None, ge=0, le=1)
    injury_designation: str | None = None
    bye_week_return: int | None = Field(None, ge=0, le=1)


class PredictRequest(BaseModel):
    player_id: str
    scoring_format: str = "ppr"
    fantasy_threshold: float | None = None
    context: GameContext = Field(default_factory=GameContext)

    @field_validator("scoring_format")
    @classmethod
    def _valid_format(cls, v):
        if v not in scoring.VALID_SCORING_FORMATS:
            raise ValueError(
                f"scoring_format must be one of {scoring.VALID_SCORING_FORMATS}")
        return v


class ScoreRequest(BaseModel):
    """Recompute fantasy points/interval from EXISTING component predictions
    without rerunning the ML models (used when the PPR format changes)."""
    position: str
    scoring_format: str = "ppr"
    predictions: dict[str, float]
    intervals: dict[str, list[float]] | None = None
    fantasy_threshold: float | None = None

    @field_validator("scoring_format")
    @classmethod
    def _valid_format(cls, v):
        if v not in scoring.VALID_SCORING_FORMATS:
            raise ValueError(
                f"scoring_format must be one of {scoring.VALID_SCORING_FORMATS}")
        return v


# ── Endpoints ────────────────────────────────────────────────────────────────────
@router.get("/health")
def health():
    fresh = load_freshness()
    return _safe({
        "status": "ok",
        "supported_positions": SUPPORTED_POSITIONS,
        "data_freshness": fresh,
        "n_players": len(load_players()),
    })


@router.get("/players")
def players(position: str | None = Query(None, description="Filter: QB/RB/WR/TE/K")):
    all_players = load_players()
    if position:
        pos = position.upper()
        if pos not in SUPPORTED_POSITIONS:
            raise HTTPException(422, f"Unsupported position filter '{position}'.")
        all_players = [p for p in all_players if p["position"] == pos]
    return _safe({"players": all_players, "total": len(all_players)})


@router.get("/next-game/{player_id}")
def next_game(player_id: str):
    player = find_player(player_id)
    if player is None:
        raise HTTPException(404, f"Player id '{player_id}' not found.")
    # Import lazily so schedule/network code isn't loaded unless needed.
    from nfl.scraping.current_context import get_next_game_context
    ctx = get_next_game_context(player_id)
    if ctx is None:
        raise HTTPException(
            404,
            f"No upcoming game found for {player['name']}. The NFL regular "
            f"season runs September through early January — check back closer "
            f"to game day.")
    return _safe(ctx)


def _validate_position(player: dict) -> str:
    pos = normalize_position(player.get("position"))
    if pos is None or pos not in SUPPORTED_POSITIONS:
        raise HTTPException(422, position_support_message(player.get("position")))
    return pos


@router.post("/predict")
def predict(req: PredictRequest):
    player = find_player(req.player_id)
    if player is None:
        raise HTTPException(404, f"Player id '{req.player_id}' not found.")
    position = _validate_position(player)

    if latest_player_row(req.player_id) is None:
        raise HTTPException(404, f"No serving data for {player['name']}.")

    try:
        X = build_upcoming_row(req.player_id, position, req.context.model_dump())
    except KeyError:
        raise HTTPException(404, f"No serving data for {player['name']}.")
    except Exception as exc:
        raise HTTPException(500, f"Feature build failed: {exc}")

    try:
        result = predict_player(position, X, req.scoring_format,
                                req.fantasy_threshold)
    except FileNotFoundError:
        raise HTTPException(503, f"Models for {position} not available.")
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")

    warnings = _build_warnings(req.player_id, position, result, req.context)
    fresh = load_freshness()
    payload = {
        "player": {"id": player["id"], "name": player["name"],
                   "position": position, "team": player.get("team"),
                   "headshot_url": player.get("headshot_url")},
        "scoring_format": result["fantasy"]["scoring_format"],
        "reception_multiplier": result["fantasy"]["reception_multiplier"],
        "predictions": result["predictions"],
        "intervals": {k: list(v) for k, v in result["intervals"].items()},
        "fantasy_points": result["fantasy"],
        "reasons": result["reasons"],
        "warnings": warnings,
        "targets": POSITION_TARGETS[position],
        "model_version": f"{fresh.get('mode', 'unknown')}-{'-'.join(map(str, fresh.get('seasons', [])))}",
        "data_freshness": {"updated_at": fresh.get("updated_at"),
                           "mode": fresh.get("mode"),
                           "seasons": fresh.get("seasons")},
    }
    # Strip internal sim samples before serializing.
    payload["fantasy_points"] = {k: v for k, v in payload["fantasy_points"].items()
                                 if k != "samples"}
    return _safe(payload)


@router.post("/score")
def score(req: ScoreRequest):
    """Recompute fantasy points + interval for a NEW scoring format from already
    predicted component stats — never touches the XGBoost models."""
    pos = req.position.upper()
    if pos not in SUPPORTED_POSITIONS:
        raise HTTPException(422, position_support_message(req.position))
    intervals = None
    if req.intervals:
        intervals = {k: (float(v[0]), float(v[1]))
                     for k, v in req.intervals.items() if len(v) == 2}
    sim = scoring.simulate_fantasy_points(req.predictions, intervals, pos,
                                          req.scoring_format)
    out = {
        "scoring_format": scoring.resolve_scoring_format(req.scoring_format),
        "reception_multiplier": scoring.reception_multiplier(req.scoring_format),
        "label": scoring.SCORING_FORMAT_LABELS[
            scoring.resolve_scoring_format(req.scoring_format)] + " Fantasy Points",
        "point_estimate": scoring.fantasy_points(req.predictions, pos,
                                                 req.scoring_format),
        "simulated_mean": sim["mean"],
        "interval": [sim["low"], sim["high"]],
    }
    if req.fantasy_threshold is not None:
        out["over_probability"] = scoring.over_under_probability(
            sim["samples"], req.fantasy_threshold, "over")
        out["under_probability"] = scoring.over_under_probability(
            sim["samples"], req.fantasy_threshold, "under")
        out["threshold"] = req.fantasy_threshold
    return _safe(out)


@router.get("/probability")
def probability(
    player_id: str,
    stat: str,
    threshold: float,
    direction: str = "over",
    scoring_format: str = "ppr",
):
    """Over/under probability for one stat. Rejects stats that don't apply to the
    player's position (422). ``fantasy_points`` uses the correlated simulator."""
    if direction not in ("over", "under"):
        raise HTTPException(422, "direction must be 'over' or 'under'.")
    player = find_player(player_id)
    if player is None:
        raise HTTPException(404, f"Player id '{player_id}' not found.")
    position = _validate_position(player)

    valid = set(POSITION_TARGETS[position]) | {"fantasy_points"}
    if stat not in valid:
        raise HTTPException(
            422, f"Stat '{stat}' does not apply to a {position}. "
                 f"Valid: {sorted(valid)}.")
    if scoring_format not in scoring.VALID_SCORING_FORMATS:
        raise HTTPException(422, f"Invalid scoring_format '{scoring_format}'.")

    try:
        X = build_upcoming_row(player_id, position, {})
        result = predict_player(position, X, scoring_format, threshold)
    except Exception as exc:
        raise HTTPException(500, f"Probability computation failed: {exc}")

    if stat == "fantasy_points":
        samples = result["_sim_samples"]
        prob = scoring.over_under_probability(samples, threshold, direction)
        return _safe({
            "player_id": player_id, "stat": stat, "threshold": threshold,
            "direction": direction,
            "scoring_format": result["fantasy"]["scoring_format"],
            "scoring_label": result["fantasy"]["label"],
            "probability": prob, "pct_display": f"{round(prob * 100)}%",
            "predicted": result["fantasy"]["simulated_mean"],
        })

    # Component stat: normal-approx from the predicted mean + interval spread.
    mean = float(result["predictions"].get(stat, 0.0))
    lo, hi = result["intervals"].get(stat, (mean, mean))
    sigma = max((hi - lo) / 2.07, 1e-6)
    z = (threshold - mean) / sigma
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    prob = cdf if direction == "under" else 1 - cdf
    prob = max(0.01, min(0.99, prob))
    return _safe({
        "player_id": player_id, "stat": stat, "threshold": threshold,
        "direction": direction, "probability": prob,
        "pct_display": f"{round(prob * 100)}%", "predicted": round(mean, 2),
        "interval": [lo, hi],
    })


@router.get("/players/{player_id}/recent")
def player_recent(player_id: str, n: int = Query(6, ge=1, le=20)):
    player = find_player(player_id)
    if player is None:
        raise HTTPException(404, f"Player id '{player_id}' not found.")
    return _safe({"player": {"id": player["id"], "name": player["name"],
                             "position": player["position"]},
                  "recent_games": recent_games(player_id, n)})


def _build_warnings(player_id, position, result, context) -> list[str]:
    warnings = []
    des = (context.injury_designation or injury_from_row(player_id) or "").upper()
    if des in ("OUT", "DOUBTFUL"):
        warnings.append(f"Injury designation: {des.title()} — high risk of "
                        f"missing or limited play.")
    elif des == "QUESTIONABLE":
        warnings.append("Questionable injury designation increases uncertainty.")

    base = latest_player_row(player_id)
    if base is not None:
        games = int(base.get("games_played_season", 0) or 0)
        if games < 4:
            warnings.append(f"Limited current-season sample ({games} prior "
                            f"games) — prediction less reliable.")
        snap = base.get("offense_pct")
        if position != "K" and snap is not None and not (isinstance(snap, float)
                                                         and math.isnan(snap)):
            if float(snap) < 0.4:
                warnings.append("Low recent snap share — role uncertainty.")
    if context.opponent_team is None:
        warnings.append("No upcoming-game context supplied — using neutral "
                        "matchup values. Provide opponent for a sharper read.")
    fresh = load_freshness()
    if fresh.get("mode") == "dev":
        warnings.append("Development model (reduced seasons) — not final "
                        "production accuracy.")
    return warnings
