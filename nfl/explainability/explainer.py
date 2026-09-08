"""
NFL SHAP explanations in football language.
===========================================
Loads a position's model bundle lazily, produces per-target predictions, p15/p85
intervals, and SHAP-based reasoning translated into readable football statements
(never raw feature names). Also assembles the fantasy-point projection + interval
via the correlated Monte-Carlo simulator in ``nfl.scoring``.

SHAP describes how each feature moved THIS prediction relative to the model's
baseline — it is an attribution, not a claim of causation. Reason strings are
phrased accordingly.
"""

from __future__ import annotations

import json
import logging
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from nfl.config import (
    POSITION_TARGETS, NONNEGATIVE_TARGETS, MODELS_SAVED,
)
from nfl import scoring

log = logging.getLogger(__name__)

try:
    import shap
    _SHAP_OK = True
except Exception:                       # shap optional at serve time
    _SHAP_OK = False


# ── Readable feature labels (football language) ─────────────────────────────────
# Each entry: (matcher, positive_template, negative_template). Matched against the
# feature name; {val} is the feature's value. Binary/context features use value.
FEATURE_LABELS = [
    ("fantasy_points_component_roll3", "Hot recent form (last 3 games)",
                                       "Cold recent form (last 3 games)"),
    ("passing_yards_roll3",  "Increased passing volume over the last three games",
                             "Reduced passing volume over the last three games"),
    ("passing_yards_roll5",  "Strong passing baseline (last 5 games)",
                             "Modest passing baseline (last 5 games)"),
    ("passing_tds_roll5",    "Finding the end zone through the air lately",
                             "Few passing touchdowns recently"),
    ("passing_epa_roll5",    "Efficient passing recently (high EPA)",
                             "Inefficient passing recently (low EPA)"),
    ("attempts_roll3",       "High recent pass attempts",
                             "Low recent pass attempts"),
    ("carries_roll3",        "Heavy recent carry volume",
                             "Light recent carry volume"),
    ("carries_roll5",        "Consistent carry workload",
                             "Limited carry workload"),
    ("rushing_yards_roll3",  "Productive on the ground recently",
                             "Quiet rushing production recently"),
    ("targets_roll3",        "High target volume over the last three games",
                             "Low target volume over the last three games"),
    ("targets_roll5",        "Consistent target share",
                             "Inconsistent target volume"),
    ("target_share_roll5",   "High share of team targets",
                             "Low share of team targets"),
    ("air_yards_share_roll5", "Commands downfield targets (air-yard share)",
                              "Limited downfield role"),
    ("receptions_roll3",     "Catching passes at a high rate recently",
                             "Fewer receptions recently"),
    ("receiving_yards_roll3", "Strong recent receiving yardage",
                              "Modest recent receiving yardage"),
    ("offense_pct_roll3",    "High snap share recently",
                             "Reduced snap share last few weeks"),
    ("offense_pct_prev",     "On the field for most snaps last week",
                             "Limited snaps last week"),
    ("fg_made_roll5",        "Reliable recent field-goal production",
                             "Few field goals recently"),
    ("kicking_points_roll5", "Strong recent kicking production",
                             "Limited recent kicking production"),
    # Context
    ("implied_team_total",   "Team implied total is above average (good game script)",
                             "Team implied total is below average"),
    ("game_total",           "High game total (shootout potential)",
                             "Low game total"),
    ("spread_line_team",     "Team favored by the spread",
                             "Team an underdog by the spread"),
    ("opp_fantasy_allowed_pos", "Opponent allows above-average production to this position",
                                "Opponent stingy against this position"),
    ("opp_pass_yards_allowed", "Favorable passing matchup (opponent allows yards)",
                               "Tough passing matchup"),
    ("opp_rush_yards_allowed", "Favorable rushing matchup",
                               "Tough rushing matchup"),
    ("opp_rec_yards_allowed_pos", "Opponent gives up receiving yards to this position",
                                  "Opponent limits receiving production"),
    ("home_away",            "Home-field advantage",
                             "Playing on the road"),
    ("days_rest",            "Extra rest days",
                             "Short rest"),
    ("wind",                 "Strong wind may reduce passing and kicking production",
                             "Calm conditions"),
    ("temp",                 "Warm-weather game",
                             "Cold-weather game may affect production"),
    ("injury_designation_enc", "Injury designation increases uncertainty",
                               "No injury designation"),
    ("is_starter",           "Listed as a starter",
                             "Not listed as a clear starter"),
    ("years_experience",     "Veteran experience",
                             "Less experienced player"),
    ("bye_week_return",      "Returning from a bye/off week",
                             "Normal weekly cadence"),
]

_BINARY = {"home_away", "is_starter", "bye_week_return"}


def _label(feature: str, shap_val: float, feat_val: float) -> str | None:
    for pattern, pos, neg in FEATURE_LABELS:
        if pattern == feature:
            if feature in _BINARY:
                tmpl = pos if feat_val > 0.5 else neg
            elif feature == "injury_designation_enc":
                tmpl = pos if feat_val > 0.5 else neg
            elif feature == "wind":
                tmpl = pos if feat_val >= 15 else neg
            elif feature == "temp":
                tmpl = pos if feat_val >= 45 else neg
            else:
                tmpl = pos if shap_val > 0 else neg
            return tmpl
    return None


# ── Lazy, bounded model loading ─────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_feature_names() -> list[str]:
    path = MODELS_SAVED / "feature_names.json"
    return json.loads(path.read_text())


@lru_cache(maxsize=1)
def load_calibration() -> dict:
    """Per-(position,target) interval widening factors from evaluate.py. Absent
    file -> no widening (factor 1.0 everywhere)."""
    path = MODELS_SAVED / "calibration.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


@lru_cache(maxsize=6)                     # bounded: at most 6 positions cached
def load_position_bundle(position: str) -> dict:
    """Load and cache all models + metadata for one position. Bounded LRU keeps
    memory flat; the deployed API never loads every position at startup."""
    position = position.upper()
    pdir = MODELS_SAVED / position
    if not pdir.exists():
        raise FileNotFoundError(f"No trained models for position {position}")
    meta = json.loads((pdir / "metadata.json").read_text())
    models, q15, q85 = {}, {}, {}
    for target in POSITION_TARGETS.get(position, []):
        mp = pdir / f"{target}_model.pkl"
        if not mp.exists():
            continue
        with open(mp, "rb") as f:
            models[target] = pickle.load(f)
        for tag, store in (("q15", q15), ("q85", q85)):
            qp = pdir / f"{target}_{tag}.pkl"
            if qp.exists():
                with open(qp, "rb") as f:
                    store[target] = pickle.load(f)
    return {"models": models, "q15": q15, "q85": q85, "meta": meta}


def _clip(target: str, value: float) -> float:
    if target in NONNEGATIVE_TARGETS:
        return max(0.0, value)
    return value


def _shap_reasons(model, X: pd.DataFrame, feats: list[str],
                  n: int = 4) -> list[dict]:
    if not _SHAP_OK:
        return []
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        vals = vals[0] if getattr(vals, "ndim", 1) == 2 else vals
    except Exception as exc:
        log.warning("SHAP failed: %s", exc)
        return []
    row = X.iloc[0]
    pairs = sorted(zip(feats, vals, [row[f] for f in feats]),
                   key=lambda t: abs(t[1]), reverse=True)
    reasons, seen = [], set()
    for feat, sv, fv in pairs:
        if abs(sv) < 1e-4:
            break
        label = _label(feat, float(sv), float(fv))
        if not label or label in seen:
            continue
        seen.add(label)
        reasons.append({"direction": "+" if sv > 0 else "-",
                        "label": label, "feature": feat,
                        "impact": round(float(sv), 3)})
        if len(reasons) >= n:
            break
    return reasons


def predict_player(
    position: str,
    feature_row: pd.DataFrame,
    scoring_format: str = "ppr",
    fantasy_threshold: float | None = None,
    interval_widen: float = 1.0,
    explain: bool = True,
    simulate: bool = True,
) -> dict:
    """Full prediction bundle for one player-game feature row.

    Returns predictions per target, p15/p85 intervals, SHAP reasons per target,
    a merged football-language reasoning list, and the simulated fantasy-point
    projection/interval (kickers use kicker scoring, format-independent).

    ``interval_widen`` (>1) stretches every interval symmetrically around the
    point — used for team-changers, whose new-situation outcome is more uncertain.
    """
    position = position.upper()
    bundle = load_position_bundle(position)
    feats = load_feature_names()
    calib = load_calibration().get(position, {})
    X = feature_row.reindex(columns=feats).astype(float)
    X = X.fillna(0.0)

    predictions, intervals, reasons_by_target = {}, {}, {}
    for target, model in bundle["models"].items():
        pred = _clip(target, float(model.predict(X)[0]))
        predictions[target] = round(pred, 2)
        lo = hi = None
        if target in bundle["q15"] and target in bundle["q85"]:
            lo = _clip(target, float(bundle["q15"][target].predict(X)[0]))
            hi = float(bundle["q85"][target].predict(X)[0])
            hi = max(hi, lo + 0.1)
            lo, hi = min(lo, pred), max(hi, pred)
            # Widen by the holdout-calibrated factor (nominal ~70% coverage),
            # times any extra widen (e.g. team-changers' new-situation risk).
            k = float(calib.get(target, 1.0)) * max(1.0, float(interval_widen))
            if k > 1.0:
                lo = _clip(target, pred - k * (pred - lo))
                hi = pred + k * (hi - pred)
            intervals[target] = (round(lo, 2), round(hi, 2))
        reasons_by_target[target] = _shap_reasons(model, X, feats) if explain else []

    # Enforce cross-stat physical constraints on point predictions.
    if "attempts" in predictions and "completions" in predictions:
        predictions["completions"] = min(predictions["completions"],
                                         predictions["attempts"])
    if "targets" in predictions and "receptions" in predictions:
        predictions["receptions"] = min(predictions["receptions"],
                                        predictions["targets"])

    # Merge + dedupe reasons across targets into one ranked list.
    merged, seen = [], set()
    for target in POSITION_TARGETS.get(position, []):
        for r in reasons_by_target.get(target, []):
            if r["label"] in seen:
                continue
            seen.add(r["label"])
            merged.append(r)
    merged.sort(key=lambda r: abs(r["impact"]), reverse=True)

    # Fantasy points. The correlated simulation gives the interval + over/under
    # (never re-runs component models); ``simulate=False`` skips it for cheap
    # batch scoring where only the point estimate is needed.
    fantasy = {
        "scoring_format": scoring.resolve_scoring_format(scoring_format),
        "reception_multiplier": scoring.reception_multiplier(scoring_format),
        "label": scoring.SCORING_FORMAT_LABELS[
            scoring.resolve_scoring_format(scoring_format)] + " Fantasy Points",
        "point_estimate": round(scoring.fantasy_points(
            predictions, position, scoring_format), 2),
    }
    sim = None
    if simulate:
        sim = scoring.simulate_fantasy_points(
            predictions, intervals, position, scoring_format)
        fantasy["simulated_mean"] = round(sim["mean"], 2)
        fantasy["interval"] = (round(sim["low"], 2), round(sim["high"], 2))
        if fantasy_threshold is not None:
            fantasy["over_probability"] = round(
                scoring.over_under_probability(sim["samples"], fantasy_threshold, "over"), 3)
            fantasy["under_probability"] = round(
                scoring.over_under_probability(sim["samples"], fantasy_threshold, "under"), 3)
            fantasy["threshold"] = fantasy_threshold

    return {
        "position": position,
        "predictions": predictions,
        "intervals": intervals,
        "reasons": merged[:6],
        "reasons_by_target": reasons_by_target,
        "fantasy": fantasy,
        "_sim_samples": sim["samples"] if sim is not None else None,  # internal
    }
