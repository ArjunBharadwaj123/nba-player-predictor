"""
Train pooled, position-specific NFL models.
===========================================
One model per (position, target) pair — pooled across all players at that
position (NFL players have too few games for per-player models). For every pair
we train:
  * a point model (Poisson objective for non-negative counts, squared-error for
    continuous yardage),
  * two quantile models (p15 / p85) for calibrated prediction intervals.

Recency sample weighting down-weights older seasons. Final models fit on a
chronological order with early stopping on the most recent slice (never random).

Artifacts (per position):
    nfl/models/saved/<POS>/<target>_model.pkl
    nfl/models/saved/<POS>/<target>_q15.pkl
    nfl/models/saved/<POS>/<target>_q85.pkl
    nfl/models/saved/<POS>/metadata.json
Global:
    nfl/models/saved/feature_names.json
    nfl/models/saved/training_metadata.json
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from nfl.config import (
    POSITION_TARGETS, SUPPORTED_POSITIONS, COUNT_TARGETS, NONNEGATIVE_TARGETS,
    QUANTILE_LOW, QUANTILE_HIGH, RECENCY_HALFLIFE_DAYS, MODELS_SAVED,
    DATA_PROCESSED, default_seasons, is_dev_mode,
)
from nfl.features.engineer import feature_columns

log = logging.getLogger(__name__)

FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"

# Per-target-family hyperparameters. Kept modest — NFL weekly data is noisy and
# pooled, so shallow trees generalize best (mirrors the NBA project's findings).
BASE_PARAMS = dict(
    n_estimators=400, learning_rate=0.04, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.2, reg_lambda=1.5, random_state=42, n_jobs=-1, verbosity=0,
)


def _load_tuned() -> dict:
    path = MODELS_SAVED / "tuned_params.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


_TUNED = _load_tuned()


def _params_for(target: str, position: str | None = None) -> dict:
    p = dict(BASE_PARAMS)
    if target in COUNT_TARGETS:
        p["objective"] = "count:poisson"
    else:
        p["objective"] = "reg:squarederror"
    # Override with tuned params for this (position,target) when available.
    key = f"{position}/{target}" if position else None
    if key and key in _TUNED:
        p.update({k: v for k, v in _TUNED[key].items()})
    return p


def recency_weights(dates: pd.Series) -> np.ndarray | None:
    if RECENCY_HALFLIFE_DAYS is None:
        return None
    dates = pd.to_datetime(dates).reset_index(drop=True)
    ref = dates.max()
    age = (ref - dates).dt.days.clip(lower=0).to_numpy()
    return np.power(0.5, age / RECENCY_HALFLIFE_DAYS)


def _fit_point(X, y, params, weights=None):
    split = max(1, int(len(X) * 0.85))
    model = XGBRegressor(**params, early_stopping_rounds=30)
    w_tr = None if weights is None else weights[:split]
    model.fit(
        X.iloc[:split], y.iloc[:split], sample_weight=w_tr,
        eval_set=[(X.iloc[split:], y.iloc[split:])], verbose=False,
    )
    return model


def _fit_quantile(X, y, params, alpha, weights=None):
    p = {k: v for k, v in params.items() if k != "objective"}
    p.update(objective="reg:quantileerror", quantile_alpha=alpha,
             n_estimators=min(p.get("n_estimators", 400), 350))
    model = XGBRegressor(**p)
    model.fit(X, y, sample_weight=weights, verbose=False)
    return model


def load_features() -> tuple[pd.DataFrame, list[str]]:
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{FEATURES_PARQUET} not found. Run `python -m nfl.pipeline.update` "
            "(or nfl.pipeline.update.build_features) first."
        )
    df = pd.read_parquet(FEATURES_PARQUET)
    feats = feature_columns(df)
    return df, feats


def train_position(df: pd.DataFrame, feats: list[str], position: str) -> dict:
    sub = df[df["position"] == position].copy()
    sub = sub.sort_values("gameday").reset_index(drop=True)
    if sub.empty:
        log.warning("No rows for %s — skipping", position)
        return {}

    weights = recency_weights(sub["gameday"])
    out_dir = MODELS_SAVED / position
    out_dir.mkdir(parents=True, exist_ok=True)

    X = sub[feats].astype(float)
    pos_meta = {
        "position": position,
        "n_rows": int(len(sub)),
        "targets": {},
        "train_date_min": str(sub["gameday"].min())[:10],
        "train_date_max": str(sub["gameday"].max())[:10],
    }

    for target in POSITION_TARGETS[position]:
        if target not in sub.columns:
            log.warning("  %s/%s: target column missing — skipping", position, target)
            continue
        y = pd.to_numeric(sub[target], errors="coerce")
        mask = y.notna()
        Xv, yv = X[mask], y[mask]
        w = None if weights is None else weights[mask.to_numpy()]
        if len(yv) < 50:
            log.warning("  %s/%s: only %d rows — skipping", position, target, len(yv))
            continue
        params = _params_for(target, position)

        point = _fit_point(Xv, yv, params, w)
        q15 = _fit_quantile(Xv, yv, params, QUANTILE_LOW, w)
        q85 = _fit_quantile(Xv, yv, params, QUANTILE_HIGH, w)

        with open(out_dir / f"{target}_model.pkl", "wb") as f:
            pickle.dump(point, f)
        with open(out_dir / f"{target}_q15.pkl", "wb") as f:
            pickle.dump(q15, f)
        with open(out_dir / f"{target}_q85.pkl", "wb") as f:
            pickle.dump(q85, f)

        # In-sample residual std (a serving fallback for the simulator when a
        # quantile interval is degenerate). Not an accuracy claim.
        resid = yv - np.clip(point.predict(Xv), 0, None)
        importances = sorted(
            zip(feats, point.feature_importances_), key=lambda t: t[1], reverse=True)
        pos_meta["targets"][target] = {
            "n_rows": int(len(yv)),
            "objective": params["objective"],
            "residual_std": float(np.std(resid)),
            "y_mean": float(yv.mean()),
            "top_features": [[str(f), float(i)] for f, i in importances[:8]],
        }
        log.info("  %s/%-22s trained on %d rows (obj=%s)",
                 position, target, len(yv), params["objective"])

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(pos_meta, f, indent=2)
    return pos_meta


def train_all(positions: list[str] | None = None) -> dict:
    df, feats = load_features()
    with open(MODELS_SAVED / "feature_names.json", "w") as f:
        json.dump(feats, f, indent=2)

    positions = positions or SUPPORTED_POSITIONS
    metadata = {
        "mode": "dev" if is_dev_mode() else "production",
        "seasons": default_seasons(),
        "n_features": len(feats),
        "positions": {},
    }
    log.info("Training %d positions | %d features | mode=%s",
             len(positions), len(feats), metadata["mode"])
    for pos in positions:
        metadata["positions"][pos] = train_position(df, feats, pos)

    with open(MODELS_SAVED / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Training complete. Artifacts in %s", MODELS_SAVED)
    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", help="Train only this position")
    args = ap.parse_args()
    train_all([args.position.upper()] if args.position else None)
