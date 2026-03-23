"""
Train XGBoost Models
=====================
Trains one XGBoost model per target stat with per-stat hyperparameters,
early stopping, and TimeSeriesSplit cross-validation.

Input:  data/processed/features.csv
        data/processed/feature_names.txt

Output: models/saved/{stat}_model.pkl  (one per stat)
        models/saved/training_metadata.json

Usage:
    python models/train.py
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT        = Path(__file__).parent.parent
PROCESSED   = ROOT / "data" / "processed"
MODELS_DIR  = ROOT / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TARGETS = ["pts", "reb", "ast", "stl", "blk", "minutes"]

# ── Per-stat hyperparameters ──────────────────────────────────────────────────
# High-signal stats (pts/reb/ast): deeper trees, moderate regularisation
# Low-signal stats (stl/blk):     shallow trees, heavy regularisation
# Minutes:                         constrained — high variance from load mgmt
PARAMS = {
    "pts": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
    },
    "reb": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
    },
    "ast": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
    },
    "stl": {
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 20,
    },
    "blk": {
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 20,
    },
    "minutes": {
        "n_estimators": 500, "learning_rate": 0.03, "max_depth": 4,
        "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 8,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 25,
    },
}


def load_data():
    log.info("Loading feature matrix...")
    df = pd.read_csv(PROCESSED / "features.csv", parse_dates=["game_date"])
    feature_names = (PROCESSED / "feature_names.txt").read_text().splitlines()
    feature_names = [f for f in feature_names if f in df.columns]
    log.info(f"  {len(df):,} rows, {len(feature_names)} features")
    return df, feature_names


def evaluate_cv(X, y, params, n_splits=5):
    """TimeSeriesSplit CV — always trains on past, tests on future."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s, best_rounds = [], [], [], []

    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor(**model_params)
        if early_stop:
            model.set_params(early_stopping_rounds=early_stop)

        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

        preds = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(np.sqrt(mean_squared_error(y_te, preds)))
        r2s.append(r2_score(y_te, preds))
        if hasattr(model, "best_iteration"):
            best_rounds.append(model.best_iteration)

    result = {
        "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses)),
        "r2": float(np.mean(r2s)), "mae_std": float(np.std(maes)),
    }
    if best_rounds:
        result["avg_best_round"] = float(np.mean(best_rounds))
    return result


def train_final(X, y, params):
    """Train on full data with early stopping on held-out 15% slice."""
    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")

    model = XGBRegressor(**model_params)
    if early_stop:
        model.set_params(early_stopping_rounds=early_stop)

    split = int(len(X) * 0.85)
    model.fit(
        X.iloc[:split], y.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        verbose=False,
    )
    return model


def top_features(model, feature_names, n=8):
    pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    return [[str(f), float(i)] for f, i in pairs[:n]]


def train_all(df, feature_names):
    df = df.sort_values("game_date").reset_index(drop=True)
    X = df[feature_names].copy()

    metadata = {"feature_names": feature_names, "models": {}}

    log.info(f"\n{'='*55}\n Training {len(TARGETS)} XGBoost models\n{'='*55}")

    for target in TARGETS:
        if target not in df.columns:
            log.warning("Target '%s' not found — skipping", target)
            continue

        y    = df[target].copy()
        mask = y.notna()
        X_v, y_v = X[mask], y[mask]
        p = PARAMS[target]

        log.info(f"\n── {target.upper()} {'─'*(48-len(target))}")
        log.info(f"   {len(y_v):,} rows | depth={p['max_depth']} | min_child={p['min_child_weight']}")

        log.info("   Running 5-fold TimeSeriesSplit CV...")
        cv = evaluate_cv(X_v, y_v, p)
        log.info(f"   MAE: {cv['mae']:.3f} ± {cv['mae_std']:.3f} | RMSE: {cv['rmse']:.3f} | R²: {cv['r2']:.3f}")
        if "avg_best_round" in cv:
            log.info(f"   Avg early-stop round: {cv['avg_best_round']:.0f}")

        log.info("   Training final model...")
        model = train_final(X_v, y_v, p)

        path = MODELS_DIR / f"{target}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"   Saved -> {path.name}")

        top = top_features(model, feature_names)
        log.info("   Top features:")
        for feat, imp in top:
            log.info(f"     {feat:42s}: {imp:.4f}")

        metadata["models"][target] = {
            "cv_mae": cv["mae"], "cv_mae_std": cv["mae_std"],
            "cv_rmse": cv["rmse"], "cv_r2": cv["r2"],
            "top_features": top,
            "n_training_rows": int(len(y_v)),
        }

    meta_path = MODELS_DIR / "training_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"\nMetadata saved -> {meta_path.name}")
    return metadata


def print_summary(metadata):
    benchmarks = {
        "pts": (5.0, 3.5), "reb": (2.5, 1.8), "ast": (2.0, 1.4),
        "stl": (0.6, 0.4), "blk": (0.5, 0.35), "minutes": (4.0, 2.5),
    }
    print(f"\n{'='*60}\n Model training summary\n{'='*60}")
    print(f"  {'Stat':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8}  Verdict")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")
    for target, scores in metadata["models"].items():
        mae, rmse, r2 = scores["cv_mae"], scores["cv_rmse"], scores["cv_r2"]
        if target in benchmarks:
            good, great = benchmarks[target]
            verdict = "great" if mae <= great else ("good" if mae <= good else "needs data")
        else:
            verdict = ""
        print(f"  {target:<10} {mae:>8.3f} {rmse:>8.3f} {r2:>8.3f}  {verdict}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    df, feature_names = load_data()
    metadata = train_all(df, feature_names)
    print_summary(metadata)