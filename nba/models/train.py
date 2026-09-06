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

TARGETS = ["pts", "reb", "ast", "stl", "blk", "minutes", "fg3"]

# ── Recency weighting ─────────────────────────────────────────────────────────
# The models retrain continuously as new games arrive (pipeline/update.py), so a
# game played last week should count for more than one from 2022. We weight each
# training row by an exponential decay on its age: a game RECENCY_HALFLIFE_DAYS
# old counts half as much as tonight's. Full history is kept for stability — old
# rows are down-weighted, not dropped. Set to None to disable (equal weights).
RECENCY_HALFLIFE_DAYS = 365

# ── Per-stat hyperparameters ──────────────────────────────────────────────────
# Tuned with models/tune.py (5-fold TimeSeriesSplit CV). Game-level stats are
# noisy, so shallow trees (max_depth=3) generalise best across the board — see
# reports/tuning/*.png.
# High-signal stats (pts/ast):     depth 3, moderate regularisation, squared error
# Count stats (reb/stl/blk/fg3):   Poisson objective — natural for non-negative
#                                  counts, keeps predictions >= 0 and better
#                                  calibrates the low-count intervals.
# Minutes:                         depth 3, constrained (load-management variance)
PARAMS = {
    "pts": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 3,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
    },
    "reb": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 3,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
        "objective": "count:poisson",
    },
    "ast": {
        "n_estimators": 600, "learning_rate": 0.04, "max_depth": 3,
        "subsample": 0.8, "colsample_bytree": 0.75, "min_child_weight": 4,
        "reg_alpha": 0.1, "reg_lambda": 1.5, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 30,
    },
    "stl": {
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 20,
        "objective": "count:poisson",
    },
    "blk": {
        "n_estimators": 400, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.7, "colsample_bytree": 0.6, "min_child_weight": 10,
        "reg_alpha": 1.0, "reg_lambda": 3.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 20,
        "objective": "count:poisson",
    },
    "minutes": {
        "n_estimators": 500, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 8,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 25,
    },
    # Threes made — a non-negative count with more signal than stl/blk but far
    # noisier than pts. Poisson objective, moderate regularisation.
    "fg3": {
        "n_estimators": 450, "learning_rate": 0.03, "max_depth": 3,
        "subsample": 0.75, "colsample_bytree": 0.7, "min_child_weight": 6,
        "reg_alpha": 0.5, "reg_lambda": 2.0, "random_state": 42,
        "n_jobs": -1, "verbosity": 0, "early_stopping_rounds": 25,
        "objective": "count:poisson",
    },
}


def _load_tuned_params():
    """Override PARAMS with any Optuna-tuned configs (models/optuna_tune.py).

    Only stats present in tuned_params.json are replaced; the rest keep their
    hand-tuned defaults. This lets the tuner improve stats incrementally without
    touching the ones it couldn't beat.
    """
    path = MODELS_DIR / "tuned_params.json"
    if not path.exists():
        return
    try:
        tuned = json.loads(path.read_text())
    except Exception as exc:
        log.warning("Could not read tuned_params.json (%s) — using defaults", exc)
        return
    for stat, p in tuned.items():
        if stat in PARAMS and isinstance(p, dict):
            PARAMS[stat] = p
    if tuned:
        log.info("Loaded Optuna-tuned params for: %s", ", ".join(tuned))


_load_tuned_params()


def recency_weights(dates, ref_date=None):
    """Exponential-decay sample weights by game age.

    A game RECENCY_HALFLIFE_DAYS old gets weight 0.5 relative to `ref_date`
    (default: the most recent game). Returns a numpy array aligned to `dates`,
    or None when weighting is disabled.
    """
    if RECENCY_HALFLIFE_DAYS is None:
        return None
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    ref = pd.to_datetime(ref_date) if ref_date is not None else dates.max()
    age_days = (ref - dates).dt.days.clip(lower=0).to_numpy()
    return np.power(0.5, age_days / RECENCY_HALFLIFE_DAYS)


def load_data():
    log.info("Loading feature matrix...")
    df = pd.read_csv(PROCESSED / "features.csv", parse_dates=["game_date"])
    feature_names = (PROCESSED / "feature_names.txt").read_text().splitlines()
    feature_names = [f for f in feature_names if f in df.columns]
    log.info(f"  {len(df):,} rows, {len(feature_names)} features")
    return df, feature_names


def evaluate_cv(X, y, params, n_splits=5, sample_weight=None):
    """TimeSeriesSplit CV — always trains on past, tests on future."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s, best_rounds = [], [], [], []

    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        w_tr = None if sample_weight is None else sample_weight[train_idx]

        model = XGBRegressor(**model_params)
        if early_stop:
            model.set_params(early_stopping_rounds=early_stop)

        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], verbose=False)

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


def oof_predictions(X, y, params, n_splits=5, sample_weight=None):
    """Out-of-fold predictions via TimeSeriesSplit — each row is predicted by a
    model trained only on earlier rows, so the result is leakage-free and safe
    to feed as a feature into downstream models (stacking).

    Rows in the earliest fold never appear in a test split, so they come back as
    NaN; the caller fills those with a no-leakage proxy.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")

    oof = pd.Series(np.nan, index=X.index)
    for train_idx, test_idx in tscv.split(X):
        model = XGBRegressor(**model_params)
        if early_stop:
            model.set_params(early_stopping_rounds=early_stop)
        w_tr = None if sample_weight is None else sample_weight[train_idx]
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx], sample_weight=w_tr,
            eval_set=[(X.iloc[test_idx], y.iloc[test_idx])], verbose=False,
        )
        oof.iloc[test_idx] = np.clip(model.predict(X.iloc[test_idx]), 0, None)
    return oof


def train_final(X, y, params, sample_weight=None):
    """Train on full data with early stopping on held-out 15% slice."""
    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")

    model = XGBRegressor(**model_params)
    if early_stop:
        model.set_params(early_stopping_rounds=early_stop)

    split = int(len(X) * 0.85)
    w_tr = None if sample_weight is None else sample_weight[:split]
    model.fit(
        X.iloc[:split], y.iloc[:split], sample_weight=w_tr,
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        verbose=False,
    )
    return model


def train_quantile(X, y, params, alpha, sample_weight=None):
    """Train a quantile regressor (for prediction intervals). Uses the pinball
    ('reg:quantileerror') objective at the given alpha, e.g. 0.15 / 0.85 for a
    ~70% band. No early stopping — the quantile objective is trained for a fixed,
    capped number of rounds for stability.

    Note: the quantile (pinball) objective replaces any base objective such as
    count:poisson — that's intended; the point model keeps Poisson, the interval
    models use pinball loss.
    """
    p = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    p.update(objective="reg:quantileerror", quantile_alpha=alpha)
    p["n_estimators"] = min(p.get("n_estimators", 400), 400)
    model = XGBRegressor(**p)
    model.fit(X, y, sample_weight=sample_weight, verbose=False)
    return model


def top_features(model, feature_names, n=8):
    pairs = sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    return [[str(f), float(i)] for f, i in pairs[:n]]


# Counting stats that consume the projected-minutes feature (minutes drives them).
COUNTING = ["pts", "reb", "ast", "stl", "blk", "fg3"]


def train_all(df, feature_names):
    df = df.sort_values("game_date").reset_index(drop=True)

    # ── Recency weights ───────────────────────────────────────────────────────
    # One weight per row, decaying with game age. Aligned to the sorted df, so we
    # slice it with the same boolean mask used for X/y below.
    w_all = recency_weights(df["game_date"])
    if w_all is not None:
        log.info(
            "Recency weighting on (half-life %d days): weight range %.3f–%.3f",
            RECENCY_HALFLIFE_DAYS, float(w_all.min()), float(w_all.max()),
        )

    # ── Minutes stacking ──────────────────────────────────────────────────────
    # Minutes is the biggest driver of counting stats, so we give the counting
    # models a *projected* minutes feature. To keep it leakage-free we use
    # out-of-fold minutes predictions (a model never predicts a row it trained on).
    log.info("Generating out-of-fold minutes projections for stacking...")
    m_mask  = df["minutes"].notna()
    oof_min = oof_predictions(
        df.loc[m_mask, feature_names], df.loc[m_mask, "minutes"], PARAMS["minutes"],
        sample_weight=None if w_all is None else w_all[m_mask.to_numpy()],
    )
    # Early rows have no out-of-fold prediction — fall back to the player's
    # recent minutes (no leakage), then season average, then a neutral 24.
    df["pred_minutes"] = oof_min.reindex(df.index)
    df["pred_minutes"] = (
        df["pred_minutes"]
        .fillna(df.get("rolling_last5_minutes"))
        .fillna(df.get("season_avg_minutes"))
        .fillna(24.0)
    )
    log.info("  pred_minutes ready (mean %.1f)", df["pred_minutes"].mean())

    metadata = {
        "feature_names": feature_names,
        "stacked_feature": "pred_minutes",
        "counting_stats": COUNTING,
        "models": {},
    }

    log.info(f"\n{'='*55}\n Training {len(TARGETS)} XGBoost models\n{'='*55}")

    for target in TARGETS:
        if target not in df.columns:
            log.warning("Target '%s' not found — skipping", target)
            continue

        # Counting stats get pred_minutes appended to their feature set.
        feats = feature_names + (["pred_minutes"] if target in COUNTING else [])
        X = df[feats].copy()

        y    = df[target].copy()
        mask = y.notna()
        X_v, y_v = X[mask], y[mask]
        w_v = None if w_all is None else w_all[mask.to_numpy()]
        p = PARAMS[target]

        obj = p.get("objective", "reg:squarederror")
        log.info(f"\n── {target.upper()} {'─'*(48-len(target))}")
        log.info(f"   {len(y_v):,} rows | {len(feats)} feats | depth={p['max_depth']} | min_child={p['min_child_weight']} | obj={obj}")

        log.info("   Running 5-fold TimeSeriesSplit CV...")
        cv = evaluate_cv(X_v, y_v, p, sample_weight=w_v)
        log.info(f"   MAE: {cv['mae']:.3f} ± {cv['mae_std']:.3f} | RMSE: {cv['rmse']:.3f} | R²: {cv['r2']:.3f}")
        if "avg_best_round" in cv:
            log.info(f"   Avg early-stop round: {cv['avg_best_round']:.0f}")

        log.info("   Training final model...")
        model = train_final(X_v, y_v, p, sample_weight=w_v)

        path = MODELS_DIR / f"{target}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"   Saved -> {path.name}")

        # Quantile models for calibrated prediction intervals (~70% band).
        for alpha, tag in [(0.15, "q15"), (0.85, "q85")]:
            qm = train_quantile(X_v, y_v, p, alpha, sample_weight=w_v)
            with open(MODELS_DIR / f"{target}_{tag}.pkl", "wb") as f:
                pickle.dump(qm, f)
        log.info(f"   Saved -> {target}_q15.pkl, {target}_q85.pkl")

        top = top_features(model, feats)
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