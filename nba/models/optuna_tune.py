"""
Optuna hyperparameter search
============================
Bayesian (TPE) search over the JOINT hyperparameter space for each stat,
scored with the exact same TimeSeriesSplit CV + recency weighting that
train.py uses — so a tuned config is directly comparable to the hand-tuned
PARAMS. This complements models/tune.py (one-parameter-at-a-time sweeps) by
finding interaction effects the manual sweep can't.

For count stats the base objective (count:poisson) is held fixed; only the
tree/regularisation knobs are searched.

Only stats whose best trial BEATS the current PARAMS baseline are written to
models/saved/tuned_params.json; train.py loads that file and overrides the
matching PARAMS entries. A stat that Optuna can't improve keeps its hand-tuned
config untouched.

Usage:
    python models/optuna_tune.py --stats pts,reb,ast,minutes,fg3 --trials 30
    python models/optuna_tune.py --stats all --trials 40
"""

import argparse
import json
import logging
from pathlib import Path

import optuna

from train import (
    load_data, evaluate_cv, recency_weights, PARAMS, TARGETS,
)

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "saved"
TUNED_PATH = MODELS_DIR / "tuned_params.json"

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Fixed keys carried onto every trial's params (not searched).
FIXED = {"random_state": 42, "n_jobs": -1, "verbosity": 0}


def _suggest(trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 300, 900, step=50),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
        "max_depth":        trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_int("min_child_weight", 2, 12),
        "subsample":        trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 4.0),
        "early_stopping_rounds": 30,
    }


def tune_stat(stat: str, n_trials: int, n_splits: int, df, feats):
    """Return (best_params, best_mae, baseline_mae) for one stat."""
    y = df[stat]
    mask = y.notna()
    X, yv = df.loc[mask, feats], y[mask]
    w = recency_weights(df["game_date"])
    wv = None if w is None else w[mask.to_numpy()]
    base_obj = PARAMS[stat].get("objective")   # keep e.g. count:poisson fixed

    def objective(trial):
        params = _suggest(trial)
        params.update(FIXED)
        if base_obj:
            params["objective"] = base_obj
        return evaluate_cv(X, yv, params, n_splits=n_splits, sample_weight=wv)["mae"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Baseline scored with the SAME n_splits for a fair comparison.
    baseline_mae = evaluate_cv(X, yv, PARAMS[stat], n_splits=n_splits,
                               sample_weight=wv)["mae"]

    best = dict(study.best_params)
    best.update(FIXED)
    best["early_stopping_rounds"] = 30
    if base_obj:
        best["objective"] = base_obj
    return best, study.best_value, baseline_mae


def main():
    ap = argparse.ArgumentParser(description="Optuna joint hyperparameter search.")
    ap.add_argument("--stats", default="pts,reb,ast,minutes,fg3",
                    help="Comma-separated stats, or 'all'.")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--splits", type=int, default=3,
                    help="TimeSeriesSplit folds during search (3 = faster).")
    args = ap.parse_args()

    stats = TARGETS if args.stats == "all" else args.stats.split(",")
    df, feats = load_data()

    existing = json.loads(TUNED_PATH.read_text()) if TUNED_PATH.exists() else {}
    improved = dict(existing)

    for stat in stats:
        if stat not in df.columns:
            log.warning("skip %s (not in data)", stat); continue
        log.info("── Tuning %s (%d trials, %d-fold CV) ──", stat, args.trials, args.splits)
        best, best_mae, base_mae = tune_stat(stat, args.trials, args.splits, df, feats)
        delta = base_mae - best_mae
        verdict = "IMPROVED" if delta > 0 else "no gain"
        log.info("   baseline MAE %.4f | best MAE %.4f | Δ %+.4f → %s",
                 base_mae, best_mae, -delta if delta < 0 else delta, verdict)
        if delta > 0:
            improved[stat] = best
            log.info("   best params: %s",
                     {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in best.items() if k not in FIXED})

    if improved != existing:
        TUNED_PATH.write_text(json.dumps(improved, indent=2))
        log.info("Wrote %d tuned stat(s) -> %s", len(improved), TUNED_PATH.name)
    else:
        log.info("No improvements over current PARAMS — nothing written.")


if __name__ == "__main__":
    main()
