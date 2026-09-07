"""
Backtest Evaluation Harness
===========================
Trains on the earliest games and tests on the most recent ones (a single
chronological holdout — never train on the future), then reports per-stat
accuracy and prediction-interval calibration.

Why a separate holdout from train.py's CV:
    train.py uses 5-fold TimeSeriesSplit to *tune* each model. This script gives
    one clean, comparable number per stat on a fixed recent slice, so we can
    measure whether a change (new features, minutes stacking, etc.) actually
    helped. Run it before and after a change and diff the tables.

What it reports per stat:
    MAE, RMSE, R^2                  — point accuracy
    interval coverage + mean width  — are the p15-p85 bands calibrated?
                                      (coverage should land near 70%)

Usage:
    python models/evaluate.py                 # 15% holdout, with intervals
    python models/evaluate.py --holdout 0.2   # bigger test slice
    python models/evaluate.py --stack-minutes # feed OOF minutes into counting stats
    python models/evaluate.py --no-intervals  # skip quantile coverage (faster)

Output:
    Prints a comparison table and writes models/saved/eval_report.json
    (appended to a dated history list so accuracy drift is visible over time).
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Reuse the canonical params + loaders from train.py
from train import PARAMS, TARGETS, load_data, recency_weights

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models" / "saved"
REPORT     = MODELS_DIR / "eval_report.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Fallback interval half-widths (mirrors api/main.py MIN_STD) when a stat has
# too little data for a stable quantile fit.
MIN_STD = {"pts": 4.0, "reb": 2.0, "ast": 1.5,
           "stl": 0.5, "blk": 0.4, "minutes": 3.0}

# Counting stats that receive the projected-minutes feature under --stack-minutes.
COUNTING = ["pts", "reb", "ast", "stl", "blk", "fg3"]


def _fit(params: dict, X_tr, y_tr, X_val, y_val, sample_weight=None) -> XGBRegressor:
    """Fit one XGBRegressor with early stopping on a validation slice."""
    model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    early_stop   = params.get("early_stopping_rounds")
    model = XGBRegressor(**model_params)
    if early_stop:
        model.set_params(early_stopping_rounds=early_stop)
    model.fit(X_tr, y_tr, sample_weight=sample_weight,
              eval_set=[(X_val, y_val)], verbose=False)
    return model


def _fit_quantile(base_params: dict, alpha: float, X_tr, y_tr,
                  sample_weight=None) -> XGBRegressor:
    """Fit a quantile regressor for the given alpha (no early stopping — the
    quantile objective doesn't pair cleanly with an eval_set here)."""
    p = {k: v for k, v in base_params.items() if k != "early_stopping_rounds"}
    p.update(objective="reg:quantileerror", quantile_alpha=alpha)
    # A touch shallower / fewer trees keeps the quantile fit stable and fast.
    p["n_estimators"] = min(p.get("n_estimators", 400), 400)
    model = XGBRegressor(**p)
    model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
    return model


def backtest(holdout: float, intervals: bool, stack_minutes: bool) -> dict:
    df, feature_names = load_data()
    df = df.sort_values("game_date").reset_index(drop=True)

    split = int(len(df) * (1 - holdout))
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    log.info(
        "Holdout %.0f%%  |  train %d rows (%s..%s)  |  test %d rows (%s..%s)",
        holdout * 100, len(train_df),
        str(train_df["game_date"].min())[:10], str(train_df["game_date"].max())[:10],
        len(test_df),
        str(test_df["game_date"].min())[:10], str(test_df["game_date"].max())[:10],
    )

    # Inner validation slice (last 15% of the TRAIN block) for early stopping.
    inner = int(len(train_df) * 0.85)

    results = {}
    pred_minutes_train = pred_minutes_test = None

    # Minutes first so counting stats can consume its projection when stacking.
    order = ["minutes"] + [t for t in TARGETS if t != "minutes"]

    for target in order:
        if target not in df.columns:
            continue

        feats = list(feature_names)
        if stack_minutes and target in COUNTING and pred_minutes_train is not None:
            feats = feats + ["pred_minutes"]

        # Assemble X for train/test, injecting pred_minutes if present.
        def _make_X(block, pm):
            X = block[feature_names].copy()
            if "pred_minutes" in feats:
                X["pred_minutes"] = pm
            return X

        y_tr_full = train_df[target]
        mask_tr   = y_tr_full.notna()
        X_tr_full = _make_X(train_df, pred_minutes_train)[mask_tr.values]
        y_tr_full = y_tr_full[mask_tr]

        # Recency weights, aligned to the (masked) train block — mirrors the
        # weighting train.py applies to the production models.
        w_full = recency_weights(train_df["game_date"])
        w_tr_full = None if w_full is None else w_full[mask_tr.values]

        X_tr, X_val = X_tr_full.iloc[:inner], X_tr_full.iloc[inner:]
        y_tr, y_val = y_tr_full.iloc[:inner], y_tr_full.iloc[inner:]
        w_tr = None if w_tr_full is None else w_tr_full[:inner]

        y_te   = test_df[target]
        mask_te = y_te.notna()
        X_te   = _make_X(test_df, pred_minutes_test)[mask_te.values]
        y_te   = y_te[mask_te]

        model = _fit(PARAMS[target], X_tr, y_tr, X_val, y_val, sample_weight=w_tr)
        preds = np.clip(model.predict(X_te), 0, None)

        row = {
            "mae":  float(mean_absolute_error(y_te, preds)),
            "rmse": float(np.sqrt(mean_squared_error(y_te, preds))),
            "r2":   float(r2_score(y_te, preds)),
            "n_test": int(len(y_te)),
        }

        if intervals:
            try:
                lo = _fit_quantile(PARAMS[target], 0.15, X_tr_full, y_tr_full,
                                   sample_weight=w_tr_full).predict(X_te)
                hi = _fit_quantile(PARAMS[target], 0.85, X_tr_full, y_tr_full,
                                   sample_weight=w_tr_full).predict(X_te)
                lo = np.clip(lo, 0, None)
                hi = np.maximum(hi, lo + 1e-6)
                inside = ((y_te.values >= lo) & (y_te.values <= hi)).mean()
                row["coverage"] = float(inside)
                row["interval_width"] = float(np.mean(hi - lo))
            except Exception as exc:
                log.warning("Quantile fit failed for %s: %s", target, exc)

        results[target] = row
        log.info(
            "  %-8s MAE %.3f  RMSE %.3f  R2 %.3f%s",
            target, row["mae"], row["rmse"], row["r2"],
            f"  cov {row['coverage']*100:.0f}% w{row['interval_width']:.1f}"
            if "coverage" in row else "",
        )

        # Cache minutes projections for the counting-stat models.
        if target == "minutes" and stack_minutes:
            pred_minutes_train = np.clip(model.predict(_make_X(train_df, None)), 0, None)
            pred_minutes_test  = np.clip(model.predict(_make_X(test_df, None)), 0, None)

    return results


def print_table(results: dict) -> None:
    print(f"\n{'='*72}\n Backtest results (chronological holdout)\n{'='*72}")
    header = f"  {'Stat':<9}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Coverage':>10}{'Width':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stat, r in results.items():
        cov = f"{r['coverage']*100:.0f}%" if "coverage" in r else "-"
        wid = f"{r['interval_width']:.1f}" if "interval_width" in r else "-"
        print(f"  {stat:<9}{r['mae']:>8.3f}{r['rmse']:>8.3f}{r['r2']:>8.3f}{cov:>10}{wid:>8}")
    print(f"{'='*72}\n")


def save_report(results: dict, args) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout": args.holdout,
        "stack_minutes": args.stack_minutes,
        "results": results,
    }
    history = []
    if REPORT.exists():
        try:
            history = json.loads(REPORT.read_text())
            if isinstance(history, dict):   # migrate an older single-object file
                history = [history]
        except Exception:
            history = []
    history.append(entry)
    REPORT.write_text(json.dumps(history[-50:], indent=2))
    log.info("Report appended -> %s (%d runs kept)", REPORT.name, len(history[-50:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest the stat models on a recent holdout.")
    ap.add_argument("--holdout", type=float, default=0.15,
                    help="Fraction of most-recent games to hold out (default 0.15).")
    ap.add_argument("--stack-minutes", action="store_true",
                    help="Feed projected minutes into the counting-stat models.")
    ap.add_argument("--no-intervals", dest="intervals", action="store_false",
                    help="Skip quantile interval coverage (faster).")
    args = ap.parse_args()

    results = backtest(args.holdout, args.intervals, args.stack_minutes)
    print_table(results)
    save_report(results, args)
