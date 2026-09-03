"""
Chronological evaluation of NFL models.
=======================================
Holds out the latest completed portion of the dataset (by season+week) and
trains only on earlier rows — never a random split. For every (position, target)
reports MAE, RMSE, R^2, a naive baseline MAE (a shifted rolling/season average),
improvement over baseline, p15-p85 interval coverage, row counts, date ranges,
and top features. Targets that fail to beat their baseline are flagged.

Because all engineered features are leakage-safe (each row only sees prior
weeks), a simple chronological split is a valid walk-forward test: the holdout
weeks never influenced the train rows' features, and models are re-fit on the
train split only here.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from nfl.config import (
    POSITION_TARGETS, SUPPORTED_POSITIONS, QUANTILE_LOW, QUANTILE_HIGH,
    MODELS_SAVED,
)
from nfl.models.train import (
    load_features, recency_weights, _fit_point, _fit_quantile, _params_for,
)

log = logging.getLogger(__name__)

# Baseline: the shifted 5-game rolling average of the same stat (a reasonable,
# no-model projection). Falls back to season average, then previous value.
BASELINE_SUFFIXES = ["_roll5", "_season_avg", "_prev"]


def _baseline_series(sub: pd.DataFrame, target: str) -> pd.Series:
    for suf in BASELINE_SUFFIXES:
        col = f"{target}{suf}"
        if col in sub.columns:
            return pd.to_numeric(sub[col], errors="coerce")
    return pd.Series(np.nan, index=sub.index)


def _holdout_mask(sub: pd.DataFrame, holdout_frac: float) -> pd.Series:
    """Latest `holdout_frac` of rows by (season, week) become the test set."""
    order = sub.sort_values(["season", "week"]).index
    n_test = max(1, int(len(order) * holdout_frac))
    test_idx = order[-n_test:]
    return sub.index.isin(test_idx)


def evaluate_position(df: pd.DataFrame, feats: list[str], position: str,
                      holdout_frac: float = 0.2) -> dict:
    sub = df[df["position"] == position].copy()
    sub = sub.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
    if len(sub) < 100:
        return {}

    test_mask = _holdout_mask(sub, holdout_frac)
    train, test = sub[~test_mask], sub[test_mask]
    weights = recency_weights(train["gameday"])
    Xtr = train[feats].astype(float)
    Xte = test[feats].astype(float)

    results = {}
    for target in POSITION_TARGETS[position]:
        if target not in sub.columns:
            continue
        ytr = pd.to_numeric(train[target], errors="coerce")
        yte = pd.to_numeric(test[target], errors="coerce")
        m_tr, m_te = ytr.notna(), yte.notna()
        if m_tr.sum() < 50 or m_te.sum() < 10:
            continue
        params = _params_for(target, position)
        w = None if weights is None else weights[m_tr.to_numpy()]

        point = _fit_point(Xtr[m_tr], ytr[m_tr], params, w)
        q15 = _fit_quantile(Xtr[m_tr], ytr[m_tr], params, QUANTILE_LOW, w)
        q85 = _fit_quantile(Xtr[m_tr], ytr[m_tr], params, QUANTILE_HIGH, w)

        pred = np.nan_to_num(np.clip(point.predict(Xte[m_te]), 0, None))
        yt = yte[m_te].to_numpy()
        mae = mean_absolute_error(yt, pred)
        rmse = float(np.sqrt(mean_squared_error(yt, pred)))
        r2 = float(r2_score(yt, pred)) if len(set(yt)) > 1 else float("nan")

        base = _baseline_series(test, target)[m_te]
        fill = base.mean()
        if not np.isfinite(fill):
            fill = float(ytr[m_tr].mean())
        base = base.fillna(fill).to_numpy()
        base_mae = mean_absolute_error(yt, base)
        improvement = (base_mae - mae) / base_mae if base_mae > 0 else 0.0

        lo = np.clip(q15.predict(Xte[m_te]), 0, None)
        hi = q85.predict(Xte[m_te])
        coverage = float(np.mean((yt >= lo) & (yt <= hi)))

        importances = sorted(zip(feats, point.feature_importances_),
                             key=lambda t: t[1], reverse=True)
        results[target] = {
            "mae": round(float(mae), 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3) if not np.isnan(r2) else None,
            "baseline_mae": round(float(base_mae), 3),
            "improvement_over_baseline": round(float(improvement), 3),
            "beats_baseline": bool(mae < base_mae),
            "interval_coverage_p15_p85": round(coverage, 3),
            "n_train": int(m_tr.sum()),
            "n_eval": int(m_te.sum()),
            "top_features": [[str(f), round(float(i), 4)] for f, i in importances[:6]],
        }
        flag = "" if mae < base_mae else "  <-- FAILS BASELINE"
        log.info("  %s/%-22s MAE %.2f (base %.2f, %+.0f%%) cov %.0f%%%s",
                 position, target, mae, base_mae, improvement * 100,
                 coverage * 100, flag)

    return {
        "position": position,
        "train_date_range": [str(train["gameday"].min())[:10],
                             str(train["gameday"].max())[:10]],
        "holdout_date_range": [str(test["gameday"].min())[:10],
                              str(test["gameday"].max())[:10]],
        "targets": results,
    }


def evaluate_all(holdout_frac: float = 0.2) -> dict:
    df, feats = load_features()
    report = {"holdout_frac": holdout_frac, "positions": {}, "failed_baselines": []}
    for pos in SUPPORTED_POSITIONS:
        res = evaluate_position(df, feats, pos, holdout_frac)
        if not res:
            continue
        report["positions"][pos] = res
        for target, m in res["targets"].items():
            if not m["beats_baseline"]:
                report["failed_baselines"].append(f"{pos}/{target}")

    with open(MODELS_SAVED / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    if report["failed_baselines"]:
        log.info("Targets failing baseline: %s", report["failed_baselines"])
    else:
        log.info("All evaluated targets beat their baseline.")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    evaluate_all()
