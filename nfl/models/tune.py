"""
Lightweight hyperparameter tuning for NFL models.
=================================================
A small, chronological (walk-forward) search over the most impactful XGBoost
knobs (max_depth, learning_rate, min_child_weight) per (position, target). Writes
the best config to ``nfl/models/saved/tuned_params.json``; ``train.py`` picks it
up automatically if present (per (position,target)).

Never uses a random split — candidates are scored on the latest chronological
holdout, same as evaluate.py.

    python -m nfl.models.tune --position QB --target passing_yards
    python -m nfl.models.tune            # all positions/targets (slower)
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from nfl.config import POSITION_TARGETS, SUPPORTED_POSITIONS, MODELS_SAVED
from nfl.models.train import (
    load_features, recency_weights, _fit_point, _params_for,
)
from nfl.models.evaluate import _holdout_mask

log = logging.getLogger(__name__)

GRID = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.03, 0.05],
    "min_child_weight": [3, 6],
}
TUNED_PATH = MODELS_SAVED / "tuned_params.json"


def _candidates():
    keys = list(GRID)
    for combo in itertools.product(*(GRID[k] for k in keys)):
        yield dict(zip(keys, combo))


def tune_target(df, feats, position, target, holdout_frac=0.2) -> dict | None:
    sub = df[df["position"] == position].sort_values(["season", "week", "gameday"])
    if target not in sub.columns or len(sub) < 150:
        return None
    test_mask = _holdout_mask(sub, holdout_frac)
    y = pd.to_numeric(sub[target], errors="coerce")
    train, test = sub[~test_mask], sub[test_mask]
    ytr, yte = y[~test_mask], y[test_mask]
    mtr, mte = ytr.notna(), yte.notna()
    if mtr.sum() < 80 or mte.sum() < 15:
        return None
    Xtr, Xte = train[feats].astype(float), test[feats].astype(float)
    w = recency_weights(train["gameday"])
    w = None if w is None else w[mtr.to_numpy()]

    best, best_mae = None, np.inf
    base = _params_for(target)
    for cand in _candidates():
        params = dict(base, **cand)
        model = _fit_point(Xtr[mtr], ytr[mtr], params, w)
        pred = np.nan_to_num(np.clip(model.predict(Xte[mte]), 0, None))
        mae = mean_absolute_error(yte[mte].to_numpy(), pred)
        if mae < best_mae:
            best_mae, best = mae, params
    log.info("  %s/%s best MAE %.3f -> depth=%d lr=%.2f mcw=%d",
             position, target, best_mae, best["max_depth"],
             best["learning_rate"], best["min_child_weight"])
    return best


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--position")
    ap.add_argument("--target")
    args = ap.parse_args()

    df, feats = load_features()
    positions = [args.position.upper()] if args.position else SUPPORTED_POSITIONS
    tuned = {}
    if TUNED_PATH.exists():
        tuned = json.loads(TUNED_PATH.read_text())
    for pos in positions:
        targets = [args.target] if args.target else POSITION_TARGETS[pos]
        for target in targets:
            best = tune_target(df, feats, pos, target)
            if best:
                tuned[f"{pos}/{target}"] = best
    TUNED_PATH.write_text(json.dumps(tuned, indent=2))
    log.info("Wrote %s (%d entries)", TUNED_PATH.name, len(tuned))


if __name__ == "__main__":
    main()
