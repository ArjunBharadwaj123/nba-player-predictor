"""
Hyperparameter Sweep + Graphs
=============================
Sweep ONE hyperparameter for ONE stat at a time, holding the others at their
current train.py values, and plot how MAE and R^2 respond. This turns
hyperparameter choices into evidence you can look at instead of guesses.

Each config is scored with the exact 5-fold TimeSeriesSplit CV that train.py
uses (via `evaluate_cv`), so the numbers are directly comparable to training.

Usage:
    python models/tune.py --stat pts --param max_depth      # one sweep
    python models/tune.py --stat reb --param learning_rate
    python models/tune.py --stat minutes --param n_estimators --values 300,600,900,1200
    python models/tune.py --stat all --param all            # calibrate EVERYTHING
    python models/tune.py --stat pts --param all            # all params for one model

Output:
    - Prints a results table per sweep (best MAE / R² highlighted) and, when
      sweeping multiple params, a per-model best-by-param summary.
    - Saves a dual-panel MAE / R^2 chart with the value labelled at every point
      to reports/tuning/{stat}/{stat}_{param}.png  (one folder per model).

Workflow: run a sweep -> read the graph -> update PARAMS in train.py -> retrain.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless — just write PNGs
import matplotlib.pyplot as plt
import numpy as np

from train import PARAMS, evaluate_cv, load_data, oof_predictions, COUNTING

ROOT     = Path(__file__).parent.parent
OUT_DIR  = ROOT / "reports" / "tuning"

# Sensible default grids per hyperparameter.
DEFAULT_GRIDS = {
    "max_depth":        [2, 3, 4, 5, 6, 7, 8],
    "learning_rate":    [0.01, 0.02, 0.04, 0.08, 0.12, 0.2],
    "n_estimators":     [200, 300, 400, 600, 800, 1000],
    "min_child_weight": [1, 2, 4, 6, 8, 10, 15],
    "subsample":        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":        [0.0, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda":       [0.5, 1.0, 1.5, 3.0, 5.0],
}


def sweep(stat: str, param: str, values: list, stack_minutes: bool):
    df, feature_names = load_data()
    df = df.sort_values("game_date").reset_index(drop=True)

    feats = list(feature_names)
    if stack_minutes and stat in COUNTING:
        # Reproduce the training-time projected-minutes feature (leakage-free).
        oof = oof_predictions(df[feature_names], df["minutes"], PARAMS["minutes"])
        df["pred_minutes"] = (
            oof.fillna(df.get("rolling_last5_minutes"))
               .fillna(df.get("season_avg_minutes")).fillna(24.0)
        )
        feats = feats + ["pred_minutes"]

    y    = df[stat]
    mask = y.notna()
    X, y = df.loc[mask, feats], y[mask]

    base = dict(PARAMS[stat])
    rows = []
    print(f"\nSweeping {stat.upper()} · {param}  (holding other params at train.py defaults)")
    print(f"  {'value':>12}{'MAE':>9}{'RMSE':>9}{'R2':>9}")
    print("  " + "-" * 39)
    for v in values:
        params = dict(base)
        params[param] = v
        cv = evaluate_cv(X, y, params)
        rows.append({"value": v, "mae": cv["mae"], "rmse": cv["rmse"], "r2": cv["r2"]})
        print(f"  {str(v):>12}{cv['mae']:>9.3f}{cv['rmse']:>9.3f}{cv['r2']:>9.3f}")

    best = min(rows, key=lambda r: r["mae"])
    cur  = base.get(param)
    print(f"\n  Best MAE at {param}={best['value']} "
          f"(MAE {best['mae']:.3f}, R2 {best['r2']:.3f})   "
          f"| current train.py value: {param}={cur}\n")
    return rows, best, cur


def plot(stat, param, rows, best, cur):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xs   = [r["value"] for r in rows]
    maes = [r["mae"] for r in rows]
    r2s  = [r["r2"] for r in rows]
    xpos = range(len(xs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.suptitle(f"{stat.upper()} — sensitivity to {param}", fontweight="bold")

    def _annotate(ax, ys, fmt, color, dy):
        for x, y in zip(xpos, ys):
            ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color=color,
                        fontweight="bold")

    # ── MAE panel ──
    ax1.plot(xpos, maes, "o-", color="#C1462B", lw=2)
    _annotate(ax1, maes, "{:.3f}", "#8A2B18", 9)
    ax1.set_ylabel("CV MAE  (lower is better)")
    ax1.set_xlabel(param)
    ax1.set_xticks(list(xpos)); ax1.set_xticklabels([str(x) for x in xs])
    ax1.grid(alpha=0.25)
    ax1.margins(y=0.18)
    ax1.axvline(xs.index(best["value"]), color="#0F6E56", ls="--", lw=1,
                label=f"best MAE ({best['value']})")
    if cur in xs:
        ax1.axvline(xs.index(cur), color="#888", ls=":", lw=1, label=f"current ({cur})")
    ax1.legend(fontsize=8, loc="best")

    # ── R² panel ──
    ax2.plot(xpos, r2s, "o-", color="#185FA5", lw=2)
    _annotate(ax2, r2s, "{:.3f}", "#0C447C", 9)
    ax2.set_ylabel("CV R²  (higher is better)")
    ax2.set_xlabel(param)
    ax2.set_xticks(list(xpos)); ax2.set_xticklabels([str(x) for x in xs])
    ax2.grid(alpha=0.25)
    ax2.margins(y=0.18)
    best_r2 = max(rows, key=lambda r: r["r2"])
    ax2.axvline(xs.index(best_r2["value"]), color="#0F6E56", ls="--", lw=1,
                label=f"best R² ({best_r2['value']})")
    if cur in xs:
        ax2.axvline(xs.index(cur), color="#888", ls=":", lw=1, label=f"current ({cur})")
    ax2.legend(fontsize=8, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_dir = OUT_DIR / stat            # one folder per model
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stat}_{param}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  Chart saved -> {out.relative_to(ROOT)}\n")
    return out


def _parse_values(raw_str, param):
    raw = [v.strip() for v in raw_str.split(",")]
    floaty = ("learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda")
    return [float(v) if ("." in v or param in floaty) else int(v) for v in raw]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sweep hyperparameter(s) for model(s) and graph MAE/R². "
                    "Use --stat all / --param all to calibrate everything.")
    ap.add_argument("--stat", required=True,
                    choices=list(PARAMS.keys()) + ["all"])
    ap.add_argument("--param", required=True,
                    choices=list(DEFAULT_GRIDS.keys()) + ["all"])
    ap.add_argument("--values", default=None,
                    help="Comma-separated values (single --param only; else default grids).")
    ap.add_argument("--stack-minutes", action="store_true",
                    help="Include the projected-minutes feature (counting stats).")
    args = ap.parse_args()

    stats  = list(PARAMS.keys())        if args.stat  == "all" else [args.stat]
    params = list(DEFAULT_GRIDS.keys()) if args.param == "all" else [args.param]

    if args.values and len(params) > 1:
        print("--values is ignored when sweeping multiple params; using default grids.")
        args.values = None

    total = len(stats) * len(params)
    print(f"\nCalibrating {len(stats)} model(s) × {len(params)} hyperparameter(s) "
          f"= {total} sweeps → reports/tuning/<stat>/\n")

    n = 0
    for s in stats:
        summary = []
        for p in params:
            n += 1
            grid = _parse_values(args.values, p) if args.values else DEFAULT_GRIDS[p]
            print(f"[{n}/{total}]", end=" ")
            rows, best, cur = sweep(s, p, grid, args.stack_minutes)
            plot(s, p, rows, best, cur)
            summary.append((p, best["value"], best["mae"],
                            max(rows, key=lambda r: r["r2"])["value"]))
        # Per-model summary of the best value for each hyperparameter.
        if len(params) > 1:
            print(f"  ── {s.upper()} best-by-param summary ──")
            print(f"    {'param':<18}{'best MAE @':>12}{'MAE':>9}{'best R² @':>12}")
            for p, bv, bmae, br2 in summary:
                print(f"    {p:<18}{str(bv):>12}{bmae:>9.3f}{str(br2):>12}")
            print()
