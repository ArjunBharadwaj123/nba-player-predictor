"""
Admin dashboard API (password-protected).
=========================================
Exposes the private model-performance view for a single admin:

    GET /admin/verify           — 200 if the password is correct (frontend gate)
    GET /admin/stats            — NBA + NFL training/test metrics (normalized)
    GET /admin/top-predictions  — combined NBA+NFL confidence-weighted Top-10

Auth: a single shared password in the ``ADMIN_PASSWORD`` env var, sent by the
client in the ``X-Admin-Password`` header and compared server-side (constant
time). If ``ADMIN_PASSWORD`` is unset the admin API returns 503 (not configured)
— the password is never committed to the repo.
"""

from __future__ import annotations

import hmac
import json
import math
import os
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, Header, HTTPException

router = APIRouter(prefix="/admin", tags=["admin"])

ROOT = Path(__file__).parent.parent
NFL_SAVED = ROOT / "nfl" / "models" / "saved"
NBA_SAVED = ROOT / "nba" / "models" / "saved"
NFL_PROCESSED = ROOT / "nfl" / "data" / "processed"
TOP_PREDS = ROOT / "data" / "top_predictions.json"


def _safe(obj):
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
    return obj


def require_admin(x_admin_password: str | None = Header(default=None)):
    """Gate: compare the header to ADMIN_PASSWORD (constant-time)."""
    expected = os.environ.get("ADMIN_PASSWORD")
    if not expected:
        raise HTTPException(503, "Admin dashboard is not configured "
                                 "(ADMIN_PASSWORD env var not set).")
    if not x_admin_password or not hmac.compare_digest(x_admin_password, expected):
        raise HTTPException(401, "Invalid admin password.")
    return True


def _load_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def _nfl_stats() -> dict:
    ev = _load_json(NFL_SAVED / "eval_report.json") or {}
    meta = _load_json(NFL_SAVED / "training_metadata.json") or {}
    fresh = _load_json(NFL_PROCESSED / "freshness.json") or {}
    targets = []
    for pos, pr in (ev.get("positions") or {}).items():
        rng = pr.get("holdout_date_range")
        for target, m in (pr.get("targets") or {}).items():
            targets.append({
                "position": pos, "target": target,
                "mae": m.get("mae"), "rmse": m.get("rmse"), "r2": m.get("r2"),
                "baseline_mae": m.get("baseline_mae"),
                "improvement": m.get("improvement_over_baseline"),
                "beats_baseline": m.get("beats_baseline"),
                "coverage": m.get("interval_coverage_calibrated")
                            or m.get("interval_coverage_p15_p85"),
                "n_train": m.get("n_train"), "n_eval": m.get("n_eval"),
                "holdout_range": rng,
            })
    return {
        "sport": "NFL",
        "mode": meta.get("mode") or fresh.get("mode"),
        "seasons": meta.get("seasons") or fresh.get("seasons"),
        "n_features": meta.get("n_features"),
        "n_rows": (fresh.get("signature") or {}).get("n_rows"),
        "n_players": (fresh.get("signature") or {}).get("n_players"),
        "updated_at": fresh.get("updated_at"),
        "failed_baselines": ev.get("failed_baselines", []),
        "targets": targets,
    }


def _nba_stats() -> dict:
    ev = _load_json(NBA_SAVED / "eval_report.json")
    results = {}
    holdout = None
    if isinstance(ev, list):
        for el in ev:
            if isinstance(el, dict) and "results" in el:
                results = el.get("results", {})
                holdout = el.get("holdout")
                break
    elif isinstance(ev, dict):
        results = ev.get("results", ev)
    targets = []
    for stat, m in (results or {}).items():
        if not isinstance(m, dict):
            continue
        targets.append({
            "position": "ALL", "target": stat,
            "mae": m.get("mae"), "rmse": m.get("rmse"), "r2": m.get("r2"),
            "coverage": m.get("coverage"), "n_eval": m.get("n_test"),
        })
    return {
        "sport": "NBA",
        "mode": "production",
        "holdout_frac": holdout,
        "updated_at": None,
        "targets": targets,
    }


@router.get("/verify", dependencies=[Depends(require_admin)])
def verify_ok():
    return {"ok": True, "message": "Admin password accepted."}


@router.get("/stats", dependencies=[Depends(require_admin)])
def stats():
    return _safe({"nba": _nba_stats(), "nfl": _nfl_stats()})


@router.get("/top-predictions", dependencies=[Depends(require_admin)])
def top_predictions():
    data = _load_json(TOP_PREDS)
    if data is None:
        return _safe({"predictions": [], "count": 0,
                      "note": "Top predictions not generated yet — run "
                              "`python -m analytics.top_predictions`."})
    return _safe(data)
