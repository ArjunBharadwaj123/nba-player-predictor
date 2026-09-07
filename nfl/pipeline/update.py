"""
Idempotent NFL update pipeline.
===============================
    python -m nfl.pipeline.update [flags]

Steps:
  1. Download / refresh configured seasons (cached).
  2. Refresh players, weekly rosters, schedules, injuries, depth charts, odds.
  3. Merge + dedup by canonical player id + game id.
  4. Rebuild leakage-safe features.
  5. Detect whether new completed games were added.
  6. Retrain only if the training data changed (unless forced / skipped).
  7. Re-run chronological evaluation after training.
  8. Build a compact serving dataset + player index.
  9. Record pipeline + data-freshness metadata.

Atomicity: processed artifacts are written to a temp path, validated, and only
then moved over the previous file — a network/optional-data failure never
overwrites good processed data with empty output.

Flags:
    --seasons 2018,2019,...   explicit seasons (overrides config)
    --skip-download           use only cached raw data
    --skip-train              build features + serving data, don't retrain
    --dry-run                 run steps, write nothing
    --position QB             restrict retrain/eval to one position
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nfl.config import (
    DATA_PROCESSED, MODELS_SAVED, default_seasons, is_dev_mode,
)
from nfl.features.build_dataset import build_dataset
from nfl.features.engineer import engineer_features, feature_columns
from nfl.scraping.collect import collect

log = logging.getLogger(__name__)

FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
SERVING_PARQUET  = DATA_PROCESSED / "features_serving.parquet"
PLAYERS_JSON     = DATA_PROCESSED / "players.json"
FRESHNESS_JSON   = DATA_PROCESSED / "freshness.json"
OPP_DEFENSE_JSON = DATA_PROCESSED / "opp_defense.json"

SERVING_GAMES_PER_PLAYER = 10   # recent games kept per player for serving


def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    if len(pd.read_parquet(tmp)) != len(df):        # validate before swap
        raise RuntimeError(f"Validation failed writing {path}")
    shutil.move(str(tmp), str(path))


def _atomic_write_json(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    shutil.move(str(tmp), str(path))


def build_serving(fe: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Compact serving dataset: the most recent games per player, plus a player
    index (latest row per canonical id) for the /players endpoint."""
    fe = fe.sort_values(["gsis_id", "season", "week"])
    serving = (fe.groupby("gsis_id", group_keys=False)
                 .tail(SERVING_GAMES_PER_PLAYER)
                 .reset_index(drop=True))

    latest = fe.groupby("gsis_id", group_keys=False).tail(1)
    players = []
    for _, r in latest.iterrows():
        hs = r.get("headshot_url")
        players.append({
            "id": str(r["gsis_id"]),
            "name": str(r.get("player_name") or r.get("display_name") or ""),
            "position": str(r["position"]),
            "team": str(r.get("team") or ""),
            "headshot_url": None if pd.isna(hs) else str(hs),
            "last_season": int(r["season"]),
            "last_week": int(r["week"]),
        })
    players = [p for p in players if p["name"]]
    players.sort(key=lambda p: p["name"])
    return serving, players


def build_opp_defense(fe: pd.DataFrame) -> dict:
    """Latest opponent-defense-allowed values per (position, opponent team).

    Used at serve time to inject the upcoming opponent's matchup context into a
    player's feature row. Leakage-safe: these are the shifted season-to-date
    values as of each team's most recent game.
    """
    cols = ["opp_fantasy_allowed_pos", "opp_pass_yards_allowed",
            "opp_rush_yards_allowed", "opp_rec_yards_allowed_pos",
            "opp_pass_tds_allowed", "opp_rush_tds_allowed"]
    latest = (fe.sort_values(["season", "week"])
                .groupby(["position", "opponent_team"], group_keys=False).tail(1))
    table: dict = {}
    for _, r in latest.iterrows():
        pos = str(r["position"])
        opp = str(r["opponent_team"])
        table.setdefault(pos, {})[opp] = {
            c: (None if pd.isna(r.get(c)) else round(float(r[c]), 3)) for c in cols
        }
    return table


def data_signature(fe: pd.DataFrame) -> dict:
    """A cheap signature of the completed-games content for change detection."""
    return {
        "n_rows": int(len(fe)),
        "max_season": int(fe["season"].max()),
        "max_week_in_max_season": int(
            fe.loc[fe["season"] == fe["season"].max(), "week"].max()),
        "n_players": int(fe["gsis_id"].nunique()),
    }


def run(seasons=None, skip_download=False, skip_train=False, dry_run=False,
        position=None) -> dict:
    seasons = seasons or default_seasons()
    log.info("=== NFL pipeline update | seasons=%s | dev=%s ===",
             seasons, is_dev_mode())

    # 1-3. Collect + build canonical dataset.
    # skip_download is honored by the cache: if raw parquet exists we never hit
    # the network. To *force* cache-only, we set an env the collector respects
    # via its per-season cache (missing cache + skip -> that season is skipped).
    if skip_download:
        os.environ.setdefault("NFL_CACHE_ONLY", "1")
    bundle = collect(seasons)
    ds = build_dataset(bundle)

    # 4. Engineer leakage-safe features.
    fe = engineer_features(ds)
    feats = feature_columns(fe)
    sig = data_signature(fe)
    log.info("Feature build: %s | %d features", sig, len(feats))

    # 5. Change detection vs. previous freshness signature.
    prev_sig = {}
    if FRESHNESS_JSON.exists():
        try:
            prev_sig = json.loads(FRESHNESS_JSON.read_text()).get("signature", {})
        except Exception:
            prev_sig = {}
    data_changed = sig != prev_sig

    serving, players = build_serving(fe)
    opp_defense = build_opp_defense(fe)

    freshness = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": seasons,
        "mode": "dev" if is_dev_mode() else "production",
        "signature": sig,
        "optional_datasets": bundle.availability.get("optional", {}),
        "n_features": len(feats),
        "n_serving_players": len(players),
    }

    if dry_run:
        log.info("[dry-run] data_changed=%s — writing nothing.", data_changed)
        return {"data_changed": data_changed, "signature": sig,
                "n_players": len(players)}

    # 8-9. Persist processed artifacts atomically.
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(fe, FEATURES_PARQUET)
    _atomic_write_parquet(serving, SERVING_PARQUET)
    _atomic_write_json(players, PLAYERS_JSON)
    _atomic_write_json(opp_defense, OPP_DEFENSE_JSON)
    _atomic_write_json(freshness, FRESHNESS_JSON)
    log.info("Wrote features (%d), serving (%d), players (%d)",
             len(fe), len(serving), len(players))

    # 6-7. Retrain + evaluate only when data changed (unless skipping).
    result = {"data_changed": data_changed, "signature": sig,
              "n_players": len(players), "trained": False, "evaluated": False}
    if skip_train:
        log.info("--skip-train: features/serving updated, models untouched.")
        return result
    if not data_changed and (MODELS_SAVED / "training_metadata.json").exists():
        log.info("No new completed games since last train — skipping retrain. "
                 "(Force by changing --seasons or deleting training_metadata.json)")
        return result

    from nfl.models.train import train_all
    from nfl.models.evaluate import evaluate_all
    positions = [position.upper()] if position else None
    train_all(positions)
    result["trained"] = True
    evaluate_all()
    result["evaluated"] = True
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="NFL update pipeline")
    ap.add_argument("--seasons", help="Comma-separated seasons, e.g. 2021,2022,2023")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--position")
    args = ap.parse_args()
    seasons = None
    if args.seasons:
        seasons = sorted({int(s) for s in args.seasons.split(",") if s.strip()})
    run(seasons=seasons, skip_download=args.skip_download,
        skip_train=args.skip_train, dry_run=args.dry_run, position=args.position)


if __name__ == "__main__":
    main()
