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
TEAM_ENV_JSON    = DATA_PROCESSED / "team_environment.json"

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

    # The canonical players table decides who is ACTIVE (excludes retirees) and
    # resolves each player's CURRENT team via ``latest_team`` (fixes stale post-
    # trade teams; correct even in the offseason). The current depth chart gives
    # each player's starter rank on their current team. Both fall back gracefully
    # when the live fetch fails.
    current_team, active_ids = _current_player_map()
    depth_map = _current_depth_map(current_team)

    players = []
    for _, r in latest.iterrows():
        pid = str(r["gsis_id"])
        trained_team = str(r.get("team") or "")
        if active_ids is not None:
            if pid not in active_ids:
                continue                       # not currently rostered -> hide
            cur_team = current_team.get(pid, trained_team)
        else:
            cur_team = trained_team            # no player data -> keep trained team
        hs = r.get("headshot_url")
        d = depth_map.get(pid, {})
        players.append({
            "id": pid,
            "name": str(r.get("player_name") or r.get("display_name") or ""),
            "position": str(r["position"]),
            "team": cur_team,
            "prev_team": trained_team,
            "team_changed": bool(cur_team and trained_team and cur_team != trained_team),
            "depth_rank": d.get("depth_rank"),
            "is_starter": d.get("is_starter"),
            "headshot_url": None if pd.isna(hs) else str(hs),
            "last_season": int(r["season"]),
            "last_week": int(r["week"]),
        })
    players = [p for p in players if p["name"]]
    players.sort(key=lambda p: p["name"])
    log.info("Serving player index: %d active players (%s); depth for %d",
             len(players),
             "player-filtered" if active_ids is not None else "unfiltered fallback",
             sum(1 for p in players if p["is_starter"] is not None))
    return serving, players


_ACTIVE_STATUSES = {"ACT", "ACTIVE", "RES", "DEV"}


def _current_player_map():
    """(current_team_by_id, active_id_set) from the live canonical players table.

    Current team is ``latest_team`` (authoritative, offseason-robust). A player is
    ACTIVE if their ``last_season`` reaches the current season OR they appear in the
    current depth-chart snapshot (catches players whose ``last_season`` nflverse has
    not yet bumped). ``status`` alone is NOT used to gate activity — retirees keep
    ``status=ACT`` with an old ``last_season``.

    Returns (dict, set) or ({}, None) when unavailable — None signals callers to
    skip active-filtering and keep trained teams."""
    from nfl.config import current_nfl_season
    from nfl.features.build_dataset import TEAM_STANDARDIZE
    from nfl.scraping.collect import load_current_players, load_current_depth

    pl = load_current_players()
    if pl is None or pl.empty or "gsis_id" not in pl.columns:
        return {}, None
    season = current_nfl_season()

    current_team = {}
    if "latest_team" in pl.columns:
        for _, r in pl.iterrows():
            t = str(r.get("latest_team") or "")
            if t and t.lower() != "nan":
                current_team[str(r["gsis_id"])] = TEAM_STANDARDIZE.get(t, t)

    last_season = pd.to_numeric(pl.get("last_season"), errors="coerce")
    active_ids = set(pl.loc[last_season >= season, "gsis_id"].astype(str))
    # Union in anyone currently on a depth chart (roster-lag safety net).
    dc = load_current_depth(season)
    if dc is not None and not dc.empty and "gsis_id" in dc.columns:
        active_ids |= set(dc["gsis_id"].astype(str))
    if not active_ids:
        return current_team, None            # nothing usable -> don't over-filter
    return current_team, active_ids


def _current_depth_map(current_team: dict) -> dict:
    """{gsis_id: {"depth_rank": int, "is_starter": 0/1}} from the current depth chart.

    A player can be listed under several teams (e.g. an old and a new team after a
    trade), so we keep only the rows for the player's CURRENT team (``latest_team``)
    and read the most recent snapshot (``dt``). ``pos_rank == 1`` -> starter. Empty
    dict when the depth chart is unavailable."""
    from nfl.config import current_nfl_season
    from nfl.features.build_dataset import TEAM_STANDARDIZE
    from nfl.scraping.collect import load_current_depth

    dc = load_current_depth(current_nfl_season())
    if dc is None or dc.empty or "gsis_id" not in dc.columns:
        return {}
    dc = dc.copy()
    dc["gsis_id"] = dc["gsis_id"].astype(str)
    dc["pos_rank"] = pd.to_numeric(dc.get("pos_rank"), errors="coerce")
    if "team" in dc.columns:
        dc["team"] = dc["team"].astype("string").map(
            lambda t: TEAM_STANDARDIZE.get(t, t) if t is not None else t)
    # Keep only rows matching the player's current team when we know it.
    if "team" in dc.columns and current_team:
        dc["_cur"] = dc["gsis_id"].map(current_team)
        matched = dc[dc["_cur"].notna() & (dc["team"] == dc["_cur"])]
        dc = matched if not matched.empty else dc
    # Most recent snapshot per player, then its best (lowest) pos_rank.
    if "dt" in dc.columns:
        dc["dt"] = pd.to_datetime(dc["dt"], errors="coerce")
        dc = dc.sort_values("dt")
    dc = dc.dropna(subset=["pos_rank"])
    out: dict = {}
    for pid, grp in dc.groupby("gsis_id"):
        if "dt" in grp.columns and grp["dt"].notna().any():
            last_dt = grp["dt"].max()
            grp = grp[grp["dt"] == last_dt]
        rank = int(grp["pos_rank"].min())
        out[str(pid)] = {"depth_rank": rank, "is_starter": int(rank == 1)}
    return out


def build_team_environment(fe: pd.DataFrame) -> dict:
    """Latest own-team offensive-environment profile per team (season-to-date,
    shifted). Injected at serve time from the player's CURRENT team so a mover is
    projected in the new offense."""
    from nfl.features.engineer import TEAM_ENV_FEATURES
    latest = (fe.sort_values(["season", "week"])
                .groupby("team", group_keys=False).tail(1))
    table: dict = {}
    for _, r in latest.iterrows():
        table[str(r["team"])] = {
            c: (None if pd.isna(r.get(c)) else round(float(r[c]), 4))
            for c in TEAM_ENV_FEATURES if c in fe.columns
        }
    return table


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
    team_environment = build_team_environment(fe)

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
    _atomic_write_json(team_environment, TEAM_ENV_JSON)
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
