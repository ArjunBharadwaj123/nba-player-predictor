"""
NFL data collection (nflverse via nflreadpy)
============================================
Downloads regular-season weekly data and caches each (dataset, season) as a
local parquet under ``nfl/data/cache`` so reruns don't re-download unchanged
seasons. nflreadpy returns **polars** dataframes; we keep polars through the
cache layer and convert to **pandas** only at the boundary the feature pipeline
consumes (``as_pandas=True``).

Schema validation
-----------------
Every loader declares REQUIRED and OPTIONAL columns. If a required column is
missing we raise ``SchemaError`` (fails clearly, logs the missing columns).
Optional datasets that fail to download are skipped, and the set of datasets /
optional features that were actually available is recorded in the returned
``DataBundle.availability`` for pipeline metadata.

Raw play-by-play (``load_pbp``) is intentionally NOT part of the default bundle:
it is enormous and the opponent-defense aggregates we need are derivable from
weekly ``player_stats``. It can be enabled explicitly (``include_pbp=True``) for
advanced features but is never required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

import nflreadpy as nfl

from nfl.config import DATA_CACHE, REGULAR_SEASON_TYPE

log = logging.getLogger(__name__)


class SchemaError(RuntimeError):
    """Raised when a required column is missing from a downloaded dataset."""


# ── Required / optional columns per dataset ─────────────────────────────────────
# We validate only the columns the pipeline actually reads. Missing REQUIRED ->
# hard failure. Missing OPTIONAL -> the feature that needs it is skipped.
REQUIRED_COLUMNS = {
    "player_stats": [
        "player_id", "player_name", "player_display_name", "position",
        "position_group", "season", "week", "season_type", "game_id",
        "team", "opponent_team",
    ],
    "players": ["gsis_id", "display_name", "position", "position_group"],
    "schedules": [
        "game_id", "season", "week", "gameday", "away_team", "home_team",
    ],
}

OPTIONAL_COLUMNS = {
    "player_stats": [
        "headshot_url", "completions", "attempts", "passing_yards",
        "passing_tds", "passing_interceptions", "carries", "rushing_yards",
        "rushing_tds", "targets", "receptions", "receiving_yards",
        "receiving_tds", "fg_made", "pat_made", "fantasy_points",
        "fantasy_points_ppr", "target_share", "air_yards_share",
        "passing_epa", "rushing_epa", "receiving_epa", "passing_cpoe",
        "fumbles_lost_total",
    ],
    "schedules": [
        "gametime", "away_rest", "home_rest", "spread_line", "total_line",
        "roof", "surface", "temp", "wind", "div_game", "location",
    ],
    "injuries": ["gsis_id", "season", "week", "team", "report_status",
                 "practice_status"],
    "snap_counts": ["pfr_player_id", "season", "week", "team", "offense_pct"],
    "depth_charts": ["gsis_id", "season", "week", "depth_team", "position"],
    "rosters_weekly": ["gsis_id", "season", "week", "team", "position",
                       "status", "years_exp", "headshot_url"],
    "nextgen_stats": [],
}


@dataclass
class DataBundle:
    """Container for the collected NFL datasets (all polars unless converted)."""
    seasons: list[int]
    player_stats: pl.DataFrame
    players: pl.DataFrame
    schedules: pl.DataFrame
    injuries: pl.DataFrame | None = None
    snap_counts: pl.DataFrame | None = None
    depth_charts: pl.DataFrame | None = None
    rosters_weekly: pl.DataFrame | None = None
    nextgen_stats: pl.DataFrame | None = None
    availability: dict = field(default_factory=dict)


# ── Cache helpers ───────────────────────────────────────────────────────────────

def _cache_path(dataset: str, season: int) -> Path:
    return DATA_CACHE / f"{dataset}_{season}.parquet"


def _load_cached_per_season(dataset: str, seasons: list[int], loader,
                            season_kw: str = "seasons") -> pl.DataFrame | None:
    """Load a dataset season-by-season, caching each season's parquet locally.

    ``loader`` is the nflreadpy function; ``season_kw`` names its season arg.
    Returns a concatenated polars DataFrame, or None if nothing could be loaded.
    """
    frames: list[pl.DataFrame] = []
    for season in seasons:
        cache = _cache_path(dataset, season)
        if cache.exists():
            try:
                frames.append(pl.read_parquet(cache))
                continue
            except Exception as exc:  # corrupt cache -> re-download
                log.warning("Cache read failed for %s (%s); re-downloading",
                            cache.name, exc)
        try:
            df = loader(**{season_kw: [season]})
        except Exception as exc:
            log.warning("Download failed: %s season %d (%s)", dataset, season, exc)
            continue
        if df is None or df.is_empty():
            continue
        try:
            df.write_parquet(cache)
        except Exception as exc:
            log.warning("Cache write failed for %s (%s)", cache.name, exc)
        frames.append(df)
    if not frames:
        return None
    # Align columns across seasons (schemas can drift slightly year to year).
    return pl.concat(frames, how="diagonal_relaxed")


def _validate(dataset: str, df: pl.DataFrame) -> list[str]:
    """Return present optional columns; raise SchemaError on missing required."""
    cols = set(df.columns)
    required = REQUIRED_COLUMNS.get(dataset, [])
    missing = [c for c in required if c not in cols]
    if missing:
        raise SchemaError(
            f"Dataset '{dataset}' is missing required columns: {missing}. "
            f"Available columns: {sorted(cols)[:40]}..."
        )
    present_optional = [c for c in OPTIONAL_COLUMNS.get(dataset, []) if c in cols]
    return present_optional


# ── Public collection API ───────────────────────────────────────────────────────

def collect(
    seasons: list[int],
    *,
    regular_season_only: bool = True,
    include_injuries: bool = True,
    include_snaps: bool = True,
    include_depth: bool = True,
    include_rosters: bool = True,
    include_nextgen: bool = False,
    include_pbp: bool = False,
) -> DataBundle:
    """Download (or load from cache) the NFL data bundle for ``seasons``.

    Required datasets (player_stats, players, schedules) raise on failure.
    Optional datasets are skipped on failure and recorded in ``availability``.
    """
    availability: dict = {"seasons": list(seasons), "optional": {}}

    log.info("Collecting NFL data for seasons %s", seasons)

    # ── Required: player_stats ─────────────────────────────────────────────────
    player_stats = _load_cached_per_season("player_stats", seasons,
                                           nfl.load_player_stats)
    if player_stats is None:
        raise SchemaError("Failed to download player_stats for any season.")
    availability["player_stats_optional"] = _validate("player_stats", player_stats)
    if regular_season_only and "season_type" in player_stats.columns:
        player_stats = player_stats.filter(
            pl.col("season_type") == REGULAR_SEASON_TYPE)

    # ── Required: players (canonical IDs / headshots / experience) ─────────────
    players = None
    cache = DATA_CACHE / "players.parquet"
    if cache.exists():
        try:
            players = pl.read_parquet(cache)
        except Exception:
            players = None
    if players is None:
        players = nfl.load_players()
        try:
            players.write_parquet(cache)
        except Exception:
            pass
    _validate("players", players)

    # ── Required: schedules ────────────────────────────────────────────────────
    schedules = _load_cached_per_season("schedules", seasons, nfl.load_schedules)
    if schedules is None:
        raise SchemaError("Failed to download schedules for any season.")
    availability["schedules_optional"] = _validate("schedules", schedules)

    bundle = DataBundle(
        seasons=list(seasons),
        player_stats=player_stats,
        players=players,
        schedules=schedules,
        availability=availability,
    )

    # ── Optional datasets ──────────────────────────────────────────────────────
    def _try_optional(name, enabled, loader):
        if not enabled:
            availability["optional"][name] = False
            return None
        df = _load_cached_per_season(name, seasons, loader)
        if df is None:
            availability["optional"][name] = False
            log.warning("Optional dataset '%s' unavailable — skipping.", name)
            return None
        try:
            _validate(name, df)
        except SchemaError as exc:
            availability["optional"][name] = False
            log.warning("Optional dataset '%s' failed validation: %s", name, exc)
            return None
        availability["optional"][name] = True
        return df

    bundle.injuries      = _try_optional("injuries", include_injuries,
                                         nfl.load_injuries)
    bundle.snap_counts   = _try_optional("snap_counts", include_snaps,
                                         nfl.load_snap_counts)
    bundle.depth_charts  = _try_optional("depth_charts", include_depth,
                                         nfl.load_depth_charts)
    bundle.rosters_weekly = _try_optional("rosters_weekly", include_rosters,
                                          nfl.load_rosters_weekly)
    bundle.nextgen_stats = _try_optional("nextgen_stats", include_nextgen,
                                         nfl.load_nextgen_stats)

    if include_pbp:
        # Documented advanced path: pbp is huge and optional. We load it but
        # never require it; downstream feature code guards on availability.
        try:
            pbp = _load_cached_per_season("pbp", seasons, nfl.load_pbp)
            availability["optional"]["pbp"] = pbp is not None
        except Exception as exc:
            log.warning("pbp load failed (%s) — continuing without it.", exc)
            availability["optional"]["pbp"] = False

    log.info("Collection complete. Optional availability: %s",
             availability["optional"])
    return bundle


def to_pandas(df: pl.DataFrame | None):
    """polars -> pandas boundary. Returns None passthrough."""
    if df is None:
        return None
    return df.to_pandas()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    from nfl.config import default_seasons
    b = collect(default_seasons())
    print("player_stats:", b.player_stats.shape)
    print("schedules:", b.schedules.shape)
    print("availability:", b.availability["optional"])
