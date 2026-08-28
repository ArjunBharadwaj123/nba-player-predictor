"""
NBA Stats bulk collector (primary game-log source)
==================================================
Pulls **every player's** per-game box score for a season in a single
`LeagueGameLog(player_or_team_abbreviation="P")` call, instead of scraping one
Basketball-Reference HTML page per player. This is the hybrid plan's *primary*
collector: fast and wide (500+ players/season in a handful of API calls). The
BBRef scraper (`bbref_scraper.py`) is kept as a fallback / cross-check.

Why this is dramatically faster than the BBRef scraper:
    - BBRef: 1 HTTP page per (player, season) with a 4s+jitter delay
      → ~130 players x 4 seasons ≈ 40-50 min of wall-clock sleeping.
    - Here: 1 API call per season returns *all* players' games at once
      → ~5 seasons in well under a minute of network time.

Output (drop-in compatible with the existing pipeline):
    data/raw/all_gamelogs.csv   — same 38-column schema build_dataset.py expects

Schema parity:
    LeagueGameLog gives the basic box score. fg2/fg2a/fg2_pct/efg_pct/game_score
    are derived here so the 38 columns match the BBRef output exactly. `result`
    is the W/L flag (build_dataset/engineer only read result.startswith("W")).
    `gs` (games started) isn't in the bulk endpoint, so we derive a starter
    proxy: the top-5 minutes players per team-game are marked as starters —
    keeps engineer.py's starter features meaningful without per-game box calls.

Positions:
    LeagueGameLog carries no position. We map player -> position by name from
    data/processed/roster.json (precise BBRef positions for active players).
    Unmatched (mostly retired) players default to SF, which engineer.py already
    uses as its neutral position fallback.

Not collected here (deferred — needs per-game calls that would erase the speed
win): per-game usage%/TS%/ORtg/DRtg. Season-aggregate usage% is a cheap future
add via LeagueDashPlayerStats.

Usage:
    python scraping/nba_stats_collector.py --mode train     # 2022-2025
    python scraping/nba_stats_collector.py --mode current   # 2026 only
    python scraping/nba_stats_collector.py --mode all        # everything
    python scraping/nba_stats_collector.py --seasons 2024,2025
"""

import argparse
import json
import logging
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import LeagueGameLog

# Reuse season/format helpers and constants from the team-context client.
from nba_api_client import (
    season_to_str,
    API_DELAY,
    TRAINING_SEASONS,
    CURRENT_SEASON,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CACHE     = RAW / "nba_api"
RAW.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

ROSTER_JSON = PROCESSED / "roster.json"
ALL_OUT     = RAW / "all_gamelogs.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# The exact 38-column schema build_dataset.py / engineer.py expect.
SCHEMA = [
    "player_id", "player_name", "position", "season", "game_date", "team",
    "home_game", "opponent", "result", "minutes", "rank", "game_num_career",
    "game_num_team", "gs", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
    "fg2", "fg2a", "fg2_pct", "efg_pct", "ft", "fta", "ft_pct", "orb", "drb",
    "trb", "ast", "stl", "blk", "tov", "pf", "pts", "game_score", "plus_minus",
]


# ─────────────────────────────────────────────────────────────────────────────
# Position lookup (name -> position from roster.json)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_name(name: str) -> str:
    """Normalise a player name for matching: strip accents, punctuation, case,
    and common suffixes (Jr./Sr./II/III)."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    for suffix in (" jr", " sr", " ii", " iii", " iv"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return " ".join(s.split())


def load_position_map() -> dict:
    """name(normalised) -> position, from roster.json. Empty dict if missing."""
    if not ROSTER_JSON.exists():
        log.warning("roster.json not found — positions default to SF")
        return {}
    roster = json.loads(ROSTER_JSON.read_text())
    pos_map = {_norm_name(p["name"]): p.get("pos", "SF") for p in roster}
    log.info("Loaded %d positions from roster.json", len(pos_map))
    return pos_map


# ─────────────────────────────────────────────────────────────────────────────
# Fetch one season (cached JSON) and normalise to the 38-column schema
# ─────────────────────────────────────────────────────────────────────────────

def fetch_season_raw(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch all player-game rows for a season via LeagueGameLog (player mode).

    Caches the raw response to data/raw/nba_api/player_gamelog_{season}.csv so
    re-runs are instant. Only the current season should be force-refreshed.
    """
    cache_path = CACHE / f"player_gamelog_{season}.csv"
    if cache_path.exists() and not force_refresh:
        log.info("  [cache] %s", cache_path.name)
        return pd.read_csv(cache_path)

    season_str = season_to_str(season)
    log.info("Fetching player game logs — %s", season_str)
    try:
        gl = LeagueGameLog(
            season=season_str,
            player_or_team_abbreviation="P",   # P = per-player rows
            direction="ASC",
            timeout=60,
        )
        df = gl.get_data_frames()[0]
    except Exception as exc:
        log.error("  Failed to fetch %s: %s", season_str, exc)
        return pd.DataFrame()

    time.sleep(API_DELAY)
    df.to_csv(cache_path, index=False)
    log.info("  %d player-game rows -> %s", len(df), cache_path.name)
    return df


def _derive_starter_flag(df: pd.DataFrame) -> pd.Series:
    """Approximate games-started: the 5 highest-minutes players per team-game.

    LeagueGameLog has no start flag; minutes rank is a strong proxy and keeps
    engineer.py's starter features (is_likely_starter, start_rate_last10) alive.
    """
    rank = (
        df.groupby(["game_id", "team"])["minutes"]
        .rank(method="first", ascending=False)
    )
    return (rank <= 5).astype(int)


def normalise_season(raw: pd.DataFrame, season: int, pos_map: dict) -> pd.DataFrame:
    """Map a raw LeagueGameLog frame to the canonical 38-column schema."""
    if raw.empty:
        return pd.DataFrame(columns=SCHEMA)

    d = raw.rename(columns={
        "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
        "TEAM_ABBREVIATION": "team", "GAME_ID": "game_id",
        "GAME_DATE": "game_date", "MATCHUP": "matchup", "WL": "result",
        "MIN": "minutes", "FGM": "fg", "FGA": "fga", "FG_PCT": "fg_pct",
        "FG3M": "fg3", "FG3A": "fg3a", "FG3_PCT": "fg3_pct",
        "FTM": "ft", "FTA": "fta", "FT_PCT": "ft_pct",
        "OREB": "orb", "DREB": "drb", "REB": "trb", "AST": "ast",
        "STL": "stl", "BLK": "blk", "TOV": "tov", "PF": "pf", "PTS": "pts",
        "PLUS_MINUS": "plus_minus",
    }).copy()

    d["season"]    = season
    d["game_date"] = pd.to_datetime(d["game_date"]).dt.strftime("%Y-%m-%d")

    # Home/away and opponent from the matchup string ("LAL vs. DEN" / "LAL @ DEN")
    d["home_game"] = d["matchup"].str.contains(r"\bvs\.", regex=True).astype(int)
    d["opponent"]  = (
        d["matchup"].str.replace(r".*(vs\.|@)\s*", "", regex=True).str.strip()
    )

    # Numeric coercions
    for c in ["minutes", "fg", "fga", "fg3", "fg3a", "ft", "fta",
              "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts",
              "plus_minus", "fg_pct", "fg3_pct", "ft_pct"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # Derived stats to complete schema parity with BBRef
    d["fg2"]     = d["fg"] - d["fg3"]
    d["fg2a"]    = d["fga"] - d["fg3a"]
    d["fg2_pct"] = (d["fg2"] / d["fg2a"].replace(0, np.nan)).round(3)
    d["efg_pct"] = ((d["fg"] + 0.5 * d["fg3"]) / d["fga"].replace(0, np.nan)).round(3)
    # John Hollinger's Game Score
    d["game_score"] = (
        d["pts"] + 0.4 * d["fg"] - 0.7 * d["fga"] - 0.4 * (d["fta"] - d["ft"])
        + 0.7 * d["orb"] + 0.3 * d["drb"] + d["stl"] + 0.7 * d["ast"]
        + 0.7 * d["blk"] - 0.4 * d["pf"] - d["tov"]
    ).round(1)

    # Starter proxy + per-team game number
    d["gs"] = _derive_starter_flag(d)
    d = d.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    d["game_num_team"]   = d.groupby(["player_id", "season"]).cumcount() + 1
    d["rank"]            = np.nan   # BBRef 'Rk'; unused downstream
    d["game_num_career"] = np.nan   # not available in bulk; unused downstream

    # Position by name; default SF (engineer's neutral fallback)
    d["position"] = (
        d["player_name"].map(lambda n: pos_map.get(_norm_name(n), "SF"))
    )

    d["player_id"] = d["player_id"].astype(str)
    return d[SCHEMA]


# ─────────────────────────────────────────────────────────────────────────────
# Rotation filter — keep players with a meaningful role each season
# ─────────────────────────────────────────────────────────────────────────────

def filter_rotation(df: pd.DataFrame, min_games: int, min_mpg: float) -> pd.DataFrame:
    """Keep only (player, season) pairs with >= min_games played and >= min_mpg
    average minutes. ALL of a qualifying player's game rows are retained (rolling
    features need their full game history) — we filter players, not rows.

    A player's deep-bench season is dropped, but a star who later got hurt still
    keeps every game they played that season.
    """
    if df.empty or min_games <= 0:
        return df
    played = df[df["minutes"] >= 1]
    grp = played.groupby(["player_id", "season"])["minutes"]
    stats = pd.DataFrame({"games": grp.size(), "mpg": grp.mean()}).reset_index()
    keep = stats[(stats["games"] >= min_games) & (stats["mpg"] >= min_mpg)]
    keep_keys = set(zip(keep["player_id"], keep["season"]))

    mask = [
        (pid, s) in keep_keys
        for pid, s in zip(df["player_id"], df["season"])
    ]
    out = df[pd.Series(mask, index=df.index)].reset_index(drop=True)
    log.info(
        "Rotation filter (>=%d games, >=%.0f mpg): %d -> %d rows, %d -> %d players",
        min_games, min_mpg, len(df), len(out),
        df["player_name"].nunique(), out["player_name"].nunique(),
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def collect(seasons: list, force_current: bool = True,
            min_games: int = 20, min_mpg: float = 15.0) -> pd.DataFrame:
    """Collect + normalise all requested seasons and write all_gamelogs.csv."""
    pos_map = load_position_map()
    frames = []
    for season in seasons:
        force = force_current and season == CURRENT_SEASON
        raw = fetch_season_raw(season, force_refresh=force)
        norm = normalise_season(raw, season, pos_map)
        if not norm.empty:
            frames.append(norm)
            log.info("  %s: %d rows, %d players",
                     season, len(norm), norm["player_name"].nunique())

    if not frames:
        log.error("No data collected.")
        return pd.DataFrame(columns=SCHEMA)

    combined = pd.concat(frames, ignore_index=True)
    combined = filter_rotation(combined, min_games, min_mpg)
    combined.to_csv(ALL_OUT, index=False)
    log.info("Saved %d rows, %d players, seasons %s -> %s",
             len(combined), combined["player_name"].nunique(),
             sorted(combined["season"].unique()), ALL_OUT.name)
    return combined


def quality_report(df: pd.DataFrame) -> None:
    if df.empty:
        print("Empty dataset.")
        return
    print(f"\n{'-'*55}\n nba_stats_collector quality report\n{'-'*55}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Players : {df['player_name'].nunique()}")
    print(f"  Seasons : {sorted(df['season'].unique())}")
    print(f"  Cols    : {len(df.columns)} (schema match: {list(df.columns) == SCHEMA})")
    print(f"  Position coverage (non-SF): "
          f"{(df['position'] != 'SF').mean()*100:.0f}%")
    print("  Stat averages (minutes>=5):")
    m = df[df['minutes'] >= 5]
    for c in ["pts", "reb" if "reb" in df else "trb", "ast", "fg3", "minutes"]:
        if c in m:
            print(f"    {c:8s}: {m[c].mean():.2f}")
    print(f"{'-'*55}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bulk NBA player game-log collector.")
    ap.add_argument("--mode", choices=["train", "current", "all"], default="all")
    ap.add_argument("--seasons", default="",
                    help="Comma-separated seasons, overrides --mode.")
    ap.add_argument("--min-games", type=int, default=20,
                    help="Min games played per season to keep a player (default 20).")
    ap.add_argument("--min-mpg", type=float, default=15.0,
                    help="Min avg minutes per game to keep a player (default 15).")
    args = ap.parse_args()

    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    elif args.mode == "train":
        seasons = list(TRAINING_SEASONS)
    elif args.mode == "current":
        seasons = [CURRENT_SEASON]
    else:
        seasons = list(TRAINING_SEASONS) + [CURRENT_SEASON]

    df = collect(seasons, min_games=args.min_games, min_mpg=args.min_mpg)
    quality_report(df)


if __name__ == "__main__":
    main()
