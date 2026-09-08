"""
Upcoming-game context for a player (serve time).
=================================================
Resolves a player's current team, finds their next scheduled game from the
nflverse schedule, and attaches opponent, kickoff, season/week, home/away, rest,
spread/total/weather, and the latest injury / depth-chart designation.

Returns None when no future game exists (off-season, or the player's team has no
upcoming game) — the API turns that into a friendly 404, matching the NBA app.
Network access is only needed for the *current* season's schedule; completed
seasons come from the local cache.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd

from nfl.config import current_nfl_season
from nfl.features.build_dataset import TEAM_STANDARDIZE
from nfl.serving import find_player, load_opp_defense, injury_from_row, latest_player_row

log = logging.getLogger(__name__)


def _load_schedule_frame(season: int, live: bool) -> pd.DataFrame | None:
    """Load one season's schedule. The CURRENT season is fetched live (fixtures
    and lines update weekly); completed seasons come from the local cache.

    Resilient: a successful live fetch is cached, and a failed live fetch falls
    back to that cache — so a transient nflverse timeout never blocks next-game
    resolution once the schedule has been seen at least once.
    """
    from nfl.config import DATA_CACHE
    cache = DATA_CACHE / f"schedules_{season}.parquet"
    df = None
    if live:
        try:
            import nflreadpy as nfl
            sched = nfl.load_schedules(seasons=[season])
            if sched is not None and not sched.is_empty():
                try:
                    sched.write_parquet(cache)          # cache-on-success
                except Exception:
                    pass
                df = sched.to_pandas()
        except Exception as exc:
            log.warning("Live schedule fetch failed for %s: %s", season, exc)
        if df is None and cache.exists():               # fall back to cache
            try:
                import polars as pl
                df = pl.read_parquet(cache).to_pandas()
                log.info("Using cached schedule for %s after live fetch failed.", season)
            except Exception:
                df = None
    else:
        try:
            import nflreadpy as nfl
            from nfl.scraping.collect import _load_cached_per_season
            sched = _load_cached_per_season("schedules", [season], nfl.load_schedules)
            df = None if sched is None else sched.to_pandas()
        except Exception as exc:
            log.warning("Schedule load failed for %s: %s", season, exc)
            df = None
    if df is None:
        return None
    for c in ("home_team", "away_team"):                 # align relocated codes
        if c in df.columns:
            df[c] = df[c].replace(TEAM_STANDARDIZE)
    return df


def get_next_game_context(player_id: str,
                          today: date | None = None) -> dict | None:
    player = find_player(player_id)
    if player is None:
        return None
    team = player.get("team")
    if not team:
        return None
    today = today or date.today()

    # The current NFL season (by date) is fetched LIVE so this week's fixtures/
    # lines are current; the prior season comes from cache as a fallback.
    cur = current_nfl_season(today)
    frames = []
    live = _load_schedule_frame(cur, live=True)
    if live is not None:
        frames.append(live)
    prev = _load_schedule_frame(cur - 1, live=False)
    if prev is not None:
        frames.append(prev)
    if not frames:
        return None
    sched = pd.concat(frames, ignore_index=True)
    if sched.empty:
        return None

    sched = sched.copy()
    sched["gameday_dt"] = pd.to_datetime(sched["gameday"], errors="coerce")
    team_games = sched[
        ((sched["home_team"] == team) | (sched["away_team"] == team))
        & (sched["gameday_dt"].dt.date >= today)
    ].sort_values("gameday_dt")
    if team_games.empty:
        return None

    g = team_games.iloc[0]
    is_home = g["home_team"] == team
    opp = g["away_team"] if is_home else g["home_team"]
    total = g.get("total_line")
    spread_home = g.get("spread_line")
    team_spread = None
    implied = None
    if pd.notna(spread_home):
        team_spread = -float(spread_home) if is_home else float(spread_home)
    if pd.notna(total) and team_spread is not None:
        implied = float(total) / 2.0 - team_spread / 2.0

    roof = str(g.get("roof") or "").lower()
    surface = str(g.get("surface") or "").lower()

    ctx = {
        "player_id": player_id,
        "player_name": player["name"],
        "position": player["position"],
        "team": team,
        "opponent_team": opp,
        "home_away": 1 if is_home else 0,
        "season": int(g["season"]),
        "week": int(g["week"]),
        "game_date": str(g.get("gameday")),
        "kickoff": g.get("gametime"),
        "days_rest": _num(g.get("home_rest" if is_home else "away_rest")),
        "spread_line_team": team_spread,
        "game_total": _num(total),
        "implied_team_total": implied,
        "is_indoor": int(roof in ("dome", "closed")),
        "is_grass": int(surface == "grass"),
        "temp": _num(g.get("temp")),
        "wind": _num(g.get("wind")),
        "div_game": int(_num(g.get("div_game")) or 0),
        "roof": g.get("roof"),
        "surface": g.get("surface"),
        # missing indicators surfaced for the UI (never faked as real zeros).
        "missing_odds": int(pd.isna(total) or pd.isna(spread_home)),
        "missing_weather": int(roof in ("dome", "closed") or pd.isna(g.get("temp"))),
    }

    # Injury + depth designation (from serving row / latest data).
    ctx["injury_designation"] = injury_from_row(player_id)
    base = latest_player_row(player_id)
    if base is not None:
        depth = base.get("depth_rank")
        ctx["depth_rank"] = None if depth is None or pd.isna(depth) else int(depth)
        ctx["is_starter"] = int(base.get("is_starter", 0) or 0)

    # Opponent-defense context (for display).
    opp_def = load_opp_defense().get(player["position"], {}).get(opp)
    if opp_def:
        ctx["opponent_defense"] = opp_def
    return ctx


def _num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None
