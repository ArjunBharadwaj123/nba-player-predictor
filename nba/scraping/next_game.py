"""
Next Game Context Fetcher — BallDontLie + NBA API
==================================================
Uses BallDontLie for schedule data and NBA API for team stats.

Place in: scraping/next_game.py
"""

import logging
import time
from datetime import datetime, date

import requests
import pandas as pd
from nba_api.stats.static import teams as nba_teams_static

log = logging.getLogger(__name__)

# ── BallDontLie ───────────────────────────────────────────────────────────────
BDL_BASE    = "https://api.balldontlie.io/v1"
BDL_API_KEY = "2f24c92a-eef2-4a71-a11e-9b21c02087c7"
BDL_HEADERS = {"Authorization": BDL_API_KEY}
BDL_DELAY   = 2.5   # seconds between BDL requests — free tier is strict
_BDL_GAMES_CACHE: dict = {}  # team_id -> list of games (avoid re-fetching same team)

# ── NBA API ───────────────────────────────────────────────────────────────────
NBA_DELAY      = 1.2
CURRENT_SEASON = "2025-26"

# ── BDL abbreviation corrections ─────────────────────────────────────────────
# BDL uses shorter abbreviations for some teams
BDL_ABBREV_FIX = {
    "GS":  "GSW",
    "SA":  "SAS",
    "NY":  "NYK",
    "NO":  "NOP",
}


# ─────────────────────────────────────────────────────────────────────────────
# BallDontLie: find player by last-name search
# ─────────────────────────────────────────────────────────────────────────────

def _bdl_find_player(player_name: str) -> dict | None:
    """
    Find a player on BallDontLie using last-name search.

    BDL's search parameter is a PREFIX search on first OR last name only.
    Searching the full name "LeBron James" returns nothing.
    Searching "James" returns multiple results — we pick the right one
    by matching the full name.

    Returns the player dict (includes current team) or None.
    """
    parts     = player_name.strip().split()
    last_name = parts[-1] if parts else player_name

    url    = f"{BDL_BASE}/players"
    params = {"search": last_name, "per_page": 25}

    try:
        resp = requests.get(url, headers=BDL_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        time.sleep(BDL_DELAY)
        data = resp.json().get("data", [])
    except Exception as exc:
        log.error("BDL player search failed for '%s': %s", player_name, exc)
        return None

    if not data:
        log.warning("BDL returned no results for last name '%s'", last_name)
        return None

    # Match by comparing full names (strip apostrophes for comparison)
    def _clean(s):
        return s.lower().replace("'", "").replace(".", "").strip()

    target = _clean(player_name)
    for p in data:
        full = _clean(p.get("first_name", "") + " " + p.get("last_name", ""))
        if full == target or target in full or full in target:
            log.info("BDL found: %s %s — team: %s",
                     p["first_name"], p["last_name"],
                     p.get("team", {}).get("abbreviation", "?"))
            return p

    # Fallback: first result with a warning
    p = data[0]
    log.warning("BDL exact match failed for '%s' — using: %s %s",
                player_name, p.get("first_name"), p.get("last_name"))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# BallDontLie: next scheduled game for a team
# ─────────────────────────────────────────────────────────────────────────────

def _bdl_next_game(bdl_team_id: int, player_team_abbrev: str) -> dict | None:
    """
    Fetch the next scheduled game for a team starting from today.

    Returns dict with game_date, opponent_abbrev, player_team_abbrev, home_game.
    """
    today_str = date.today().isoformat()

    # Use cached games if we already fetched this team (e.g. LeBron + Luka both on LAL)
    if bdl_team_id in _BDL_GAMES_CACHE:
        log.info("BDL games cache hit for team_id=%d", bdl_team_id)
        games = _BDL_GAMES_CACHE[bdl_team_id]
    else:
        url    = f"{BDL_BASE}/games"
        params = {
            "team_ids[]": bdl_team_id,
            "start_date": today_str,
            "per_page"  : 10,
        }
        try:
            resp = requests.get(url, headers=BDL_HEADERS, params=params, timeout=10)
            resp.raise_for_status()
            time.sleep(BDL_DELAY)
            games = resp.json().get("data", [])
            _BDL_GAMES_CACHE[bdl_team_id] = games
        except Exception as exc:
            log.error("BDL games fetch failed: %s", exc)
            return None

    if not games:
        log.warning("No BDL games found for team_id=%d from %s",
                    bdl_team_id, today_str)
        return None

    # Sort by date, take earliest
    def _parse_date(g):
        try:
            return datetime.strptime(g["date"][:10], "%Y-%m-%d")
        except Exception:
            return datetime(2099, 1, 1)

    games.sort(key=_parse_date)
    game = games[0]

    game_date   = _parse_date(game)
    home_abbrev = BDL_ABBREV_FIX.get(
        game.get("home_team", {}).get("abbreviation", "").upper(),
        game.get("home_team", {}).get("abbreviation", "").upper()
    )
    vis_abbrev  = BDL_ABBREV_FIX.get(
        game.get("visitor_team", {}).get("abbreviation", "").upper(),
        game.get("visitor_team", {}).get("abbreviation", "").upper()
    )

    team = player_team_abbrev.upper()
    if home_abbrev == team:
        home_game, opponent = True, vis_abbrev
    elif vis_abbrev == team:
        home_game, opponent = False, home_abbrev
    else:
        # Abbreviation mismatch — log and make best guess
        log.warning(
            "Team abbrev mismatch: expected '%s', BDL has home='%s' vis='%s'",
            team, home_abbrev, vis_abbrev,
        )
        home_game, opponent = True, vis_abbrev

    log.info("Next game: %s %s %s on %s",
             team, "vs" if home_game else "@", opponent,
             game_date.strftime("%Y-%m-%d"))

    return {
        "game_date"          : game_date,
        "opponent_abbrev"    : opponent,
        "player_team_abbrev" : team,
        "home_game"          : home_game,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NBA API: team stats
# ─────────────────────────────────────────────────────────────────────────────

def _get_team_stats() -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
        resp = LeagueDashTeamStats(
            season=CURRENT_SEASON,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        time.sleep(NBA_DELAY)
        df = resp.get_data_frames()[0]
    except Exception as exc:
        log.error("LeagueDashTeamStats failed: %s", exc)
        return pd.DataFrame()

    df = df.rename(columns={
        "TEAM_ID": "team_id", "TEAM_NAME": "team_name",
        "PACE": "pace", "OFF_RATING": "off_rating", "DEF_RATING": "def_rating",
    })
    abbrev_map = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
    df["team_abbrev"] = df["team_id"].map(abbrev_map)
    df["def_rank"]    = df["def_rating"].rank(method="average", ascending=True).astype(int)
    return df.set_index("team_abbrev")


# ─────────────────────────────────────────────────────────────────────────────
# NBA API: rest days
# ─────────────────────────────────────────────────────────────────────────────

def _get_rest_days(player_team_abbrev: str,
                   next_game_date: datetime) -> tuple[int, bool]:
    try:
        from nba_api.stats.endpoints import LeagueGameLog
        resp = LeagueGameLog(
            season=CURRENT_SEASON,
            direction="DESC",
            timeout=30,
        )
        time.sleep(NBA_DELAY)
        df = resp.get_data_frames()[0]
    except Exception as exc:
        log.error("LeagueGameLog failed: %s", exc)
        return 1, False

    team_games = df[
        df["TEAM_ABBREVIATION"].str.upper() == player_team_abbrev.upper()
    ].copy()

    if team_games.empty:
        return 1, False

    team_games["game_date_dt"] = pd.to_datetime(team_games["GAME_DATE"])
    past = team_games[
        team_games["game_date_dt"] < pd.Timestamp(next_game_date)
    ].sort_values("game_date_dt", ascending=False)

    if past.empty:
        return -1, False

    last_date = past.iloc[0]["game_date_dt"]
    rest_days = int((pd.Timestamp(next_game_date) - last_date).days) - 1
    rest_days = max(0, rest_days)
    return rest_days, rest_days == 0


# ─────────────────────────────────────────────────────────────────────────────
# NBA API: injury status (free — BDL /player_injuries requires paid tier)
# ─────────────────────────────────────────────────────────────────────────────

def _get_injury_status(player_name: str) -> dict:
    """
    Detect injury status by checking the player's recent game log for DNPs.

    How it works:
        1. Fetch the player's last 5 games via PlayerGameLog.
        2. If the most recent 2+ games show "Did Not Play" or 0 minutes,
           flag as likely injured.
        3. Also check CommonPlayerInfo for roster status.

    This catches real-world injuries (like Curry's knee) that only show up
    as DNPs in the game log, not in roster status fields.
    """
    try:
        from nba_api.stats.endpoints import PlayerGameLog, CommonPlayerInfo
        from nba_api.stats.static import players as nba_players_static

        results = nba_players_static.find_players_by_full_name(player_name)
        if not results:
            return _unknown_injury()

        nba_id = results[0]["id"]

        # Check CommonPlayerInfo first
        info_resp = CommonPlayerInfo(player_id=nba_id, timeout=15)
        time.sleep(NBA_DELAY)
        info_df = info_resp.get_data_frames()[0]

        if not info_df.empty:
            roster_status = str(info_df.iloc[0].get("ROSTERSTATUS", "")).strip().lower()
            if roster_status in ("inactive", "0", ""):
                return {
                    "status"    : "Inactive",
                    "reason"    : "Listed as inactive on NBA roster",
                    "is_injured": True,
                    "warning"   : (
                        player_name + " is currently INACTIVE on the NBA roster. "
                        "Verify availability before using this prediction."
                    ),
                }

        # Check recent game log for DNPs — catches in-season injuries
        log_resp = PlayerGameLog(
            player_id=nba_id,
            season="2025-26",
            timeout=15,
        )
        time.sleep(NBA_DELAY)
        log_df = log_resp.get_data_frames()[0]

        if log_df.empty:
            return _unknown_injury()

        # ── Check 1: look at minutes in recent games ────────────────────────
        # PlayerGameLog sometimes omits DNP rows entirely, so we also check
        # for large date gaps (player hasn't played in 5+ days = likely injured).
        recent = log_df.head(5)

        def _parse_mins(val) -> float:
            s = str(val).strip()
            if not s or s in ("", "None", "nan"):
                return 0.0
            # Format "33:24" -> 33.4
            if ":" in s:
                try:
                    m, sec = s.split(":")
                    return float(m) + float(sec) / 60
                except Exception:
                    return 0.0
            try:
                return float(s)
            except Exception:
                return 0.0

        # Count games with 0 minutes (DNPs that are included in the log)
        min_col = None
        for candidate in ["MIN", "MINUTES", "MP"]:
            if candidate in recent.columns:
                min_col = candidate
                break

        dnp_count = 0
        if min_col:
            for _, row in recent.iterrows():
                if _parse_mins(row[min_col]) < 1:
                    dnp_count += 1

        log.info("Injury check for %s: min_col=%s, dnp_count=%d in last 5 log rows",
                 player_name, min_col, dnp_count)

        # ── Check 2: date gap — last game was 5+ days ago ────────────────────
        # This catches cases where DNP rows are omitted from the log entirely
        days_since_last = 0
        date_col = None
        for candidate in ["GAME_DATE", "GAME_DATE_EST", "DATE"]:
            if candidate in log_df.columns:
                date_col = candidate
                break

        if date_col:
            try:
                last_game_date = pd.to_datetime(log_df.iloc[0][date_col])
                days_since_last = (pd.Timestamp.today() - last_game_date).days
                log.info("Days since last game for %s: %d", player_name, days_since_last)
            except Exception:
                pass

        # ── Determine status ─────────────────────────────────────────────────
        if dnp_count >= 2 or days_since_last >= 7:
            reason = (
                "Did not play in " + str(dnp_count) + " of last 5 games"
                if dnp_count >= 2
                else "Has not played in " + str(days_since_last) + " days"
            )
            return {
                "status"    : "Injured / Out",
                "reason"    : reason,
                "is_injured": True,
                "warning"   : (
                    player_name + " appears to be injured or resting (" + reason + "). "
                    "Check the latest injury report before using this prediction."
                ),
            }

        if dnp_count == 1 or (days_since_last >= 4 and days_since_last < 7):
            reason = (
                "Missed 1 of last 5 games"
                if dnp_count == 1
                else "Has not played in " + str(days_since_last) + " days"
            )
            return {
                "status"    : "Questionable",
                "reason"    : reason,
                "is_injured": False,
                "warning"   : (
                    player_name + " may be injured or resting (" + reason + "). "
                    "Verify they are confirmed active tonight."
                ),
            }

        return {"status": "Active", "reason": None, "is_injured": False, "warning": None}

    except Exception as exc:
        log.warning("Injury check failed for %s: %s", player_name, exc)
        return _unknown_injury()


def _unknown_injury() -> dict:
    return {
        "status"    : "Unknown",
        "reason"    : None,
        "is_injured": False,
        "warning"   : "Could not verify injury status — check injury reports before wagering.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main public function
# ─────────────────────────────────────────────────────────────────────────────

def get_next_game_context(player_name: str,
                          player_bbref_id: str,
                          position: str = "SF") -> dict | None:
    """
    Auto-fetch all prediction inputs for a player's next game.

    API calls made (in order):
        1. BDL /players      — find player + current team
        2. BDL /games        — next scheduled game for that team
        3. NBA LeagueDashTeamStats — opponent pace + def rating
        4. NBA LeagueGameLog       — rest days
        5. NBA CommonPlayerInfo    — injury status

    Returns a dict matching the /predict request schema, or None on failure.
    """
    log.info("=== Fetching next game context for %s ===", player_name)

    # 1. Find player on BDL
    bdl_player = _bdl_find_player(player_name)
    if bdl_player is None:
        log.error("'%s' not found on BallDontLie", player_name)
        return None

    bdl_player_id      = bdl_player["id"]
    bdl_team           = bdl_player.get("team", {})
    bdl_team_id        = bdl_team.get("id")
    player_team_abbrev = BDL_ABBREV_FIX.get(
        bdl_team.get("abbreviation", "UNK").upper(),
        bdl_team.get("abbreviation", "UNK").upper(),
    )

    log.info("BDL: %s plays for %s (team_id=%s)", player_name, player_team_abbrev, bdl_team_id)

    if not bdl_team_id:
        log.error("No team found for '%s'", player_name)
        return None

    # 2. Find next game
    next_game = _bdl_next_game(bdl_team_id, player_team_abbrev)
    if next_game is None:
        log.error("No upcoming games found for %s (%s)", player_name, player_team_abbrev)
        return None

    game_date       = next_game["game_date"]
    opponent_abbrev = next_game["opponent_abbrev"]
    home_game       = next_game["home_game"]

    # 3. Team stats
    team_stats     = _get_team_stats()
    opp_def_rating = 113.5
    opp_def_rank   = 15
    team_pace      = 99.5

    if not team_stats.empty:
        if opponent_abbrev in team_stats.index:
            opp_def_rating = float(team_stats.loc[opponent_abbrev, "def_rating"])
            opp_def_rank   = int(team_stats.loc[opponent_abbrev, "def_rank"])
        else:
            log.warning("Opponent '%s' not in team stats — using league average", opponent_abbrev)
        if player_team_abbrev in team_stats.index:
            team_pace = float(team_stats.loc[player_team_abbrev, "pace"])
        else:
            log.warning("Team '%s' not in team stats — using league average", player_team_abbrev)

    # 4. Rest days
    rest_days, back_to_back = _get_rest_days(player_team_abbrev, game_date)

    # 5. Injury status
    injury = _get_injury_status(player_name)

    log.info(
        "Context: %s %s %s | opp_def=%.1f (rank %d/30) | pace=%.1f | "
        "rest=%d | b2b=%s | injury=%s",
        player_team_abbrev, "vs" if home_game else "@", opponent_abbrev,
        opp_def_rating, opp_def_rank, team_pace,
        rest_days, back_to_back, injury["status"],
    )

    return {
        "player_name"         : player_name,
        "player_id"           : player_bbref_id,
        "opponent_abbrev"     : opponent_abbrev,
        "home_game"           : home_game,
        "opp_def_rating"      : round(opp_def_rating, 1),
        "opp_def_rank"        : opp_def_rank,
        "team_pace"           : round(team_pace, 1),
        "rest_days"           : rest_days,
        "back_to_back"        : back_to_back,
        "position"            : position,
        "_game_date"          : game_date.strftime("%b %d, %Y"),
        "_player_team_abbrev" : player_team_abbrev,
        "_injury_status"      : injury["status"],
        "_injury_warning"     : injury["warning"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
    )

    test_players = [
        ("LeBron James",   "jamesle01",  "SF"),
        ("Stephen Curry",  "curryst01",  "PG"),
        ("Luka Doncic",    "doncilu01",  "PG"),
    ]

    for name, bbref_id, pos in test_players:
        print(f"\n{'='*50}")
        ctx = get_next_game_context(name, bbref_id, pos)
        if ctx:
            print(f"  {ctx['player_name']:25s}  {ctx['_player_team_abbrev']} {'vs' if ctx['home_game'] else '@'} {ctx['opponent_abbrev']}  {ctx['_game_date']}")
            print(f"  Opp def: {ctx['opp_def_rating']} (rank {ctx['opp_def_rank']}/30)  Pace: {ctx['team_pace']}  Rest: {ctx['rest_days']}d  Injury: {ctx['_injury_status']}")
        else:
            print(f"  FAILED for {name}")