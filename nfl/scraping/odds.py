"""
NFL odds collector (The Odds API).
==================================
Fetches current NFL game totals + spreads (and optionally player props) and
derives each team's implied team total. Reuses the same ``ODDS_API_KEY`` as the
NBA project. Uses the NFL sport key ``americanfootball_nfl``.

The system MUST function without the key: when it is missing we return an empty
frame and callers omit live odds / mark them missing (never faked as 0).

Sportradar is documented as a possible FUTURE source for more timely injuries,
inactives, depth charts and live game data, but is intentionally NOT required.
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from nfl.config import DATA_RAW

log = logging.getLogger(__name__)

SPORT_KEY = "americanfootball_nfl"
API_BASE = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
ODDS_HISTORY = DATA_RAW / "nfl_odds_history.csv"

# Full team name -> nflverse abbreviation (The Odds API returns full names).
TEAM_ABBREV = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def has_api_key() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def _consensus(game: dict):
    totals, spreads = [], {}
    for bk in game.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] == "totals":
                for o in mk["outcomes"]:
                    if o.get("name") == "Over" and o.get("point") is not None:
                        totals.append(float(o["point"]))
            elif mk["key"] == "spreads":
                for o in mk["outcomes"]:
                    if o.get("point") is not None:
                        spreads.setdefault(o["name"], []).append(float(o["point"]))
    total = float(pd.Series(totals).median()) if totals else None
    team_spread = {n: float(pd.Series(v).median()) for n, v in spreads.items()}
    return total, team_spread


def fetch_odds(api_key: str | None = None) -> pd.DataFrame:
    """Return one row per team-game with implied totals. Empty when no key."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.info("ODDS_API_KEY not set — skipping NFL odds (no live lines).")
        return pd.DataFrame()
    try:
        resp = requests.get(API_BASE, params={
            "apiKey": api_key, "regions": "us",
            "markets": "totals,spreads", "oddsFormat": "american",
        }, timeout=30)
        resp.raise_for_status()
        games = resp.json()
    except Exception as exc:
        log.error("NFL odds fetch failed: %s", exc)
        return pd.DataFrame()

    rows = []
    for g in games:
        total, team_spread = _consensus(g)
        if total is None:
            continue
        home, away = g.get("home_team"), g.get("away_team")
        game_date = str(g.get("commence_time", ""))[:10]
        for team, opp in ((home, away), (away, home)):
            ta, oa = TEAM_ABBREV.get(team), TEAM_ABBREV.get(opp)
            spr = team_spread.get(team)
            if ta is None or oa is None or spr is None:
                continue
            rows.append({
                "game_date": game_date, "team_abbrev": ta, "opponent_abbrev": oa,
                "game_total": round(total, 1), "team_spread": round(spr, 1),
                "implied_team_total": round((total - spr) / 2, 1),
                "is_favorite": int(spr < 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
    return pd.DataFrame(rows)


def append_history(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if ODDS_HISTORY.exists():
        df = pd.concat([pd.read_csv(ODDS_HISTORY), df], ignore_index=True)
    df = (df.sort_values("fetched_at")
            .drop_duplicates(["game_date", "team_abbrev"], keep="last")
            .reset_index(drop=True))
    df.to_csv(ODDS_HISTORY, index=False)
    return len(df)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Fetch NFL odds -> implied totals")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    df = fetch_odds()
    if df.empty:
        log.info("No NFL odds (missing key or off-season).")
        return
    append_history(df)
    if args.show:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
