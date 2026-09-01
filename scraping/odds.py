"""
Vegas odds collector (The Odds API)
===================================
Fetches NBA game totals + spreads and derives each team's IMPLIED TEAM TOTAL —
typically the single strongest external predictor of counting-stat output (it
encodes pace, blowout risk and team strength in one number).

    implied_team_total = (game_total - team_spread) / 2

    e.g. total 220, home favoured by 3 (home_spread = -3):
         home implied = (220 - (-3)) / 2 = 111.5
         away implied = (220 -  3 ) / 2 = 108.5   (sum = 220, margin = 3)

IMPORTANT — no historical odds on the free tier. This collector captures
upcoming/current lines only. Two consequences:
  • As a MODEL feature it does nothing until enough history accumulates: every
    historical training row is NaN, so the model can't learn from it yet. Run
    this daily (wired into pipeline/update.py) so odds history builds up; a
    future retrain then picks the feature up automatically (it's nullable, and
    XGBoost handles NaN).
  • Right now its value is SERVE-TIME CONTEXT — next_game.py surfaces tonight's
    implied team total / spread alongside the prediction.

The API key is read from the ODDS_API_KEY environment variable — never hard-code
or commit it.

Usage:
    export ODDS_API_KEY=...          # your the-odds-api.com key
    python scraping/odds.py           # fetch + append to data/raw/odds_history.csv
"""

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

try:                       # load ODDS_API_KEY from a local .env if present
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from nba_api.stats.static import teams as nba_teams_static

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
ODDS_HISTORY = RAW / "odds_history.csv"

API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Full team name -> NBA abbreviation (BOS, LAL, ...), from nba_api's static list.
_NAME_TO_ABBREV = {t["full_name"]: t["abbreviation"] for t in nba_teams_static.get_teams()}


def _abbrev(full_name: str) -> str | None:
    if full_name in _NAME_TO_ABBREV:
        return _NAME_TO_ABBREV[full_name]
    # Fallback: match on nickname (last word), e.g. "LA Clippers" quirks.
    for name, ab in _NAME_TO_ABBREV.items():
        if name.split()[-1] == full_name.split()[-1]:
            return ab
    return None


def _consensus(game: dict) -> tuple[float | None, dict]:
    """Median game total and per-team spread across all bookmakers for one game."""
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
    team_spread = {name: float(pd.Series(v).median()) for name, v in spreads.items()}
    return total, team_spread


def fetch_odds(api_key: str | None = None) -> pd.DataFrame:
    """Fetch current NBA lines and return one row per team-game with implied totals."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        log.error("ODDS_API_KEY not set — cannot fetch odds.")
        return pd.DataFrame()

    try:
        resp = requests.get(API_BASE, params={
            "apiKey": api_key, "regions": "us",
            "markets": "totals,spreads", "oddsFormat": "american",
        }, timeout=30)
        resp.raise_for_status()
        games = resp.json()
    except Exception as exc:
        log.error("Odds fetch failed: %s", exc)
        return pd.DataFrame()

    remaining = resp.headers.get("x-requests-remaining")
    log.info("Fetched %d games (API requests remaining: %s)", len(games), remaining)

    rows = []
    for g in games:
        total, team_spread = _consensus(g)
        if total is None:
            continue
        home, away = g.get("home_team"), g.get("away_team")
        game_date = str(g.get("commence_time", ""))[:10]
        for team, opp in ((home, away), (away, home)):
            ta, oa = _abbrev(team), _abbrev(opp)
            spr = team_spread.get(team)
            if ta is None or oa is None or spr is None:
                continue
            rows.append({
                "game_date": game_date,
                "team_abbrev": ta,
                "opponent_abbrev": oa,
                "game_total": round(total, 1),
                "team_spread": round(spr, 1),
                "implied_team_total": round((total - spr) / 2, 1),
                "is_favorite": int(spr < 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info("Parsed %d team-game odds rows (%d games)",
                 len(df), df["game_date"].nunique())
    return df


def append_history(df: pd.DataFrame) -> int:
    """Append fresh odds to odds_history.csv, keeping the LATEST line per
    (game_date, team_abbrev). Returns rows written."""
    if df.empty:
        return 0
    if ODDS_HISTORY.exists():
        prev = pd.read_csv(ODDS_HISTORY)
        df = pd.concat([prev, df], ignore_index=True)
    df = (
        df.sort_values("fetched_at")
        .drop_duplicates(subset=["game_date", "team_abbrev"], keep="last")
        .sort_values(["game_date", "team_abbrev"])
        .reset_index(drop=True)
    )
    df.to_csv(ODDS_HISTORY, index=False)
    log.info("odds_history.csv now %d rows across %d game-dates",
             len(df), df["game_date"].nunique())
    return len(df)


def main():
    ap = argparse.ArgumentParser(description="Fetch NBA odds -> implied team totals.")
    ap.add_argument("--show", action="store_true", help="Print the fetched rows.")
    args = ap.parse_args()
    df = fetch_odds()
    if df.empty:
        log.info("No odds rows fetched (off-season or no lines posted yet).")
        return
    append_history(df)
    if args.show:
        cols = ["game_date", "team_abbrev", "opponent_abbrev",
                "game_total", "team_spread", "implied_team_total", "is_favorite"]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
