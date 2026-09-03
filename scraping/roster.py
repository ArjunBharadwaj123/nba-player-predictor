"""
Roster Auto-Collection
======================
Builds the player roster automatically from Basketball Reference's season
per-game page instead of a hand-maintained list. That page lists *every* player
who played that season, with their exact BBRef player IDs — the same IDs the
game-log scraper uses — so the roster and the scrape stay in sync.

Output:
    data/processed/roster.json   ->  [{"id","name","pos","games","mpg"}, ...]

The game-log scraper (bbref_scraper.py) and the API (/players) both read this,
so expanding the roster automatically expands the training data and the
frontend's searchable player list. No more hand-typed, duplicated lists.

Usage:
    python scraping/roster.py                       # current season, defaults
    python scraping/roster.py --season 2025
    python scraping/roster.py --min-games 15 --min-mpg 12
"""

import argparse
import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

# Reuse the scraper's session, caching fetch, and constants.
from bbref_scraper import BBREF_BASE, CURRENT_SEASON, fetch_html

ROOT      = Path(__file__).parent.parent
OUT_FILE  = ROOT / "data" / "processed" / "roster.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BBRef positions (incl. combos like "PG-SG") -> our five buckets. Take the
# player's primary (first) listed position.
VALID_POS = {"PG", "SG", "SF", "PF", "C"}


def _extract_per_game_table(raw_html: str):
    """Return the BeautifulSoup <table id='per_game_stats'>, handling the case
    where BBRef ships it inside an HTML comment."""
    soup = BeautifulSoup(raw_html, "lxml")
    table = soup.find("table", {"id": "per_game_stats"})
    if table is not None:
        return table
    # Comment-wrapped fallback (same trick the game-log scraper uses).
    for match in re.findall(r"<!--(.*?)-->", raw_html, flags=re.DOTALL):
        if "per_game_stats" in match:
            t = BeautifulSoup(match, "lxml").find("table", {"id": "per_game_stats"})
            if t is not None:
                return t
    return None


def _norm_pos(pos_str: str) -> str:
    primary = (pos_str or "").strip().upper().split("-")[0]
    return primary if primary in VALID_POS else "SF"


def build_roster(season: int = CURRENT_SEASON,
                 min_games: int = 20, min_mpg: float = 15.0,
                 force_refresh: bool = False) -> list[dict]:
    """Scrape the season per-game page and return a filtered, deduped roster."""
    url = f"{BBREF_BASE}/leagues/NBA_{season}_per_game.html"
    log.info("Building roster from %s", url)
    raw = fetch_html(url, "roster", season, force_refresh=force_refresh)

    table = _extract_per_game_table(raw)
    if table is None:
        raise RuntimeError("Could not find per_game_stats table on the season page.")

    players: dict[str, dict] = {}
    for tr in table.select("tbody tr"):
        # BBRef renamed the player cell stat to "name_display"; keep the old
        # "player" key as a fallback for older cached pages.
        cell = (tr.find("td", {"data-stat": "name_display"})
                or tr.find("td", {"data-stat": "player"}))
        if cell is None:      # section header / spacer row
            continue

        pid = cell.get("data-append-csv")
        if not pid:
            link = cell.find("a", href=True)
            m = re.search(r"/players/\w/(\w+)\.html", link["href"]) if link else None
            pid = m.group(1) if m else None
        if not pid:
            continue

        def _num(stat, default=0.0):
            td = tr.find("td", {"data-stat": stat})
            try:
                return float(td.get_text(strip=True))
            except (AttributeError, ValueError):
                return default

        games = _num("games", _num("g"))
        mpg   = _num("mp_per_g")
        pos   = tr.find("td", {"data-stat": "pos"})
        entry = {
            "id":   pid,
            "name": cell.get_text(strip=True),
            "pos":  _norm_pos(pos.get_text(strip=True) if pos else ""),
            "games": int(games),
            "mpg":  round(mpg, 1),
        }
        # A traded player has multiple rows (per team + a combined "2TM/3TM"
        # total). Keep whichever row has the most games — that's the aggregate.
        prev = players.get(pid)
        if prev is None or entry["games"] > prev["games"]:
            players[pid] = entry

    roster = [
        p for p in players.values()
        if p["games"] >= min_games and p["mpg"] >= min_mpg
    ]
    roster.sort(key=lambda p: p["name"])
    log.info("  %d players scraped, %d kept (>= %d games, >= %.0f mpg)",
             len(players), len(roster), min_games, min_mpg)
    return roster


def save_roster(roster: list[dict]) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(roster, indent=2))
    log.info("Saved -> %s (%d players)", OUT_FILE.relative_to(ROOT), len(roster))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-build the player roster from Basketball Reference.")
    ap.add_argument("--season", type=int, default=CURRENT_SEASON)
    ap.add_argument("--min-games", type=int, default=20)
    ap.add_argument("--min-mpg", type=float, default=15.0)
    ap.add_argument("--force-refresh", action="store_true",
                    help="Ignore the cached season page and re-fetch.")
    args = ap.parse_args()

    roster = build_roster(args.season, args.min_games, args.min_mpg, args.force_refresh)
    save_roster(roster)
