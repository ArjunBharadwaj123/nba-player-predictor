"""
Step 4 (comment-table fix): Basketball Reference Game Log Scraper
=================================================================
Fixes applied vs previous version:
  1. Robust comment-unwrapping — uses regex on the raw HTML string to extract
     the pgl_basic table whether it's in a comment or not.
  2. Increased timeout (30 s) + retry logic for network timeouts.
  3. Jitter on all sleeps to avoid metronomic request patterns.

Usage:
    python scraping/bbref_scraper.py --mode train     # historical 2022-2025
    python scraping/bbref_scraper.py --mode current   # 2025-26, always fresh
    python scraping/bbref_scraper.py --mode all       # both
"""

import argparse
import logging
import re
import time
import random
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

try:
    import cloudscraper
    _CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    _CLOUDSCRAPER_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
HTML_CACHE   = ROOT / "data" / "raw" / "html"
GAMELOGS_DIR = ROOT / "data" / "raw" / "gamelogs"
TRAIN_OUT    = ROOT / "data" / "raw" / "all_gamelogs.csv"
CURRENT_OUT  = ROOT / "data" / "raw" / "current_season.csv"

for d in (HTML_CACHE, GAMELOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BBREF_BASE       = "https://www.basketball-reference.com"
SCRAPE_DELAY     = 4.0
JITTER           = 1.5
REQUEST_TIMEOUT  = 30         # increased from 15 — Beal 2024 needs more time
MAX_RETRIES      = 3
TRAINING_SEASONS = [2022, 2023, 2024, 2025]
CURRENT_SEASON   = 2026

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Browser headers ───────────────────────────────────────────────────────────
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
}


def _make_session():
    if _CLOUDSCRAPER_AVAILABLE:
        log.info("Using cloudscraper session")
        s = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "darwin", "mobile": False}
        )
        s.headers.update(BROWSER_HEADERS)
        return s
    log.info("Using requests session with browser headers")
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    return s


def _warm_up_session(session) -> None:
    log.info("Warming up session (visiting BBRef homepage)...")
    try:
        resp = session.get(BBREF_BASE, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        log.info(f"  Homepage OK ({resp.status_code})")
        time.sleep(2.0 + random.uniform(0, 1.0))
    except Exception as exc:
        log.warning(f"  Warmup failed (continuing): {exc}")


SESSION = _make_session()

# ── Column rename map ─────────────────────────────────────────────────────────
# Covers BOTH table layouts BBRef uses:
#   player_game_log_reg  (current layout, confirmed via DOM inspector)
#   pgl_basic            (older pages)
# If parse_gamelog() logs "Unmapped columns", add the raw header name here.
# The scraper will NOT crash on unknown columns — they pass through as-is.
COLUMN_MAP = {
    # Core stats — same across both layouts
    "Rk"         : "rank",
    "G"          : "game_num",
    "Date"       : "game_date",
    "Age"        : "age",
    "Tm"         : "team",
    "Team"       : "team",
    "Opp"        : "opponent",
    "GS"         : "gs",
    "MP"         : "mp",
    "FG"         : "fg",
    "FGA"        : "fga",
    "FG%"        : "fg_pct",
    "3P"         : "fg3",
    "3PA"        : "fg3a",
    "3P%"        : "fg3_pct",
    "FT"         : "ft",
    "FTA"        : "fta",
    "FT%"        : "ft_pct",
    "ORB"        : "orb",
    "DRB"        : "drb",
    "TRB"        : "trb",
    "AST"        : "ast",
    "STL"        : "stl",
    "BLK"        : "blk",
    "TOV"        : "tov",
    "PF"         : "pf",
    "PTS"        : "pts",
    "+/-"        : "plus_minus",
    "GmSc"       : "game_score",
    # pgl_basic unnamed columns (home/away and result)
    "Unnamed: 5" : "home_away",
    "Unnamed: 7" : "result",
    # player_game_log_reg uses different unnamed column positions
    "Unnamed: 3" : "home_away",
    "Unnamed: 4" : "home_away",
    "Unnamed: 6" : "result",
    # Some pages use explicit labels
    "W/L"        : "result",
    # Newly discovered columns from player_game_log_reg
    "Gcar"       : "game_num_career",
    "Gtm"        : "game_num_team",
    "Result"     : "result",
    "2P"         : "fg2",
    "2PA"        : "fg2a",
    "2P%"        : "fg2_pct",
    "eFG%"       : "efg_pct",
}

# ── Players ───────────────────────────────────────────────────────────────────
PLAYERS = [
 
    # ── POINT GUARDS (34) ────────────────────────────────────────────────────
    {"id": "curryst01",  "name": "Stephen Curry",           "pos": "PG"},
    {"id": "doncilu01",  "name": "Luka Doncic",             "pos": "PG"},
    {"id": "gilgesh01",  "name": "Shai Gilgeous-Alexander", "pos": "PG"},
    {"id": "lillada01",  "name": "Damian Lillard",          "pos": "PG"},
    {"id": "holidjr01",  "name": "Jrue Holiday",            "pos": "PG"},
    {"id": "hardeja01",  "name": "James Harden",            "pos": "PG"},
    {"id": "youngtr01",  "name": "Trae Young",              "pos": "PG"},
    {"id": "foxde01",    "name": "De'Aaron Fox",            "pos": "PG"},
    {"id": "murraja01",  "name": "Jamal Murray",            "pos": "PG"},
    {"id": "moranja01",  "name": "Ja Morant",               "pos": "PG"},
    {"id": "halibty01",  "name": "Tyrese Haliburton",       "pos": "PG"},
    {"id": "mcbrimi01",  "name": "Miles McBride",           "pos": "PG"},
    {"id": "garlada01",  "name": "Darius Garland",          "pos": "PG"},
    {"id": "irvinky01",  "name": "Kyrie Irving",            "pos": "PG"},
    {"id": "paulch01",   "name": "Chris Paul",              "pos": "PG"},
    {"id": "smartma01",  "name": "Marcus Smart",            "pos": "PG"},
    {"id": "sextoco01",  "name": "Collin Sexton",           "pos": "PG"},
    {"id": "iveyja01",   "name": "Jaden Ivey",              "pos": "PG"},
    {"id": "pritcpa01",  "name": "Payton Pritchard",        "pos": "PG"},
    {"id": "schrode01",  "name": "Dennis Schroder",         "pos": "PG"},
    {"id": "roziete01",  "name": "Terry Rozier",            "pos": "PG"},
    {"id": "georgke01",  "name": "Keyonte George",          "pos": "PG"},
    {"id": "murrade01",  "name": "Dejounte Murray",         "pos": "PG"},
    {"id": "mccolcj01",  "name": "CJ McCollum",             "pos": "PG"},
    {"id": "millspa02",  "name": "Patty Mills",             "pos": "PG"},
    {"id": "clarkjo02",  "name": "Jordan Clarkson",         "pos": "PG"},
    {"id": "anthoca02",  "name": "Cole Anthony",            "pos": "PG"},
    {"id": "dinwspe01",  "name": "Spencer Dinwiddie",       "pos": "PG"},
    {"id": "vincega01",  "name": "Gabe Vincent",            "pos": "PG"},
    {"id": "giddejo01",  "name": "Josh Giddey",             "pos": "PG"},
    {"id": "quickim01",  "name": "Immanuel Quickley",       "pos": "PG"},
    {"id": "leverca01",  "name": "Caris LeVert",            "pos": "PG"},
    {"id": "fultzma01",  "name": "Markelle Fultz",          "pos": "PG"},
    {"id": "balllo01",   "name": "Lonzo Ball",              "pos": "PG"},
 
    # ── SHOOTING GUARDS (24) ─────────────────────────────────────────────────
    {"id": "bookede01",  "name": "Devin Booker",            "pos": "SG"},
    {"id": "edwaran01",  "name": "Anthony Edwards",         "pos": "SG"},
    {"id": "mitchdo01",  "name": "Donovan Mitchell",        "pos": "SG"},
    {"id": "bealbr01",   "name": "Bradley Beal",            "pos": "SG"},
    {"id": "maxeyty01",  "name": "Tyrese Maxey",            "pos": "SG"},
    {"id": "brownja02",  "name": "Jaylen Brown",            "pos": "SG"},
    {"id": "lavinza01",  "name": "Zach LaVine",             "pos": "SG"},
    {"id": "hartjo01",   "name": "Josh Hart",               "pos": "SG"},
    {"id": "herroty01",  "name": "Tyler Herro",             "pos": "SG"},
    {"id": "greenja05",  "name": "Jalen Green",             "pos": "SG"},
    {"id": "thompkl01",  "name": "Klay Thompson",           "pos": "SG"},
    {"id": "monkma01",   "name": "Malik Monk",              "pos": "SG"},
    {"id": "bogdabo01",  "name": "Bogdan Bogdanovic",       "pos": "SG"},
    {"id": "brogdma01",  "name": "Malcolm Brogdon",         "pos": "SG"},
    {"id": "sharpsh01",  "name": "Shaedon Sharpe",          "pos": "SG"},
    {"id": "banede01",   "name": "Desmond Bane",            "pos": "SG"},
    {"id": "matthwe01",  "name": "Wesley Matthews",         "pos": "SG"},
    {"id": "connapa01",  "name": "Pat Connaughton",         "pos": "SG"},
    {"id": "vassede01",  "name": "Devin Vassell",           "pos": "SG"},
    {"id": "mykhasv01",  "name": "Svi Mykhailiuk",          "pos": "SG"},
    {"id": "trentga02",  "name": "Gary Trent Jr.",          "pos": "SG"},
    {"id": "willizi01",  "name": "Ziaire Williams",         "pos": "SG"},
    {"id": "elliske01",  "name": "Keon Ellis",              "pos": "SG"},
    {"id": "nesmiaa01",  "name": "Aaron Nesmith",           "pos": "SG"},
 
    # ── SMALL FORWARDS (29) ──────────────────────────────────────────────────
    {"id": "jamesle01",  "name": "LeBron James",            "pos": "SF"},
    {"id": "duranke01",  "name": "Kevin Durant",            "pos": "SF"},
    {"id": "tatumja01",  "name": "Jayson Tatum",            "pos": "SF"},
    {"id": "georgpa01",  "name": "Paul George",             "pos": "SF"},
    {"id": "leonaka01",  "name": "Kawhi Leonard",           "pos": "SF"},
    {"id": "anunoog01",  "name": "OG Anunoby",              "pos": "SF"},
    {"id": "huntede01",  "name": "De'Andre Hunter",         "pos": "SF"},
    {"id": "princta01",  "name": "Taurean Prince",          "pos": "SF"},
    {"id": "carusal01",  "name": "Alex Caruso",             "pos": "SF"},
    {"id": "joneshe01",  "name": "Herbert Jones",           "pos": "SF"},
    {"id": "batumni01",  "name": "Nic Batum",               "pos": "SF"},
    {"id": "ingrabr01",  "name": "Brandon Ingram",          "pos": "SF"},
    {"id": "wiggian01",  "name": "Andrew Wiggins",          "pos": "SF"},
    {"id": "onealro01",  "name": "Royce O'Neale",           "pos": "SF"},
    {"id": "hardati01",  "name": "Tim Hardaway Jr.",        "pos": "SF"},
    {"id": "middlkh01",  "name": "Khris Middleton",         "pos": "SF"},
    {"id": "barneha01",  "name": "Harrison Barnes",         "pos": "SF"},
    {"id": "bridgmi01",  "name": "Mikal Bridges",           "pos": "SF"},
    {"id": "portibo01",  "name": "Bobby Portis",            "pos": "SF"},
    {"id": "kuminga01",  "name": "Jonathan Kuminga",        "pos": "SF"},
    {"id": "mcdanja01",  "name": "Jalen McDaniels",         "pos": "SF"},
    {"id": "knechda01",  "name": "Dalton Knecht",           "pos": "SF"},
    {"id": "diengou01",  "name": "Ousmane Dieng",           "pos": "SF"},
    {"id": "willipa05",  "name": "Patrick Williams",        "pos": "SF"},
    {"id": "niangge01",  "name": "Georges Niang",           "pos": "SF"},
    {"id": "johnsca02",  "name": "Cam Johnson",             "pos": "SF"},
    {"id": "greendra01", "name": "Draymond Green",          "pos": "SF"},
    {"id": "nancela02",  "name": "Larry Nance Jr.",         "pos": "SF"},
    {"id": "thybumat01", "name": "Matisse Thybulle",        "pos": "SF"},
 
    # ── POWER FORWARDS (19) ──────────────────────────────────────────────────
    {"id": "antetgi01",  "name": "Giannis Antetokounmpo",   "pos": "PF"},
    {"id": "siakapa01",  "name": "Pascal Siakam",           "pos": "PF"},
    {"id": "grantjer01", "name": "Jerami Grant",            "pos": "PF"},
    {"id": "randlju01",  "name": "Julius Randle",           "pos": "PF"},
    {"id": "adebaba01",  "name": "Bam Adebayo",             "pos": "PF"},
    {"id": "marshna01",  "name": "Naji Marshall",           "pos": "PF"},
    {"id": "johnske04",  "name": "Keldon Johnson",          "pos": "PF"},
    {"id": "champju01",  "name": "Justin Champagnie",       "pos": "PF"},
    {"id": "collijo01",  "name": "John Collins",            "pos": "PF"},
    {"id": "achiupr01",  "name": "Precious Achiuwa",        "pos": "PF"},
    {"id": "portemi01",  "name": "Michael Porter Jr.",      "pos": "PF"},
    {"id": "stewais01",  "name": "Isaiah Stewart",          "pos": "PF"},
    {"id": "washipa02",  "name": "PJ Washington",           "pos": "PF"},
    {"id": "murrake01",  "name": "Keegan Murray",           "pos": "PF"},
    {"id": "willizi02",  "name": "Zion Williamson",         "pos": "PF"},
    {"id": "hernawi01",  "name": "Willy Hernangomez",       "pos": "PF"},
    {"id": "bitadgo01",  "name": "Goga Bitadze",            "pos": "PF"},
    {"id": "vandeja01",  "name": "Jarred Vanderbilt",       "pos": "PF"},
    {"id": "toppiobi01", "name": "Obi Toppin",              "pos": "PF"},
 
    # ── CENTERS (28) ─────────────────────────────────────────────────────────
    {"id": "jokicni01",  "name": "Nikola Jokic",            "pos": "C"},
    {"id": "embiijo01",  "name": "Joel Embiid",             "pos": "C"},
    {"id": "wembavi01",  "name": "Victor Wembanyama",       "pos": "C"},
    {"id": "goberru01",  "name": "Rudy Gobert",             "pos": "C"},
    {"id": "townska01",  "name": "Karl-Anthony Towns",      "pos": "C"},
    {"id": "davisan02",  "name": "Anthony Davis",           "pos": "C"},
    {"id": "sabondo01",  "name": "Domantas Sabonis",        "pos": "C"},
    {"id": "capelcl01",  "name": "Clint Capela",            "pos": "C"},
    {"id": "harteis01",  "name": "Isaiah Hartenstein",      "pos": "C"},
    {"id": "looneke01",  "name": "Kevon Looney",            "pos": "C"},
    {"id": "robinmi02",  "name": "Mitchell Robinson",       "pos": "C"},
    {"id": "durenja01",  "name": "Jalen Duren",             "pos": "C"},
    {"id": "sengual01",  "name": "Alperen Sengun",          "pos": "C"},
    {"id": "vucevni01",  "name": "Nikola Vucevic",          "pos": "C"},
    {"id": "holmgch01",  "name": "Chet Holmgren",           "pos": "C"},
    {"id": "adamsst01",  "name": "Steven Adams",            "pos": "C"},
    {"id": "bambamo01",  "name": "Mo Bamba",                "pos": "C"},
    {"id": "claxtni01",  "name": "Nicolas Claxton",         "pos": "C"},
    {"id": "zubaciv01",  "name": "Ivica Zubac",             "pos": "C"},
    {"id": "wisemja01",  "name": "James Wiseman",           "pos": "C"},
    {"id": "poeltja01",  "name": "Jakob Poeltl",            "pos": "C"},
    {"id": "willima07",  "name": "Mark Williams",           "pos": "C"},
    {"id": "smitija04",  "name": "Jalen Smith",             "pos": "C"},
    {"id": "plumlma01",  "name": "Mason Plumlee",           "pos": "C"},
    {"id": "gaffodan01", "name": "Daniel Gafford",          "pos": "C"},
    {"id": "bolbo01",    "name": "Bol Bol",                 "pos": "C"},
    {"id": "kesslwa01",  "name": "Walker Kessler",          "pos": "C"},
    {"id": "colliza01",  "name": "Zach Collins",            "pos": "C"},
]
 
# Quick validation
if __name__ == "__main__":
    from collections import Counter
    ids   = [p["id"] for p in PLAYERS]
    names = [p["name"] for p in PLAYERS]
    dup_ids   = {k for k, v in Counter(ids).items() if v > 1}
    dup_names = {k for k, v in Counter(names).items() if v > 1}
    pos_counts = Counter(p["pos"] for p in PLAYERS)
    print(f"Total players : {len(PLAYERS)}")
    print(f"By position   : {dict(pos_counts)}")
    if dup_ids:   print(f"Duplicate IDs  : {dup_ids}")
    if dup_names: print(f"Duplicate names: {dup_names}")
    if not dup_ids and not dup_names:
        print("No duplicates found.")

# ─────────────────────────────────────────────────────────────────────────────
# THE KEY FIX: robust table extraction from HTML comments
# ─────────────────────────────────────────────────────────────────────────────

def _extract_table_html(raw_html: str) -> str | None:
    """
    Extract the raw HTML of the game log table from a BBRef page.

    BBRef uses TWO different table IDs depending on the page version:
        id="player_game_log_reg"   ← current layout (confirmed via inspector)
        id="pgl_basic"             ← older layout, still appears on some pages

    Strategy:
        1. Parse with BeautifulSoup and try both known IDs (fast path, live DOM)
        2. Regex on raw HTML for comment-wrapped tables containing either ID
        3. BeautifulSoup Comment node fallback (belt-and-suspenders)

    Returns the HTML string of the table element, or None if not found.
    """
    # All known table IDs BBRef has used for game logs
    TABLE_IDS = ["player_game_log_reg", "pgl_basic"]

    soup = BeautifulSoup(raw_html, "lxml")

    # Strategy 1: table is live in the DOM
    for tid in TABLE_IDS:
        table = soup.find("table", {"id": tid})
        if table is not None:
            log.info(f"  Found table id='{tid}' in live DOM")
            return str(table)

    # Strategy 2: table is inside an HTML comment — regex on raw string
    for tid in TABLE_IDS:
        pattern = re.compile(rf"<!--(.*?{re.escape(tid)}.*?)-->", re.DOTALL)
        for match in pattern.findall(raw_html):
            comment_soup = BeautifulSoup(match, "lxml")
            table = comment_soup.find("table", {"id": tid})
            if table is not None:
                log.info(f"  Found table id='{tid}' inside HTML comment")
                return str(table)

    # Strategy 3: BeautifulSoup Comment nodes fallback
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        for tid in TABLE_IDS:
            if tid in comment:
                comment_soup = BeautifulSoup(str(comment), "lxml")
                table = comment_soup.find("table", {"id": tid})
                if table is not None:
                    log.info(f"  Found table id='{tid}' via Comment node fallback")
                    return str(table)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Fetch with caching, 403 handling, and timeout retries
# ─────────────────────────────────────────────────────────────────────────────

def fetch_html(url: str, player_id: str, season: int,
               force_refresh: bool = False) -> str:
    """
    Fetch a BBRef page with:
      - Disk caching (skips network if file exists and not force_refresh)
      - 403 recovery (re-warms session, retries once)
      - 429 backoff (60 s sleep then retry)
      - Timeout retries with exponential backoff (fixes the Beal 2024 issue)
    """
    cache_path = HTML_CACHE / f"{player_id}_{season}.html"

    if force_refresh:
        cache_path.unlink(missing_ok=True)

    if cache_path.exists():
        log.info(f"  Cache hit: {cache_path.name}")
        return cache_path.read_text(encoding="utf-8")

    referer = f"{BBREF_BASE}/players/{player_id[0]}/{player_id}.html"
    SESSION.headers.update({"Referer": referer})

    log.info(f"  Fetching: {url}")

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 403:
                log.warning(f"  403 on attempt {attempt} — re-warming session...")
                time.sleep(15)
                _warm_up_session(SESSION)
                time.sleep(3)
                continue

            if resp.status_code == 429:
                log.warning(f"  429 rate limit — sleeping 60 s...")
                time.sleep(60)
                continue

            resp.raise_for_status()

            # Success — save and return
            cache_path.write_text(resp.text, encoding="utf-8")
            sleep_time = SCRAPE_DELAY + random.uniform(0, JITTER)
            log.info(f"  Saved to cache — sleeping {sleep_time:.1f} s")
            time.sleep(sleep_time)
            return resp.text

        except requests.exceptions.Timeout as exc:
            last_exc = exc
            backoff = 10 * attempt   # 10 s, 20 s, 30 s
            log.warning(
                f"  Timeout on attempt {attempt}/{MAX_RETRIES} "
                f"— retrying in {backoff} s..."
            )
            time.sleep(backoff)
            continue

        except requests.exceptions.RequestException as exc:
            last_exc = exc
            log.warning(f"  Request error on attempt {attempt}: {exc}")
            time.sleep(10 * attempt)
            continue

    raise requests.exceptions.RetryError(
        f"All {MAX_RETRIES} attempts failed for {url}. Last error: {last_exc}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parse game log
# ─────────────────────────────────────────────────────────────────────────────

def _parse_minutes(mp_str: str) -> float:
    mp_str = str(mp_str).strip()
    if ":" in mp_str:
        try:
            mins, secs = mp_str.split(":")
            return int(mins) + int(secs) / 60
        except ValueError:
            pass
    try:
        return float(mp_str)
    except ValueError:
        return float("nan")


def parse_gamelog(raw_html: str, player_id: str,
                  player_name: str, season: int,
                  position: str = "") -> pd.DataFrame:
    """
    Parse the game log table from a raw BBRef HTML page.
    Uses _extract_table_html() which handles both live and comment-wrapped tables.
    """
    from io import StringIO

    table_html = _extract_table_html(raw_html)

    if table_html is None:
        log.warning(f"  No game log table found for {player_name} {season}")
        log.warning(
            f"  Tip: delete data/raw/html/{player_id}_{season}.html and re-run."
        )
        return pd.DataFrame()

    # StringIO wrapper silences the pandas FutureWarning
    df = pd.read_html(StringIO(table_html), flavor="lxml")[0]

    # Drop repeated header rows BBRef injects every ~20 rows
    df = df[df.iloc[:, 0] != "Rk"].copy()
    df.columns = [str(c) for c in df.columns]

    # ── Diagnostic: log raw columns so we can debug mapping issues ────────────
    log.debug(f"  Raw columns: {list(df.columns)}")

    df = df.rename(columns=COLUMN_MAP)

    # ── home_game ─────────────────────────────────────────────────────────────
    # BBRef encodes away games as "@" in an unlabelled column.
    # After renaming, that column is "home_away" (if the rename matched) or
    # it may still be its raw unnamed header. Check both.
    home_col = None
    if "home_away" in df.columns:
        home_col = "home_away"
    else:
        # Fallback: find any column whose values are only "@" or blank/NaN
        for col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().unique()
            if set(vals).issubset({"@", "", "nan"}):
                home_col = col
                log.info(f"  Detected home/away column as '{col}'")
                break

    if home_col:
        df["home_game"] = (df[home_col].astype(str).str.strip() != "@").astype(int)
        df = df.drop(columns=[home_col])
    else:
        df["home_game"] = 1

    # ── Drop summary / total rows ────────────────────────────────────────────
    # BBRef appends cumulative total rows at the bottom of each game log.
    # These have no date and inflate per-game stat averages significantly.
    # Identified by: missing/unparseable date OR "TOT" in the rank column.
    _date_col = "game_date" if "game_date" in df.columns else None
    _rank_col = "rank" if "rank" in df.columns else None
    _summary_mask = pd.Series(False, index=df.index)
    if _date_col:
        _summary_mask |= pd.to_datetime(df[_date_col], errors="coerce").isna()
    if _rank_col:
        _summary_mask |= df[_rank_col].astype(str).str.strip().isin(["TOT", "Total", ""])
    _n_summary = int(_summary_mask.sum())
    df = df[~_summary_mask].copy()
    if _n_summary:
        log.info(f"  Dropped {_n_summary} summary/total rows")

    # ── Drop DNP / inactive rows ──────────────────────────────────────────────
    # These rows have text like "Did Not Play" in the minutes column.
    # The minutes column may be named "mp" (after rename) or "MP" (raw).
    mp_col = "mp" if "mp" in df.columns else ("MP" if "MP" in df.columns else None)
    if mp_col:
        dnp = df[mp_col].astype(str).str.contains(
            r"Did Not|Inactive|Not With|Suspended", case=False, na=True
        )
        n_dropped = int(dnp.sum())
        df = df[~dnp].copy()
        if n_dropped:
            log.info(f"  Dropped {n_dropped} DNP rows")

    # ── Metadata ──────────────────────────────────────────────────────────────
    df["player_id"]   = player_id
    df["player_name"] = player_name
    df["position"]    = position
    df["season"]      = season

    # ── Numeric cast ──────────────────────────────────────────────────────────
    NUMERIC = [
        "game_num", "gs", "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
        "ft", "fta", "ft_pct", "orb", "drb", "trb", "ast", "stl", "blk",
        "tov", "pf", "pts", "plus_minus", "game_score",
    ]
    for col in NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Minutes ───────────────────────────────────────────────────────────────
    if mp_col and mp_col in df.columns:
        df["minutes"] = df[mp_col].apply(_parse_minutes)
        df = df.drop(columns=[mp_col])

    # ── Date ──────────────────────────────────────────────────────────────────
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # ── Column ordering — only include cols that actually exist ───────────────
    # This was the crash: we listed "team" and "result" in lead even when
    # the rename hadn't mapped them (because the raw header differs).
    # Now we only put a column in lead if it's present in the DataFrame.
    desired_lead = ["player_id", "player_name", "position", "season", "game_date",
                    "team", "home_game", "opponent", "result", "minutes"]
    lead = [c for c in desired_lead if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    df = df[lead + rest].reset_index(drop=True)

    # ── Log any unmapped columns so we can extend COLUMN_MAP if needed ────────
    expected = {"player_id", "player_name", "position", "season", "game_date", "team",
                "home_game", "opponent", "result", "minutes",
                "fg", "fga", "fg_pct", "fg3", "fg3a", "fg3_pct",
                "ft", "fta", "ft_pct", "orb", "drb", "trb",
                "ast", "stl", "blk", "tov", "pf", "pts",
                "plus_minus", "game_score", "rank", "game_num", "age", "gs",
                "game_num_career", "game_num_team", "fg2", "fg2a", "fg2_pct", "efg_pct"}
    unmapped = [c for c in df.columns if c not in expected]
    if unmapped:
        log.info(f"  Unmapped columns (extend COLUMN_MAP if needed): {unmapped}")

    log.info(f"  Parsed {len(df)} games — {player_name} ({season})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Single player/season
# ─────────────────────────────────────────────────────────────────────────────

def scrape_one(player: dict, season: int,
               force_refresh: bool = False) -> pd.DataFrame:
    pid, name = player["id"], player["name"]
    log.info(f"Processing {name} — season {season}")
    try:
        pos  = player.get("pos", "")
        url  = f"{BBREF_BASE}/players/{pid[0]}/{pid}/gamelog/{season}"
        html = fetch_html(url, pid, season, force_refresh=force_refresh)
        df   = parse_gamelog(html, pid, name, season, position=pos)
    except Exception as exc:
        log.error(f"  FAILED {name} {season}: {exc}")
        return pd.DataFrame()

    if not df.empty:
        out_path = GAMELOGS_DIR / f"{pid}_{season}.csv"
        df.to_csv(out_path, index=False)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Training scrape
# ─────────────────────────────────────────────────────────────────────────────

def scrape_training_data(players=PLAYERS, seasons=TRAINING_SEASONS) -> pd.DataFrame:
    """
    Scrape historical seasons. HTML is cached — re-runs are fast.

    IMPORTANT: The cached HTML files for players that returned "No table found"
    are likely stale error pages or redirects. Before re-running, delete those
    specific cache files so they get re-fetched:

        python scraping/bbref_scraper.py --mode clear-failed
    """
    log.info(f"=== Training scrape: {len(players)} players x {seasons} ===")
    _warm_up_session(SESSION)

    all_dfs = []
    for player in players:
        for season in seasons:
            df = scrape_one(player, season, force_refresh=False)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(TRAIN_OUT, index=False)
    log.info(f"Saved {len(combined):,} rows -> {TRAIN_OUT}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Current season
# ─────────────────────────────────────────────────────────────────────────────

def scrape_current_season(players=PLAYERS) -> pd.DataFrame:
    log.info("=== Current season scrape (2025-26) ===")
    _warm_up_session(SESSION)

    all_dfs = []
    for player in players:
        df = scrape_one(player, CURRENT_SEASON, force_refresh=True)
        if not df.empty:
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(CURRENT_OUT, index=False)
    log.info(f"Saved {len(combined):,} rows -> {CURRENT_OUT}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Clear failed cache files so they get re-fetched
# ─────────────────────────────────────────────────────────────────────────────

def clear_failed_cache(players=PLAYERS, seasons=TRAINING_SEASONS + [CURRENT_SEASON]):
    """
    Delete cached HTML files for player/season combos that produced no data.
    Identifies them by checking whether the corresponding CSV exists and has rows.
    Safe to run before re-running --mode train.
    """
    deleted = 0
    for player in players:
        for season in seasons:
            csv_path  = GAMELOGS_DIR / f"{player['id']}_{season}.csv"
            html_path = HTML_CACHE   / f"{player['id']}_{season}.html"

            # If no CSV or empty CSV, and there's a cached HTML, delete the HTML
            csv_empty = (not csv_path.exists()) or (csv_path.stat().st_size < 100)
            if csv_empty and html_path.exists():
                html_path.unlink()
                log.info(f"  Deleted stale cache: {html_path.name}")
                deleted += 1

    log.info(f"Cleared {deleted} stale cache file(s). Re-run --mode train to re-fetch.")


# ─────────────────────────────────────────────────────────────────────────────
# Quality check
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(df: pd.DataFrame, label: str = "Dataset") -> None:
    if df.empty:
        print(f"{label}: empty.")
        return
    if "reb" not in df.columns and "trb" in df.columns:
        df = df.copy()
        df["reb"] = df["trb"]
    print(f"\n{'─'*50}\n {label}\n{'─'*50}")
    print(f"  Rows    : {len(df):,}")
    print(f"  Players : {df['player_name'].nunique()}")
    print(f"  Seasons : {sorted(df['season'].unique())}")
    print(f"\n  Games per player:")
    counts = df.groupby("player_name")["game_date"].count().sort_values(ascending=False)
    for name, count in counts.items():
        flag = "  <-- low" if count < 50 else ""
        print(f"    {name:30s}: {count}{flag}")
    print(f"\n  Stat averages (sanity check):")
    for col in ["pts", "reb", "ast", "stl", "blk", "minutes"]:
        if col in df.columns:
            print(f"    {col:10s}: {df[col].mean():.1f}")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "current", "all", "clear-failed"],
        default="train",
        help=(
            "train        = historical seasons 2022-2025\n"
            "current      = 2025-26 season, always refreshed\n"
            "all          = both\n"
            "clear-failed = delete stale cache files then exit (run before re-trying)"
        ),
    )
    args = parser.parse_args()

    if args.mode == "clear-failed":
        clear_failed_cache()

    elif args.mode in ("train", "all"):
        quality_report(scrape_training_data(), "Training data (2022-2025)")
        if args.mode == "all":
            quality_report(scrape_current_season(), "Current season (2025-26)")

    elif args.mode == "current":
        quality_report(scrape_current_season(), "Current season (2025-26)")