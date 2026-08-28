"""
In-Season Update Pipeline
=========================
Keeps the model learning as the season progresses. Re-scrapes the current
season, folds any newly-played games into the training corpus, and — only if
new games actually arrived — rebuilds features and retrains.

Steps (each is best-effort and logged; a failure in a network step skips
retraining rather than corrupting the dataset):

    1. Refresh the roster            (scraping/roster.py)
    2. Scrape current-season logs    (bbref_scraper.scrape_current_season)
    3. Merge new games -> all_gamelogs.csv   (dedup by player_id + game_date)
    4. Refresh team pace/defense/schedule    (nba_api_client --mode all)
    5. Rebuild dataset + features    (build_dataset.py, engineer.py)
    6. Retrain + backtest            (train.py, evaluate.py)

Idempotent: run it as often as you like. If no new games are found, it stops
before the expensive rebuild/retrain.

Usage:
    python pipeline/update.py                 # full run
    python pipeline/update.py --skip-scrape   # just merge/rebuild/retrain from cached data
    python pipeline/update.py --dry-run       # scrape + merge, but don't retrain
    python pipeline/update.py --no-team-refresh

Scheduling (pick one; the script itself is schedule-agnostic):
    • macOS cron:      0 6 * * *  cd /path/to/repo && /usr/bin/python3 pipeline/update.py
    • launchd:         a StartCalendarInterval plist calling the same command
    • GitHub Actions:  a `schedule:` workflow that runs this and commits models/
    • Claude Code:     the /schedule skill to run it as a daily cloud task
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scraping"))
sys.path.insert(0, str(ROOT))

ALL_GAMELOGS = ROOT / "data" / "raw" / "all_gamelogs.csv"
ROSTER_JSON  = ROOT / "data" / "processed" / "roster.json"
KEYS         = ["player_id", "game_date"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _run(script: str, *args) -> bool:
    """Run a repo script in a subprocess; return True on success."""
    cmd = [sys.executable, str(ROOT / script), *args]
    log.info("→ %s", " ".join(cmd[1:]))
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        log.warning("  step failed (%s) — continuing", exc)
        return False


def refresh_roster() -> list[dict]:
    """Rebuild roster.json (best effort); fall back to the existing file."""
    _run("scraping/roster.py")
    if ROSTER_JSON.exists():
        roster = json.loads(ROSTER_JSON.read_text())
        log.info("Roster: %d players", len(roster))
        return roster
    log.warning("No roster.json — falling back to scraper's built-in PLAYERS list")
    return []


def scrape_current(roster: list[dict]) -> pd.DataFrame:
    """Scrape current-season game logs for the roster players."""
    from bbref_scraper import scrape_current_season, PLAYERS
    players = roster or PLAYERS
    return scrape_current_season(players=players)


def merge_new_games(fresh: pd.DataFrame) -> int:
    """Fold freshly-scraped games into all_gamelogs.csv, deduped by
    (player_id, game_date). Returns the number of NEW rows added."""
    if fresh is None or fresh.empty:
        log.info("No freshly-scraped rows to merge.")
        return 0

    base = pd.read_csv(ALL_GAMELOGS)
    before = len(base.drop_duplicates(subset=KEYS))

    combined = (
        pd.concat([base, fresh], ignore_index=True)
        .drop_duplicates(subset=KEYS, keep="last")   # prefer the fresh version
    )
    added = len(combined) - before
    if added > 0:
        combined.to_csv(ALL_GAMELOGS, index=False)
        log.info("Merged: +%d new games (now %d total)", added, len(combined))
    else:
        log.info("No new games since last run.")
    return added


def rebuild_and_retrain() -> None:
    """Rebuild dataset + features, retrain, and backtest."""
    log.info("Rebuilding dataset + features and retraining...")
    _run("features/build_dataset.py")
    _run("features/engineer.py")
    _run("models/train.py")
    _run("models/evaluate.py", "--stack-minutes")
    log.info("Retrain complete. See models/saved/eval_report.json for the latest metrics.")


def main():
    ap = argparse.ArgumentParser(description="In-season re-scrape + retrain pipeline.")
    ap.add_argument("--skip-scrape", action="store_true",
                    help="Skip roster/game-log scraping; just rebuild + retrain from cached data.")
    ap.add_argument("--no-team-refresh", action="store_true",
                    help="Skip the nba_api team/schedule refresh.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scrape + merge but do not rebuild/retrain.")
    ap.add_argument("--force", action="store_true",
                    help="Retrain even if no new games were found.")
    args = ap.parse_args()

    log.info("=" * 60)
    log.info(" NBA predictor — in-season update")
    log.info("=" * 60)

    added = 0
    if not args.skip_scrape:
        roster = refresh_roster()
        try:
            fresh = scrape_current(roster)
            added = merge_new_games(fresh)
        except Exception as exc:
            log.warning("Scrape/merge failed (%s) — skipping retrain to avoid corruption", exc)
            return

        if not args.no_team_refresh:
            _run("scraping/nba_api_client.py", "--mode", "all")

    if args.dry_run:
        log.info("Dry run — skipping rebuild/retrain. New games this run: %d", added)
        return

    if added == 0 and not args.skip_scrape and not args.force:
        log.info("Nothing new to learn from — done. (use --force to retrain anyway)")
        return

    rebuild_and_retrain()
    log.info("Update pipeline finished.")


if __name__ == "__main__":
    main()
