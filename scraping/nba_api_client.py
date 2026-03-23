"""
Step 5: NBA API Client
======================
Fetches team-level contextual data that the BBRef scraper cannot provide:

    - Team pace (possessions per 48 minutes)
    - Team offensive and defensive rating
    - Opponent defensive rating (how well each team defends)
    - Rest days and back-to-back flags for every game

These become the contextual features that separate a good model from a
naive rolling-average model. A player on a fast team vs a slow-defense
opponent on full rest is a completely different prediction than the same
player on a back-to-back vs the Celtics.

Usage:
    python scraping/nba_api_client.py --mode all     # fetch everything
    python scraping/nba_api_client.py --mode teams   # team stats only
    python scraping/nba_api_client.py --mode schedule # rest days only

Output files:
    data/raw/team_stats_{season}.csv      <- pace, ratings per team per season
    data/raw/schedule_{season}.csv        <- every game with rest-day flags
    data/raw/all_team_stats.csv           <- all seasons combined
    data/raw/all_schedules.csv            <- all seasons combined
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

# nba_api endpoints we use
from nba_api.stats.endpoints import LeagueDashTeamStats, LeagueGameLog
from nba_api.stats.static import teams as nba_teams_static

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
RAW       = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
# nba_api season strings use "YYYY-YY" format
# Our internal season ints use the ending year (2022 = 2021-22)
TRAINING_SEASONS = [2022, 2023, 2024, 2025]
CURRENT_SEASON   = 2026
API_DELAY        = 1.0    # seconds between API calls — NBA.com rate limits ~3 req/s

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: convert our season int to NBA API string format
# ─────────────────────────────────────────────────────────────────────────────

def season_to_str(season: int) -> str:
    """
    Convert season ending year to NBA API format.

    Examples:
        2022 -> "2021-22"
        2026 -> "2025-26"
    """
    start = season - 1
    end   = str(season)[2:]   # last two digits
    return f"{start}-{end}"


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: Team stats (pace, offensive rating, defensive rating)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_stats(season: int) -> pd.DataFrame:
    """
    Fetch season-level team stats for all 30 NBA teams.

    Uses LeagueDashTeamStats with measure_type="Advanced" to get:
        - PACE           : possessions per 48 minutes
        - OFF_RATING     : points scored per 100 possessions
        - DEF_RATING     : points allowed per 100 possessions (lower = better defense)
        - NET_RATING     : OFF_RATING - DEF_RATING
        - PACE_PER40     : pace normalised to 40 minutes (some analyses prefer this)

    Why advanced stats and not basic box scores?
        Basic stats (FG, REB, etc.) don't account for pace. A team with 110
        points per game on a fast team may actually be worse offensively than
        a team with 105 on a slow team. Ratings normalize for pace.

    Args:
        season: Season ending year, e.g. 2024 for 2023-24

    Returns:
        DataFrame with one row per team, columns including:
        team_id, team_name, team_abbrev, season,
        pace, off_rating, def_rating, net_rating
    """
    season_str = season_to_str(season)
    log.info(f"Fetching team stats — {season_str}")

    try:
        stats = LeagueDashTeamStats(
            season=season_str,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=30,
        )
        df = stats.get_data_frames()[0]
    except Exception as exc:
        log.error(f"  Failed to fetch team stats for {season_str}: {exc}")
        return pd.DataFrame()

    time.sleep(API_DELAY)

    # Rename to clean snake_case
    rename = {
        "TEAM_ID"    : "team_id",
        "TEAM_NAME"  : "team_name",
        "PACE"       : "team_pace",
        "OFF_RATING" : "team_off_rating",
        "DEF_RATING" : "team_def_rating",
        "NET_RATING" : "team_net_rating",
    }
    df = df.rename(columns=rename)

    # Add the team abbreviation (ATL, BOS, etc.) — useful for joining later
    abbrev_map = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
    df["team_abbrev"] = df["team_id"].map(abbrev_map)

    df["season"] = season

    # Keep only the columns we need
    keep = ["team_id", "team_name", "team_abbrev", "season",
            "team_pace", "team_off_rating", "team_def_rating", "team_net_rating"]
    df = df[[c for c in keep if c in df.columns]]

    log.info(f"  Got {len(df)} teams — pace range: "
             f"{df['team_pace'].min():.1f} to {df['team_pace'].max():.1f}")
    return df


def fetch_all_team_stats(seasons: list = TRAINING_SEASONS + [CURRENT_SEASON]) -> pd.DataFrame:
    """Fetch and combine team stats for all seasons."""
    log.info(f"=== Fetching team stats for seasons {seasons} ===")
    all_dfs = []

    for season in seasons:
        df = fetch_team_stats(season)
        if not df.empty:
            out = RAW / f"team_stats_{season}.csv"
            df.to_csv(out, index=False)
            log.info(f"  Saved -> {out.name}")
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(RAW / "all_team_stats.csv", index=False)
    log.info(f"Combined team stats: {len(combined)} rows -> all_team_stats.csv")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: Schedule with rest days and back-to-back flags
# ─────────────────────────────────────────────────────────────────────────────

def fetch_schedule(season: int) -> pd.DataFrame:
    """
    Fetch the full game schedule for a season and compute rest-day features.

    Uses LeagueGameLog which returns one row per team per game — so every
    game appears twice (once for the home team, once for the away team).
    This is exactly what we want: each row tells us how many days rest
    THAT TEAM had before THAT GAME.

    Computed columns:
        rest_days    : days since the team's last game (NaN for season opener)
        back_to_back : 1 if rest_days == 0, else 0
        home_game    : 1 if the team was at home, 0 if away

    Why rest days matter for prediction:
        - Players average ~1.5 fewer minutes on back-to-backs
        - Shooting efficiency drops ~2% on 0 rest
        - Stars are more likely to be rested entirely (load management)

    Args:
        season: Season ending year, e.g. 2024

    Returns:
        DataFrame with one row per team per game, sorted by team and date.
    """
    season_str = season_to_str(season)
    log.info(f"Fetching schedule — {season_str}")

    try:
        gl = LeagueGameLog(
            season=season_str,
            direction="ASC",        # chronological order
            timeout=60,             # schedule endpoint can be slow
        )
        df = gl.get_data_frames()[0]
    except Exception as exc:
        log.error(f"  Failed to fetch schedule for {season_str}: {exc}")
        return pd.DataFrame()

    time.sleep(API_DELAY)

    # ── Rename ────────────────────────────────────────────────────────────────
    rename = {
        "TEAM_ID"           : "team_id",
        "TEAM_ABBREVIATION" : "team_abbrev",
        "TEAM_NAME"         : "team_name",
        "GAME_ID"           : "game_id",
        "GAME_DATE"         : "game_date",
        "MATCHUP"           : "matchup",
        "WL"                : "result",
    }
    df = df.rename(columns=rename)

    # ── Parse date ────────────────────────────────────────────────────────────
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Derive home_game from the matchup string ──────────────────────────────
    # Matchup format: "LAL vs. DEN" (home) or "LAL @ DEN" (away)
    df["home_game"] = df["matchup"].str.contains(r"\bvs\.", regex=True).astype(int)

    # ── Extract opponent abbreviation ─────────────────────────────────────────
    # "LAL vs. DEN" -> opponent is "DEN"
    # "LAL @ DEN"   -> opponent is "DEN"
    df["opponent_abbrev"] = (
        df["matchup"]
        .str.replace(r".*(vs\.|@)\s*", "", regex=True)
        .str.strip()
    )

    # ── Compute rest days per team ────────────────────────────────────────────
    # Sort by team and date, then compute the difference between consecutive games
    df = df.sort_values(["team_id", "game_date"]).reset_index(drop=True)

    df["prev_game_date"] = df.groupby("team_id")["game_date"].shift(1)
    df["rest_days"] = (
        (df["game_date"] - df["prev_game_date"]).dt.days - 1
    ).fillna(-1).astype(int)
    # rest_days = -1 means first game of the season (no previous game)
    # rest_days = 0  means back-to-back
    # rest_days = 1  means one day rest (played two days ago)

    df["back_to_back"] = (df["rest_days"] == 0).astype(int)

    df["season"] = season
    df = df.drop(columns=["prev_game_date"], errors="ignore")

    # Keep only useful columns
    keep = ["game_id", "game_date", "season", "team_id", "team_abbrev",
            "team_name", "opponent_abbrev", "home_game", "result",
            "rest_days", "back_to_back"]
    df = df[[c for c in keep if c in df.columns]]

    b2b_count = df["back_to_back"].sum()
    log.info(f"  {len(df)} team-game rows, {b2b_count} back-to-backs")
    return df


def fetch_all_schedules(seasons: list = TRAINING_SEASONS + [CURRENT_SEASON]) -> pd.DataFrame:
    """Fetch and combine schedules for all seasons."""
    log.info(f"=== Fetching schedules for seasons {seasons} ===")
    all_dfs = []

    for season in seasons:
        df = fetch_schedule(season)
        if not df.empty:
            out = RAW / f"schedule_{season}.csv"
            df.to_csv(out, index=False)
            log.info(f"  Saved -> {out.name}")
            all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(RAW / "all_schedules.csv", index=False)
    log.info(f"Combined schedules: {len(combined)} rows -> all_schedules.csv")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Opponent defensive rating lookup
# ─────────────────────────────────────────────────────────────────────────────

def build_opponent_def_ratings(
    team_stats: pd.DataFrame,
    schedules: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join opponent defensive ratings onto the schedule.

    After this function, each game row has both:
        team_pace        : the playing team's pace
        opp_def_rating   : the opponent's defensive rating
        opp_pace         : the opponent's pace (affects total possessions)

    This is the table we'll join onto the player game logs in Step 8
    (dataset construction). Every player game gets these two context columns:
    how good was the opponent's defense, and how fast was the game?

    Args:
        team_stats  : output of fetch_all_team_stats()
        schedules   : output of fetch_all_schedules()

    Returns:
        Schedule DataFrame enriched with opponent defensive context.
    """
    log.info("Building opponent defensive rating lookup...")

    # Build a lookup: (team_abbrev, season) -> defensive stats
    def_lookup = team_stats.set_index(["team_abbrev", "season"])[
        ["team_pace", "team_def_rating", "team_off_rating"]
    ].rename(columns={
        "team_pace"       : "opp_pace",
        "team_def_rating" : "opp_def_rating",
        "team_off_rating" : "opp_off_rating",
    })

    # Join opponent stats onto the schedule using opponent_abbrev + season
    enriched = schedules.join(
        def_lookup,
        on=["opponent_abbrev", "season"],
        how="left",
    )

    # Also join the playing team's own pace
    team_pace_lookup = team_stats.set_index(["team_abbrev", "season"])[["team_pace"]]
    enriched = enriched.join(
        team_pace_lookup,
        on=["team_abbrev", "season"],
        how="left",
    )

    missing = enriched["opp_def_rating"].isna().sum()
    if missing > 0:
        log.warning(
            f"  {missing} rows missing opp_def_rating "
            f"(likely expansion/relocation team name mismatches)"
        )

    log.info(f"  Enriched schedule: {len(enriched)} rows")
    log.info(f"  Opp def rating range: "
             f"{enriched['opp_def_rating'].min():.1f} – "
             f"{enriched['opp_def_rating'].max():.1f}")

    out = RAW / "schedule_with_context.csv"
    enriched.to_csv(out, index=False)
    log.info(f"  Saved -> {out.name}")
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Quality check
# ─────────────────────────────────────────────────────────────────────────────

def quality_report_teams(team_stats: pd.DataFrame) -> None:
    if team_stats.empty:
        print("Team stats: empty.")
        return
    print(f"\n{'─'*50}\n Team stats quality report\n{'─'*50}")
    print(f"  Rows    : {len(team_stats)}")
    print(f"  Seasons : {sorted(team_stats['season'].unique())}")
    print(f"\n  Pace by season (league average ~100 possessions):")
    pace_by_season = team_stats.groupby("season")["team_pace"].mean()
    for season, pace in pace_by_season.items():
        print(f"    {season}: {pace:.1f}")
    print(f"\n  Best defenses this dataset (lowest def_rating):")
    best_def = (
        team_stats.sort_values("team_def_rating")
        .groupby("season")
        .first()[["team_name", "team_def_rating"]]
    )
    print(best_def.to_string())
    print(f"{'─'*50}\n")


def quality_report_schedule(schedule: pd.DataFrame) -> None:
    if schedule.empty:
        print("Schedule: empty.")
        return
    print(f"\n{'─'*50}\n Schedule quality report\n{'─'*50}")
    print(f"  Rows    : {len(schedule):,}")
    print(f"  Seasons : {sorted(schedule['season'].unique())}")
    print(f"\n  Back-to-backs per season:")
    b2b = schedule.groupby("season")["back_to_back"].sum()
    for season, count in b2b.items():
        print(f"    {season}: {int(count)} team-games on B2B")
    print(f"\n  Rest day distribution:")
    rest_counts = (
        schedule[schedule["rest_days"] >= 0]["rest_days"]
        .value_counts()
        .sort_index()
        .head(6)
    )
    for days, count in rest_counts.items():
        print(f"    {days} rest day(s): {count:,} games")
    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# Positional defense: opponent pts allowed per game by position
# ─────────────────────────────────────────────────────────────────────────────

def fetch_positional_defense(seasons: list = None) -> pd.DataFrame:
    """
    Build a table of how many points each team allows per game to each position.

    Method:
        1. Fetch each team's defensive rating for the season (LeagueDashTeamStats)
        2. Scale league-average pts by position using each team's defensive strength
           relative to the league average defensive rating

    Formula:
        pts_allowed_to_PG = LEAGUE_AVG_PG_PTS * (team_def_rating / league_avg_def)

    Output: data/processed/opp_pos_defense.csv
        season, opponent_abbrev, position, pts_allowed_per_game, rank
    """
    if seasons is None:
        seasons = TRAINING_SEASONS + [CURRENT_SEASON]

    # League-wide average pts per game by position (stable NBA historical averages)
    LEAGUE_AVG_PTS_BY_POS = {
        "Point Guard"    : 14.8,
        "Shooting Guard" : 13.2,
        "Small Forward"  : 12.1,
        "Power Forward"  : 10.9,
        "Center"         :  9.4,
    }

    all_rows = []

    for season in seasons:
        season_str = season_to_str(season)
        log.info(f"Fetching team defense for positional scaling — {season_str}")

        try:
            resp = LeagueDashTeamStats(
                season=season_str,
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="PerGame",
                timeout=30,
            )
            time.sleep(API_DELAY)
            ts_df = resp.get_data_frames()[0]
        except Exception as exc:
            log.error(f"  LeagueDashTeamStats failed for {season_str}: {exc}")
            continue

        if ts_df.empty:
            log.warning(f"  Empty response for {season_str}")
            continue

        ts_df = ts_df.rename(columns={"TEAM_ID": "team_id", "DEF_RATING": "def_rating"})
        abbrev_map = {t["id"]: t["abbreviation"] for t in nba_teams_static.get_teams()}
        ts_df["opponent_abbrev"] = ts_df["team_id"].map(abbrev_map)
        ts_df = ts_df.dropna(subset=["opponent_abbrev", "def_rating"])

        league_avg_def = float(ts_df["def_rating"].mean())
        log.info(f"  League avg def rating: {league_avg_def:.1f}")

        # Build one row per team per position
        for pos_name, league_avg_pts in LEAGUE_AVG_PTS_BY_POS.items():
            pos_rows = ts_df[["opponent_abbrev", "def_rating"]].copy()
            pos_rows["pts_allowed_per_game"] = (
                league_avg_pts * (pos_rows["def_rating"] / league_avg_def)
            ).round(2)
            pos_rows["position"] = pos_name
            pos_rows["season"]   = season
            pos_rows["rank"] = pos_rows["pts_allowed_per_game"].rank(
                method="average", ascending=True
            ).astype(int)

            all_rows.append(
                pos_rows[["season", "opponent_abbrev", "position",
                           "pts_allowed_per_game", "rank"]]
            )

        log.info(
            f"  {season_str}: added {len(LEAGUE_AVG_PTS_BY_POS)} positions "
            f"x {len(ts_df)} teams = "
            f"{len(LEAGUE_AVG_PTS_BY_POS) * len(ts_df)} rows"
        )

    if not all_rows:
        log.error("No positional defense data built")
        return pd.DataFrame()

    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=["season", "opponent_abbrev", "position"])

    PROCESSED = ROOT / "data" / "processed"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / "opp_pos_defense.csv"
    combined.to_csv(out, index=False)

    log.info(f"Saved -> {out} ({len(combined):,} rows)")
    return combined