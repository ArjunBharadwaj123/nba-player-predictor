"""
Feature Engineering
====================
Builds the full feature matrix from the merged training dataset.

Input:  data/processed/training_dataset.csv
Output: data/processed/features.csv
        data/processed/feature_names.txt

Feature groups:
    1. Rolling averages (last 3, 5, 10 games) for all target stats
    2. Season averages (cumulative, no leakage)
    3. True usage rate (FGA + 0.44*FTA + TOV) / team possessions proxy
    4. Trend features (last3 / last10 ratio)
    5. Shooting efficiency (TS%, rolling TS%, pts per minute)
    6. Game context (rest, pace, opp defense, home/away)
    7. Opponent history (rolling avg vs this specific opponent)
    8. Schedule fatigue (days into season, games in last 7 days, home court factor)
    9. Position encoding

CRITICAL: All rolling features use shift(1) — no data leakage.

Usage:
    python features/engineer.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT      = Path(__file__).parent.parent
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WINDOWS    = [3, 5, 10]
ROLL_STATS = ["pts", "reb", "ast", "stl", "blk", "minutes",
              "usage_rate", "fg_pct", "fg3_pct", "ft_pct", "tov", "fantasy_score"]
TARGETS    = ["pts", "reb", "ast", "stl", "blk", "minutes", "fantasy_score"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rolling_mean(df, col, window, group="player_id"):
    return (
        df.groupby(group)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )


def rolling_std(df, col, window, group="player_id"):
    return (
        df.groupby(group)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=2).std())
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 1: True usage rate
# ─────────────────────────────────────────────────────────────────────────────

def add_usage_rate(df):
    """
    Compute per-game usage rate: what fraction of team possessions does the
    player use while on the court?

    True usage rate formula:
        USG% = (FGA + 0.44*FTA + TOV) / (minutes/team_minutes * team_possessions)

    We don't have team_minutes per game so we use a pace-based proxy:
        team_possessions_per_game ≈ team_pace  (pace = possessions per 48 min)
        player_possession_share  = FGA + 0.44*FTA + TOV
        usage_rate               = player_share / (minutes/48 * team_pace)

    This gives a 0–1 float (e.g. 0.28 = 28% usage rate).
    Stars like Luka/SGA sit around 0.33-0.38; role players around 0.12-0.18.
    """
    log.info("  Computing usage rate...")

    has_cols = all(c in df.columns for c in ["fga", "fta", "tov", "minutes"])
    has_pace = "team_pace" in df.columns

    if has_cols:
        player_share = df["fga"] + 0.44 * df["fta"].fillna(0) + df["tov"].fillna(0)

        if has_pace:
            # Possessions player used / total team possessions while on court
            team_poss_while_on = (df["minutes"].clip(lower=1) / 48) * df["team_pace"]
            df["usage_rate"] = (player_share / team_poss_while_on.clip(lower=0.1)).clip(0, 1)
        else:
            # Fallback: normalise by minutes
            df["usage_rate"] = (player_share / df["minutes"].clip(lower=1)).clip(0, 1)
    else:
        # Fallback proxy if raw stats missing
        df["usage_rate"] = df.get("usage_proxy", pd.Series(0.2, index=df.index))

    log.info(f"  Usage rate — mean: {df['usage_rate'].mean():.3f}, "
             f"max: {df['usage_rate'].max():.3f}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 2: Rolling averages
# ─────────────────────────────────────────────────────────────────────────────

def add_rolling_features(df):
    log.info("  Building rolling average features...")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    for stat in ROLL_STATS:
        if stat not in df.columns:
            continue
        for w in WINDOWS:
            df[f"rolling_last{w}_{stat}"] = rolling_mean(df, stat, w)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 3: Season averages
# ─────────────────────────────────────────────────────────────────────────────

def add_season_averages(df):
    log.info("  Building season average features...")

    for stat in ["pts", "reb", "ast", "stl", "blk", "minutes", "usage_rate"]:
        if stat not in df.columns:
            continue
        df[f"season_avg_{stat}"] = (
            df.groupby(["player_id", "season"])[stat]
            .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 4: Trend features
# ─────────────────────────────────────────────────────────────────────────────

def add_trend_features(df):
    log.info("  Building trend features...")

    for stat in ["pts", "reb", "ast", "fantasy_score", "usage_rate"]:
        l3 = f"rolling_last3_{stat}"
        l10 = f"rolling_last10_{stat}"
        if l3 in df.columns and l10 in df.columns:
            df[f"trend_{stat}"] = (
                df[l3] / df[l10].replace(0, np.nan)
            ).clip(0.3, 3.0)

    df["games_played_season"] = df.groupby(["player_id", "season"]).cumcount()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 5: Efficiency features
# ─────────────────────────────────────────────────────────────────────────────

def add_efficiency_features(df):
    log.info("  Building efficiency features...")

    if all(c in df.columns for c in ["pts", "fga", "fta"]):
        denom = (2 * (df["fga"] + 0.44 * df["fta"])).replace(0, np.nan)
        df["ts_pct"] = (df["pts"] / denom).clip(0, 1)
        for w in [3, 5]:
            df[f"rolling_last{w}_ts_pct"] = rolling_mean(df, "ts_pct", w)

    if "pts" in df.columns and "minutes" in df.columns:
        df["pts_per_min"] = df["pts"] / df["minutes"].clip(lower=1)
        df["rolling_last5_pts_per_min"] = rolling_mean(df, "pts_per_min", 5)

    # Rolling std dev of minutes — captures load management volatility
    if "minutes" in df.columns:
        df["rolling_last5_minutes_std"] = rolling_std(df, "minutes", 5)
        df["rolling_last5_minutes_std"] = df["rolling_last5_minutes_std"].fillna(3.5)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 6: Game context
# ─────────────────────────────────────────────────────────────────────────────

def add_context_features(df):
    log.info("  Building context features...")

    for col in ["rest_days", "back_to_back", "home_game", "team_pace", "opp_def_rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "rest_days" in df.columns:
        df["rest_days_capped"] = df["rest_days"].clip(0, 7)

    # Opponent defensive rank — rank unique teams per season (not per row)
    if all(c in df.columns for c in ["opp_def_rating", "season", "opponent_abbrev"]):
        team_ratings = (
            df.groupby(["opponent_abbrev", "season"])["opp_def_rating"]
            .first()
            .reset_index()
        )
        team_ratings["opp_def_rank"] = (
            team_ratings.groupby("season")["opp_def_rating"]
            .rank(method="average", ascending=True)
        )
        rank_lookup = team_ratings.set_index(["opponent_abbrev", "season"])["opp_def_rank"]
        df["opp_def_rank"] = (
            df.set_index(["opponent_abbrev", "season"]).index.map(rank_lookup).values
        )

    # Position encoding: PG=1, SG=2, SF=3, PF=4, C=5
    POSITION_MAP = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
    if "position" in df.columns:
        df["position_enc"] = (
            df["position"].str.upper().str.strip()
            .map(POSITION_MAP)
            .fillna(3)
            .astype(int)
        )
    else:
        df["position_enc"] = 3

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 7: Opponent history
# ─────────────────────────────────────────────────────────────────────────────

def add_opponent_history_features(df):
    log.info("  Building opponent-history features...")
    df = df.sort_values(["player_id", "opponent_abbrev", "game_date"])

    for stat in ["pts", "reb", "ast"]:
        if stat not in df.columns:
            continue
        df[f"vs_opp_rolling3_{stat}"] = (
            df.groupby(["player_id", "opponent_abbrev"])[stat]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )

    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 8: Schedule fatigue and season timing
# ─────────────────────────────────────────────────────────────────────────────

def add_schedule_features(df):
    log.info("  Building schedule fatigue features...")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Days into season
    if "game_date" in df.columns and "season" in df.columns:
        season_start = df.groupby("season")["game_date"].transform("min")
        df["days_into_season"] = (df["game_date"] - season_start).dt.days
    else:
        df["days_into_season"] = 0

    # Games in last 7 days (cumulative fatigue beyond B2B)
    def _games_last_7(group):
        dates = group["game_date"].values
        counts = []
        for i, date in enumerate(dates):
            cutoff = date - np.timedelta64(7, "D")
            counts.append(int(np.sum(dates[:i] >= cutoff)))
        return counts

    results = []
    for pid, group in df.groupby("player_id"):
        group = group.sort_values("game_date")
        results.extend(zip(group.index, _games_last_7(group)))

    if results:
        idx, vals = zip(*results)
        df["games_in_last_7_days"] = pd.Series(dict(zip(idx, vals))).reindex(df.index).fillna(0).astype(int)
    else:
        df["games_in_last_7_days"] = 0

    # Home court factor per team (actual home win rate from training data)
    if all(c in df.columns for c in ["team_abbrev", "result", "home_game"]):
        home_games = df[df["home_game"] == 1].copy()
        home_games["won"] = home_games["result"].astype(str).str.startswith("W").astype(int)
        home_wr = home_games.groupby("team_abbrev")["won"].mean().rename("home_court_factor")
        df["home_court_factor"] = df["team_abbrev"].map(home_wr).fillna(0.575).round(3)
    else:
        df["home_court_factor"] = 0.575

    return df



# ─────────────────────────────────────────────────────────────────────────────
# Feature group 9: Opponent points allowed by position
# ─────────────────────────────────────────────────────────────────────────────

def add_positional_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add how many points each opponent allows per game to each position.

    Why this matters for pts accuracy:
        opp_def_rating treats all opponents the same regardless of position.
        But the Celtics might be 2nd best at guarding PGs while being 18th
        at guarding centers. A PG vs Boston is very different from a C vs Boston.

    Source: data/processed/opp_pos_defense.csv
        Built by nba_api_client.py --mode pos-defense
        Columns: season, opponent_abbrev, position, pts_allowed_per_game, rank

    If the file doesn't exist yet, this function fills with neutral values
    so engineer.py still runs — you can re-run after fetching the data.
    """
    log.info("  Building positional defense features...")

    POS_FILE = PROCESSED / "opp_pos_defense.csv"

    if not POS_FILE.exists():
        log.warning("  opp_pos_defense.csv not found — fill with neutral 0.0")
        log.warning("  Run: python scraping/nba_api_client.py --mode pos-defense")
        df["opp_pts_allowed_pos"]      = 0.0
        df["opp_pos_defense_rank"]     = 15.0
        return df

    pos_df = pd.read_csv(POS_FILE)

    # Map our position strings to the NBA API position labels
    POS_MAP = {"PG": "Point Guard", "SG": "Shooting Guard",
               "SF": "Small Forward", "PF": "Power Forward", "C": "Center"}

    # Build lookup: (opponent_abbrev, season, nba_position) -> pts_allowed
    pos_df["join_pos"] = pos_df["position"]
    lookup = pos_df.set_index(["opponent_abbrev", "season", "position"])

    def _lookup_pts(row):
        pos_str = str(row.get("position", "SF")).upper().strip()
        nba_pos = POS_MAP.get(pos_str, "Small Forward")
        key = (str(row.get("opponent_abbrev", "")),
               int(row.get("season", 2024)),
               nba_pos)
        try:
            return float(lookup.loc[key, "pts_allowed_per_game"])
        except (KeyError, TypeError):
            return float("nan")

    def _lookup_rank(row):
        pos_str = str(row.get("position", "SF")).upper().strip()
        nba_pos = POS_MAP.get(pos_str, "Small Forward")
        key = (str(row.get("opponent_abbrev", "")),
               int(row.get("season", 2024)),
               nba_pos)
        try:
            return float(lookup.loc[key, "rank"])
        except (KeyError, TypeError):
            return 15.0

    df["opp_pts_allowed_pos"]  = df.apply(_lookup_pts, axis=1)
    df["opp_pos_defense_rank"] = df.apply(_lookup_rank, axis=1)

    # Fill NaN with season-position median
    df["opp_pts_allowed_pos"]  = df.groupby(["season", "position"])["opp_pts_allowed_pos"].transform(
        lambda x: x.fillna(x.median())
    )
    df["opp_pos_defense_rank"] = df["opp_pos_defense_rank"].fillna(15.0)

    valid = df["opp_pts_allowed_pos"].notna().sum()
    log.info(f"  Positional defense: {valid:,} rows filled "
             f"(mean allowed: {df['opp_pts_allowed_pos'].mean():.1f} pts)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 10: Starter flag
# ─────────────────────────────────────────────────────────────────────────────

def add_starter_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add starter-related features derived from the game log's 'gs' column
    (games started — already scraped from BBRef).

    Features:
        is_likely_starter    : 1 if started in 80%+ of last 10 games
        start_rate_last10    : rolling fraction of games started (last 10)

    Why this helps minutes:
        Starters average 32-36 min; bench players 18-24 min.
        A player who has been spot-starting (mixed starter/bench role)
        has highly variable minutes — the model should know this.
    """
    log.info("  Building starter features...")

    if "gs" not in df.columns:
        log.warning("  'gs' column not found — skipping starter features")
        df["is_likely_starter"] = 1
        df["start_rate_last10"] = 0.8
        return df

    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    df["gs_binary"] = pd.to_numeric(df["gs"], errors="coerce").fillna(0).clip(0, 1)

    # Rolling start rate over last 10 games (shift to avoid leakage)
    df["start_rate_last10"] = (
        df.groupby("player_id")["gs_binary"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
        .fillna(0.8)  # assume starter if unknown
    )
    df["is_likely_starter"] = (df["start_rate_last10"] >= 0.6).astype(int)

    starters = df["is_likely_starter"].mean()
    log.info(f"  {starters*100:.1f}% of rows classified as likely starters")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 11: Season phase
# ─────────────────────────────────────────────────────────────────────────────

def add_season_phase_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode which phase of the season a game falls in.

    NBA seasons have four distinct phases with different performance patterns:
        1 = Early season   (games 1-20):   players ramping up, rotations fluid
        2 = Mid season     (games 21-55):  established roles, peak form
        3 = Late season    (games 56-72):  load management ramps up for stars
        4 = End stretch    (games 73-82):  playoff push or tanking, extreme variance

    Encoded as a 1-4 integer so XGBoost can learn non-linear phase effects.
    Also adds:
        is_late_season : binary flag for games 56+ (load management risk)
    """
    log.info("  Building season phase features...")

    if "games_played_season" not in df.columns:
        df["season_phase"]   = 2
        df["is_late_season"] = 0
        return df

    conditions = [
        df["games_played_season"] <= 20,
        df["games_played_season"] <= 55,
        df["games_played_season"] <= 72,
    ]
    choices = [1, 2, 3]
    df["season_phase"] = np.select(conditions, choices, default=4)
    df["is_late_season"] = (df["games_played_season"] >= 56).astype(int)

    phase_counts = df["season_phase"].value_counts().sort_index()
    log.info(f"  Season phase distribution: {phase_counts.to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Final feature selection
# ─────────────────────────────────────────────────────────────────────────────

def select_features(df):
    rolling_cols = [
        f"rolling_last{w}_{stat}"
        for w in WINDOWS for stat in ROLL_STATS
        if f"rolling_last{w}_{stat}" in df.columns
    ]
    season_avg_cols  = [c for c in df.columns if c.startswith("season_avg_")]
    trend_cols       = [c for c in df.columns if c.startswith("trend_")]
    efficiency_cols  = [c for c in df.columns if c in [
        "rolling_last3_ts_pct", "rolling_last5_ts_pct",
        "rolling_last5_pts_per_min", "rolling_last5_minutes_std",
    ]]
    context_cols     = [c for c in df.columns if c in [
        "rest_days_capped", "back_to_back", "home_game",
        "team_pace", "opp_def_rating", "opp_def_rank",
        "games_played_season", "position_enc",
        "days_into_season", "games_in_last_7_days", "home_court_factor",
        "opp_pts_allowed_pos", "opp_pos_defense_rank",
        "is_likely_starter", "start_rate_last10",
        "season_phase", "is_late_season",
    ]]
    opp_history_cols = [c for c in df.columns if c.startswith("vs_opp_rolling")]

    feature_names = (rolling_cols + season_avg_cols + trend_cols
                     + efficiency_cols + context_cols + opp_history_cols)

    # Deduplicate preserving order
    seen = set()
    feature_names = [f for f in feature_names if not (f in seen or seen.add(f))]

    log.info(f"  Total features: {len(feature_names)}")
    log.info(f"    Rolling avgs    : {len(rolling_cols)}")
    log.info(f"    Season avgs     : {len(season_avg_cols)}")
    log.info(f"    Trend           : {len(trend_cols)}")
    log.info(f"    Efficiency      : {len(efficiency_cols)}")
    log.info(f"    Context         : {len(context_cols)}")
    log.info(f"    Opponent history: {len(opp_history_cols)}")

    return df, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def quality_report(df, feature_names):
    print(f"\n{'─'*55}\n Feature engineering quality report\n{'─'*55}")
    print(f"  Rows          : {len(df):,}")
    print(f"  Feature count : {len(feature_names)}")
    print(f"\n  Target averages:")
    for col in TARGETS:
        if col in df.columns:
            print(f"    {col:15s}: {df[col].mean():.2f}")
    print(f"\n  Usage rate averages by position:")
    if "position" in df.columns and "usage_rate" in df.columns:
        ur = df.groupby("position")["usage_rate"].mean().sort_values(ascending=False)
        for pos, rate in ur.items():
            print(f"    {pos}: {rate:.3f}")
    print(f"\n  Top 10 null rates:")
    null_rates = df[feature_names].isna().mean().sort_values(ascending=False)
    for feat, rate in null_rates.head(10).items():
        print(f"    {feat:42s}: {rate*100:.1f}%")
    print(f"{'─'*55}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_features():
    log.info("Loading training dataset...")
    df = pd.read_csv(PROCESSED / "training_dataset.csv", parse_dates=["game_date"])
    log.info(f"  {len(df):,} rows loaded")

    log.info("Building feature groups...")
    df = add_usage_rate(df)
    df = add_rolling_features(df)
    df = add_season_averages(df)
    df = add_trend_features(df)
    df = add_efficiency_features(df)
    df = add_context_features(df)
    df = add_opponent_history_features(df)
    df = add_schedule_features(df)
    df = add_positional_defense_features(df)
    df = add_starter_features(df)
    df = add_season_phase_features(df)

    df, feature_names = select_features(df)

    df.to_csv(PROCESSED / "features.csv", index=False)
    (PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))

    log.info(f"Saved features.csv and feature_names.txt")
    return df, feature_names


if __name__ == "__main__":
    df, feature_names = build_features()
    quality_report(df, feature_names)