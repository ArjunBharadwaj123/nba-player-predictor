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
              "usage_rate", "fg_pct", "fg3", "fg3a", "fg3_pct", "ft_pct",
              "tov", "fantasy_score"]
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


def ewm_mean(df, col, halflife=3, group="player_id"):
    """Recency-weighted (exponentially-weighted) mean. shift(1) = no leakage.
    Recent games are weighted more heavily than a flat rolling window, which
    tends to track hot/cold streaks better."""
    return (
        df.groupby(group)[col]
        .transform(lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1).mean())
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
# Feature group 3b: Recency-weighted (EWMA) form
# ─────────────────────────────────────────────────────────────────────────────

EWM_STATS = ["pts", "reb", "ast", "minutes", "usage_rate"]

def add_ewm_features(df):
    """Exponentially-weighted recent form (halflife = 3 games). Complements the
    flat rolling windows by emphasising the most recent games."""
    log.info("  Building EWMA (recency-weighted) features...")
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for stat in EWM_STATS:
        if stat in df.columns:
            df[f"ewm_{stat}"] = ewm_mean(df, stat, halflife=3)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 3c: Home/away form split
# ─────────────────────────────────────────────────────────────────────────────

def add_venue_split_features(df):
    """Rolling last-10 form computed *within* the player's home vs away games.
    Some players are meaningfully better at home; a single blended rolling
    average hides that."""
    log.info("  Building home/away form-split features...")
    if "home_game" not in df.columns:
        return df
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)
    for stat in ["pts", "reb", "ast", "minutes"]:
        if stat in df.columns:
            df[f"venue_last10_{stat}"] = (
                df.groupby(["player_id", "home_game"])[stat]
                .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
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

    # Opponent defensive rank — AS-OF-DATE cross-sectional rank.
    # opp_def_rating is now a per-game as-of-date value, so we rank opponents
    # against each other *on that game's date* (all games sharing a date see a
    # consistent ranking) rather than ranking one season-final aggregate. This
    # keeps the rank leakage-free and in step with the rolling rating.
    if all(c in df.columns for c in ["opp_def_rating", "game_date", "opponent_abbrev"]):
        # Rank distinct opponents on each date (not player-rows, which would make
        # the scale depend on how many players played that night). ~1-30 teams.
        ranks = (
            df[["game_date", "opponent_abbrev", "opp_def_rating"]]
            .drop_duplicates(["game_date", "opponent_abbrev"])
            .copy()
        )
        ranks["opp_def_rank"] = (
            ranks.groupby("game_date")["opp_def_rating"]
            .rank(method="average", ascending=True)
        )
        rank_lookup = ranks.set_index(["game_date", "opponent_abbrev"])["opp_def_rank"]
        df["opp_def_rank"] = (
            df.set_index(["game_date", "opponent_abbrev"]).index.map(rank_lookup).values
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
# Feature group 7a: Teammate availability / usage redistribution
# ─────────────────────────────────────────────────────────────────────────────

def add_teammate_features(df):
    """When a rotation teammate sits, the remaining players absorb their minutes,
    touches and usage — the single biggest real-world swing in box-score output.

    We infer inactives directly from the game logs (no external injury feed): a
    team's rotation for a season is the set of players with a real role (>=10
    games, >=20 mpg); for each team-game the rotation members who have NO row are
    treated as out. These signals are known pre-game (injury reports drop before
    tip-off), so using the current game's availability is not leakage. Player
    'weight' is season average minutes — a stable role descriptor.

    Features (identical for every player in a given team-game):
        teammates_out_count  : # of rotation teammates inactive
        star_teammate_out    : 1 if an inactive teammate averages >= 28 mpg
        team_minutes_vacated : summed season mpg of the inactive teammates
    """
    log.info("  Building teammate-availability features...")
    need = ["player_id", "team_abbrev", "season", "game_date", "minutes"]
    if not all(c in df.columns for c in need):
        df["teammates_out_count"]  = 0
        df["star_teammate_out"]    = 0
        df["team_minutes_vacated"] = 0.0
        return df

    STAR_MPG = 28.0
    played = df[df["minutes"] >= 1]

    # Rotation membership per (team, season, player): role weight + the window of
    # their tenure with that team (first..last appearance). A player only counts
    # as "out" for games INSIDE that window — this excludes players who were
    # traded away, hadn't arrived yet, or were done for the season (their missed
    # games aren't injuries, they just weren't on the roster then).
    role = (
        played.groupby(["team_abbrev", "season", "player_id"])["minutes"]
        .agg(games="size", mpg="mean").reset_index()
    )
    tenure = (
        played.groupby(["team_abbrev", "season", "player_id"])["game_date"]
        .agg(first="min", last="max").reset_index()
    )
    role = role.merge(tenure, on=["team_abbrev", "season", "player_id"])
    rotation = role[(role["games"] >= 10) & (role["mpg"] >= 20)]

    # (team, season) -> list of (player_id, mpg, first, last)
    rot_by_team = {}
    for t, s, p, m, f, l in zip(
        rotation["team_abbrev"], rotation["season"], rotation["player_id"],
        rotation["mpg"], rotation["first"], rotation["last"]
    ):
        rot_by_team.setdefault((t, s), []).append((p, m, f, l))

    # Who actually appeared in each team-game.
    present = (
        played.groupby(["team_abbrev", "season", "game_date"])["player_id"]
        .apply(set).reset_index(name="present")
    )

    def _row(r):
        members = rot_by_team.get((r.team_abbrev, r.season), [])
        cnt, star, vac = 0, 0, 0.0
        for p, m, first, last in members:
            # Out only if the game is within the player's tenure and they missed it.
            if first < r.game_date < last and p not in r.present:
                cnt += 1
                vac += m
                if m >= STAR_MPG:
                    star = 1
        return cnt, star, vac

    vals = present.apply(_row, axis=1, result_type="expand")
    present["teammates_out_count"]  = vals[0].astype(int)
    present["star_teammate_out"]    = vals[1].astype(int)
    present["team_minutes_vacated"] = vals[2].astype(float)

    out = df.merge(
        present[["team_abbrev", "season", "game_date",
                 "teammates_out_count", "star_teammate_out", "team_minutes_vacated"]],
        on=["team_abbrev", "season", "game_date"], how="left",
    )
    for c in ["teammates_out_count", "star_teammate_out", "team_minutes_vacated"]:
        out[c] = out[c].fillna(0)
    log.info("    %.1f%% of games have a rotation player out (mean %.1f out)",
             (out["teammates_out_count"] > 0).mean() * 100,
             out["teammates_out_count"].mean())
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 7c: Vegas odds (nullable — self-activates as history accumulates)
# ─────────────────────────────────────────────────────────────────────────────

VEGAS_COLS = ["implied_team_total", "game_total", "team_spread", "is_favorite"]


def add_vegas_features(df):
    """Join Vegas implied team totals / spreads from data/raw/odds_history.csv.

    The free odds tier has no history, so scraping/odds.py can only accumulate
    lines going forward. Rows with no matching odds stay NaN — harmless to
    XGBoost, and the moment enough history exists a retrain starts using these
    automatically (no code change needed). implied_team_total is typically the
    strongest single external predictor of counting-stat output.
    """
    log.info("  Building Vegas odds features...")
    odds_path = PROCESSED.parent / "raw" / "odds_history.csv"
    if not odds_path.exists():
        for c in VEGAS_COLS:
            df[c] = np.nan
        log.info("    no odds_history.csv yet — Vegas features left NaN")
        return df

    odds = pd.read_csv(odds_path)
    odds["game_date"] = pd.to_datetime(odds["game_date"], errors="coerce")
    keep = ["game_date", "team_abbrev"] + [c for c in VEGAS_COLS if c in odds.columns]
    odds = odds[keep].drop_duplicates(["game_date", "team_abbrev"])

    df = df.merge(odds, on=["game_date", "team_abbrev"], how="left")
    matched = df["implied_team_total"].notna().sum() if "implied_team_total" in df else 0
    log.info("    matched odds for %d / %d rows", matched, len(df))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature group 7b: Matchup / interaction features
# ─────────────────────────────────────────────────────────────────────────────

def add_interaction_features(df):
    """Give the opponent/context columns something the trees can actually use.

    - opp_pace / pace_sum : the opponent's own pace (looked up from that team's
      team_pace) and the combined game pace — a proxy for how many total
      possessions (and therefore counting-stat opportunities) the game offers.
    - usage_x_minutes      : usage rate scales counting stats, but only in
      proportion to minutes played. Their product is the real opportunity
      signal.
    """
    log.info("  Building matchup / interaction features...")

    # opp_pace is now supplied as an AS-OF-DATE column by build_dataset (from
    # nba_api_client.build_asof_team_ratings). Only fall back to a season-average
    # derivation if it's missing, so we never silently reintroduce the season-
    # aggregate leakage this column used to carry.
    if "opp_pace" not in df.columns and all(
        c in df.columns for c in ["team_abbrev", "opponent_abbrev", "season", "team_pace"]
    ):
        log.warning("  opp_pace absent — falling back to season-average pace")
        pace_lookup = df.groupby(["team_abbrev", "season"])["team_pace"].mean()
        df["opp_pace"] = (
            df.set_index(["opponent_abbrev", "season"]).index
            .map(pace_lookup).astype(float)
        )
        df["opp_pace"] = df["opp_pace"].fillna(df["team_pace"])

    if "team_pace" in df.columns and "opp_pace" in df.columns:
        df["pace_sum"] = df["team_pace"] + df["opp_pace"]

    if "season_avg_usage_rate" in df.columns and "rolling_last5_minutes" in df.columns:
        df["usage_x_minutes"] = (
            df["season_avg_usage_rate"] * df["rolling_last5_minutes"]
        )

    return df


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

    # Home court factor — AS-OF-DATE home win rate per team.
    # LEAKAGE FIX: previously this was each team's full-season home win rate
    # (computed from every home game, including the current row's own result and
    # future games). Now it's the team's expanding home win rate over PRIOR home
    # games only (shift(1)), reset per season and carried forward to away games,
    # so a row never sees its own or future outcomes.
    if all(c in df.columns for c in
           ["team_abbrev", "result", "home_game", "season", "game_date"]):
        tmp = df[["team_abbrev", "season", "game_date", "result", "home_game"]].copy()
        tmp = tmp.sort_values(["team_abbrev", "season", "game_date"])
        won = tmp["result"].astype(str).str.startswith("W").astype(float)
        # Only home games contribute; away rows are NaN and get filled forward.
        home_won = won.where(tmp["home_game"] == 1)
        keys = [tmp["team_abbrev"], tmp["season"]]
        asof = home_won.groupby(keys).transform(
            lambda s: s.shift(1).expanding(min_periods=1).mean()
        )
        # Carry the last known home win rate forward onto away-game rows too.
        asof = asof.groupby(keys).ffill()
        tmp["home_court_factor"] = asof
        df["home_court_factor"] = (
            tmp["home_court_factor"].reindex(df.index).fillna(0.575).round(3)
        )
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

    # Fill NaN with season-position median, then a global-median backstop so no
    # row is left at a nonsense value (0.0 previously killed this feature's signal).
    df["opp_pts_allowed_pos"]  = df.groupby(["season", "position"])["opp_pts_allowed_pos"].transform(
        lambda x: x.fillna(x.median())
    )
    df["opp_pts_allowed_pos"]  = df["opp_pts_allowed_pos"].fillna(
        df["opp_pts_allowed_pos"].median()
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
    ewm_cols         = [c for c in df.columns if c.startswith("ewm_")]
    venue_cols       = [c for c in df.columns if c.startswith("venue_last10_")]
    trend_cols       = [c for c in df.columns if c.startswith("trend_")]
    efficiency_cols  = [c for c in df.columns if c in [
        "rolling_last3_ts_pct", "rolling_last5_ts_pct",
        "rolling_last5_pts_per_min", "rolling_last5_minutes_std",
    ]]
    context_cols     = [c for c in df.columns if c in [
        "rest_days_capped", "back_to_back", "home_game",
        "team_pace", "opp_def_rating", "opp_def_rank",
        "opp_pace", "pace_sum", "usage_x_minutes",
        "games_played_season", "position_enc",
        "days_into_season", "games_in_last_7_days", "home_court_factor",
        "opp_pts_allowed_pos", "opp_pos_defense_rank",
        "is_likely_starter", "start_rate_last10",
        "season_phase", "is_late_season",
        "teammates_out_count", "star_teammate_out", "team_minutes_vacated",
        "implied_team_total", "game_total", "team_spread", "is_favorite",
    ]]
    opp_history_cols = [c for c in df.columns if c.startswith("vs_opp_rolling")]

    feature_names = (rolling_cols + season_avg_cols + ewm_cols + venue_cols
                     + trend_cols + efficiency_cols + context_cols + opp_history_cols)

    # Deduplicate preserving order
    seen = set()
    feature_names = [f for f in feature_names if not (f in seen or seen.add(f))]

    log.info(f"  Total features: {len(feature_names)}")
    log.info(f"    Rolling avgs    : {len(rolling_cols)}")
    log.info(f"    Season avgs     : {len(season_avg_cols)}")
    log.info(f"    EWMA form       : {len(ewm_cols)}")
    log.info(f"    Venue split     : {len(venue_cols)}")
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
    df = add_ewm_features(df)
    df = add_venue_split_features(df)
    df = add_trend_features(df)
    df = add_efficiency_features(df)
    df = add_context_features(df)
    df = add_opponent_history_features(df)
    df = add_teammate_features(df)
    df = add_vegas_features(df)
    df = add_interaction_features(df)
    df = add_schedule_features(df)
    df = add_positional_defense_features(df)
    df = add_starter_features(df)
    df = add_season_phase_features(df)

    df, feature_names = select_features(df)

    df.to_csv(PROCESSED / "features.csv", index=False)
    (PROCESSED / "feature_names.txt").write_text("\n".join(feature_names))
    write_serving_features(df)

    log.info(f"Saved features.csv and feature_names.txt")
    return df, feature_names


# Games kept per player in the slim serving file — enough for the API to build a
# prediction row (latest engineered game) plus recent-form / hit-rate endpoints.
SERVING_GAMES_PER_PLAYER = 25


def write_serving_features(df):
    """Write a slim features_serving.csv: the last N games per player.

    The full features.csv is the ~100k-row training matrix (100+ MB) — too large
    for git and far more than serving needs. The API only ever reads a player's
    most recent games (latest row for /predict, recent games for /probability and
    /recent), so we commit just the tail per player. Full matrix stays local /
    gitignored; this slim file is what the deployed API loads.
    """
    slim = (
        df.sort_values("game_date")
        .groupby("player_id", group_keys=False)
        .tail(SERVING_GAMES_PER_PLAYER)
        .reset_index(drop=True)
    )
    out = PROCESSED / "features_serving.csv"
    slim.to_csv(out, index=False)
    log.info("Saved %s (%d rows, %d players)",
             out.name, len(slim), slim["player_id"].nunique())


if __name__ == "__main__":
    df, feature_names = build_features()
    quality_report(df, feature_names)