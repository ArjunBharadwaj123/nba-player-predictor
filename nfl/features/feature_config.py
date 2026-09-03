"""
Feature configuration for the NFL predictor.
============================================
Declares which raw weekly stats get the full rolling/expanding/ewma/trend
treatment, and which position-specific features augment each position's model.
The engineer reads these lists so adding a feature is a one-line change here.
"""

from __future__ import annotations

# Rolling windows applied to every volume/production stat.
ROLL_WINDOWS = [3, 5, 8]

# Base weekly stats (present in load_player_stats) that we build lagged rolling
# features from. Any that are absent for a given position are simply skipped.
# These are ROLE / VOLUME / PRODUCTION signals — never the target's same-week
# value (that would leak); the engineer always applies .shift(1) first.
ROLLING_STATS = [
    # passing
    "attempts", "completions", "passing_yards", "passing_tds",
    "passing_interceptions", "passing_air_yards", "passing_epa",
    "passing_cpoe", "sacks_suffered", "passing_first_downs",
    # rushing
    "carries", "rushing_yards", "rushing_tds", "rushing_epa",
    "rushing_first_downs",
    # receiving
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "receiving_air_yards", "receiving_epa", "receiving_first_downs",
    "target_share", "air_yards_share", "wopr", "racr",
    # role
    "offense_pct",             # snap share (from snap_counts merge)
    # fantasy
    "fantasy_points", "fantasy_points_ppr", "fantasy_points_component",
    # kicking
    "fg_made", "fg_att", "pat_made", "pat_att", "kicking_points",
]

# Position-specific feature stems. These are the raw weekly columns whose LAGGED
# rolling features matter most for a position (used to prune the feature set per
# position model and for documentation). The engineer builds rolling features for
# all ROLLING_STATS present; POSITION_FEATURE_HINTS documents emphasis.
POSITION_FEATURE_HINTS = {
    "QB": [
        "attempts", "completions", "passing_yards", "passing_tds",
        "passing_interceptions", "passing_air_yards", "passing_epa",
        "passing_cpoe", "sacks_suffered", "carries", "rushing_yards",
    ],
    "RB": [
        "carries", "rushing_yards", "rushing_tds", "rushing_epa",
        "targets", "receptions", "receiving_yards", "target_share",
        "offense_pct",
    ],
    "WR": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "receiving_air_yards", "target_share", "air_yards_share",
        "wopr", "offense_pct",
    ],
    "TE": [
        "targets", "receptions", "receiving_yards", "receiving_tds",
        "target_share", "offense_pct",
    ],
    "K": [
        "fg_made", "fg_att", "pat_made", "pat_att", "kicking_points",
    ],
}

# Non-rolling game-context features (built once per row, leakage-safe by
# construction — all known before kickoff).
CONTEXT_FEATURES = [
    "season", "week", "home_away", "days_rest", "bye_week_return",
    "spread_line_team", "game_total", "implied_team_total",
    "is_indoor", "is_grass", "temp", "wind", "div_game",
    "years_experience", "injury_designation_enc", "depth_rank",
    "is_starter", "games_played_season",
]

# Opponent-defense features (aggregated from weekly player_stats, shifted).
OPPONENT_FEATURES = [
    "opp_fantasy_allowed_pos", "opp_pass_yards_allowed",
    "opp_rush_yards_allowed", "opp_rec_yards_allowed_pos",
    "opp_pass_tds_allowed", "opp_rush_tds_allowed",
]

# Missing-data indicator columns (1 when the underlying dataset was unavailable
# or the value was imputed). Never let a real 0 masquerade as "missing".
MISSING_INDICATORS = [
    "missing_snap_share", "missing_injury", "missing_depth",
    "missing_odds", "missing_weather",
]

# Neutral imputation defaults (documented). Used ONLY where a model needs a
# numeric and the true value is unavailable; paired with a missing-indicator.
NEUTRAL_IMPUTATION = {
    "spread_line_team": 0.0,      # pick'em
    "game_total": 44.0,           # league-typical total
    "implied_team_total": 22.0,   # half of typical total
    "temp": 65.0,                 # mild
    "wind": 5.0,                  # light
    "offense_pct": 0.0,           # unknown snap share (paired w/ indicator)
    "depth_rank": 3.0,            # unknown depth
    "injury_designation_enc": 0.0,  # no designation
}
