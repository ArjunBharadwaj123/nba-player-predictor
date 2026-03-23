"""
Canonical list of feature groups.
Keep this in sync with engineer.py.
"""

ROLLING_WINDOWS = [3, 5, 10]   # games

PLAYER_SEASON_FEATURES = [
    "season_pts_avg", "season_reb_avg", "season_ast_avg",
    "season_stl_avg", "season_blk_avg", "season_min_avg",
    "season_usage_rate", "season_ts_pct",
]

ROLLING_FEATURES = [
    f"last{w}_{stat}"
    for w in ROLLING_WINDOWS
    for stat in ["pts", "reb", "ast", "stl", "blk", "minutes", "usage"]
]

OPPONENT_FEATURES = [
    "opp_def_rating", "opp_pts_allowed_pg",
    "opp_reb_allowed_pg", "opp_pace",
    "opp_rank_defense",
]

GAME_CONTEXT_FEATURES = [
    "home_game",           # 1 = home, 0 = away
    "rest_days",           # days since last game
    "back_to_back",        # 1 if 0 rest days
    "team_pace",
    "is_playoffs",
]

MATCHUP_FEATURES = [
    "primary_defender_dpws",    # defensive win shares of likely matchup defender
    "position_opp_def_rank",    # how well opp guards the player's position
]

ALL_FEATURES = (
    PLAYER_SEASON_FEATURES
    + ROLLING_FEATURES
    + OPPONENT_FEATURES
    + GAME_CONTEXT_FEATURES
    + MATCHUP_FEATURES
)

print(f"Total planned features: {len(ALL_FEATURES)}")
