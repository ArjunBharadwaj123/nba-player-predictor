"""Leakage prevention + chronological splitting + missing indicators."""
import numpy as np
import pandas as pd

from nfl.features.engineer import (
    _add_rolling_features, _add_context_features, _impute_and_indicators,
    feature_columns,
)
from nfl.models.evaluate import _holdout_mask
from nfl.features.feature_config import MISSING_INDICATORS


def _toy_player():
    # One player, 6 chronological games with known receiving_yards.
    return pd.DataFrame({
        "gsis_id": ["p1"] * 6,
        "season": [2023] * 6,
        "week": [1, 2, 3, 4, 5, 6],
        "position": ["WR"] * 6,
        "opponent_team": ["A", "B", "C", "D", "E", "F"],
        "receiving_yards": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "receptions": [1, 2, 3, 4, 5, 6],
        "targets": [2, 4, 6, 8, 10, 12],
        "fantasy_points_component": [1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
    })


def test_prev_feature_is_strictly_previous_game():
    df = _add_rolling_features(_toy_player())
    # receiving_yards_prev at week w equals receiving_yards at week w-1.
    got = df.sort_values("week")["receiving_yards_prev"].tolist()
    assert np.isnan(got[0])                       # first game has no prior
    assert got[1:] == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_rolling_never_includes_current_game():
    df = _add_rolling_features(_toy_player()).sort_values("week")
    # roll3 at week 4 should average weeks 1-3 (10,20,30)=20, NOT include week 4.
    val = df[df["week"] == 4]["receiving_yards_roll3"].iloc[0]
    assert val == 20.0


def test_season_avg_is_shifted():
    df = _add_rolling_features(_toy_player()).sort_values("week")
    # season_avg at week 3 = mean(week1,week2)=15, excludes week 3.
    val = df[df["week"] == 3]["receiving_yards_season_avg"].iloc[0]
    assert val == 15.0


def test_bye_week_return_flag():
    df = _toy_player()
    df.loc[df["week"] == 5, "week"] = 7   # create a gap (skipped week 5/6)
    df = df[df["week"] != 6]
    out = _add_context_features(df).sort_values("week")
    # the row after the gap (week 7) is a bye/return.
    assert out[out["week"] == 7]["bye_week_return"].iloc[0] == 1
    assert out[out["week"] == 2]["bye_week_return"].iloc[0] == 0


def test_games_played_is_prior_count():
    out = _add_context_features(_toy_player()).sort_values("week")
    assert out["games_played_season"].tolist() == [0, 1, 2, 3, 4, 5]


def test_missing_indicators_created_and_binary():
    df = _toy_player()
    out = _impute_and_indicators(df)
    for col in MISSING_INDICATORS:
        assert col in out.columns
        assert set(out[col].unique()).issubset({0, 1})


def test_feature_columns_excludes_raw_targets():
    df = _add_rolling_features(_toy_player())
    df = _add_context_features(df)
    df = _impute_and_indicators(df)
    feats = feature_columns(df)
    # raw same-week production columns must never be features
    for leaky in ["receiving_yards", "receptions", "targets",
                  "fantasy_points_component"]:
        assert leaky not in feats
    # but their lagged versions are allowed
    assert "receiving_yards_roll3" in feats


def test_holdout_mask_selects_latest_weeks():
    df = pd.DataFrame({"season": [2023] * 10, "week": list(range(1, 11))})
    mask = _holdout_mask(df, 0.2)
    # latest 20% = weeks 9,10
    assert df[mask]["week"].tolist() == [9, 10]
    assert df[~mask]["week"].max() == 8
