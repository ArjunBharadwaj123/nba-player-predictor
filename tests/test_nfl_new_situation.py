"""current_nfl_season, current-season recency boost, and leakage-safe
team-environment features."""
from datetime import date

import numpy as np
import pandas as pd

from nfl.config import current_nfl_season
from nfl.models.train import recency_weights
from nfl.features.engineer import _add_team_environment_features, TEAM_ENV_FEATURES


def test_current_nfl_season_by_month():
    assert current_nfl_season(date(2026, 9, 7)) == 2026   # September -> that year
    assert current_nfl_season(date(2027, 1, 5)) == 2026   # January -> prior season
    assert current_nfl_season(date(2026, 7, 1)) == 2025   # July -> prior season


def test_recency_boost_favours_latest_season():
    # Two seasons of weekly dates; latest season should carry more weight.
    dates, seasons = [], []
    for wk in range(1, 18):
        dates.append(pd.Timestamp("2024-09-01") + pd.Timedelta(weeks=wk))
        seasons.append(2024)
    for wk in range(1, 18):
        dates.append(pd.Timestamp("2025-09-01") + pd.Timedelta(weeks=wk))
        seasons.append(2025)
    w = recency_weights(pd.Series(dates), pd.Series(seasons))
    w = np.asarray(w)
    seasons = np.asarray(seasons)
    assert (w > 0).all()
    assert w[seasons == 2025].mean() > w[seasons == 2024].mean()


def test_team_environment_is_leakage_safe():
    # One team, 4 games; team_pass_rate at week w must reflect ONLY prior weeks.
    df = pd.DataFrame({
        "season": [2023] * 4,
        "team": ["AAA"] * 4,
        "week": [1, 2, 3, 4],
        "attempts": [40, 20, 30, 10],
        "carries": [10, 20, 10, 30],
        "passing_epa": [5.0, 1.0, 3.0, 0.0],
        "rushing_epa": [1.0, 2.0, 1.0, 3.0],
        "passing_tds": [3, 1, 2, 0],
        "rushing_tds": [0, 2, 1, 3],
        "implied_team_total": [24, 20, 22, 18],
    })
    out = _add_team_environment_features(df).sort_values("week")
    # Week 1 has no prior data -> NaN (filled later in _impute).
    assert pd.isna(out[out["week"] == 1]["team_pass_rate"].iloc[0])
    # Week 2 pass rate = week1 pass rate = 40/(40+10) = 0.8 (excludes week 2).
    assert abs(out[out["week"] == 2]["team_pass_rate"].iloc[0] - 0.8) < 1e-6
    # Week 3 = mean(week1,week2 pass rate) = mean(0.8, 0.5) = 0.65.
    assert abs(out[out["week"] == 3]["team_pass_rate"].iloc[0] - 0.65) < 1e-6
    for c in TEAM_ENV_FEATURES:
        assert c in out.columns
