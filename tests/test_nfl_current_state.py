"""Current-team / current-depth resolution for NFL serving.

Covers the fix that sources a player's CURRENT team from ``players.latest_team``
and their starter status from the current-season depth chart, instead of the
(2002–2025-only) weekly rosters that silently fell back to the prior season and
left movers on their old team and clear starters marked "not a starter".

All network calls (the live nflverse loaders) are monkeypatched, so these tests
run offline.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nfl.pipeline import update as up
from nfl import serving


# ── Synthetic live data ──────────────────────────────────────────────────────
def _players_frame():
    return pd.DataFrame([
        # active clear starter
        {"gsis_id": "star", "latest_team": "KC", "status": "ACT", "last_season": 3000},
        # traded mover: current team NE, still listed on old team PHI in depth
        {"gsis_id": "mover", "latest_team": "NE", "status": "ACT", "last_season": 3000},
        # retiree: status still ACT, but old last_season and NOT in depth chart
        {"gsis_id": "retiree", "latest_team": "TB", "status": "ACT", "last_season": 1999},
        # roster-lag: last_season not bumped, but present in current depth chart
        {"gsis_id": "lagging", "latest_team": "NYJ", "status": "ACT", "last_season": 1999},
    ])


def _depth_frame():
    return pd.DataFrame([
        {"gsis_id": "star", "team": "KC", "pos_abb": "QB", "pos_rank": 1, "dt": "2026-09-01"},
        # mover appears on BOTH teams — old team as a backup, new team as starter
        {"gsis_id": "mover", "team": "PHI", "pos_abb": "WR", "pos_rank": 2, "dt": "2026-08-01"},
        {"gsis_id": "mover", "team": "NE", "pos_abb": "WR", "pos_rank": 1, "dt": "2026-09-01"},
        {"gsis_id": "lagging", "team": "NYJ", "pos_abb": "QB", "pos_rank": 1, "dt": "2026-09-01"},
    ])


@pytest.fixture
def patched_live(monkeypatch):
    monkeypatch.setattr("nfl.scraping.collect.load_current_players",
                        lambda: _players_frame())
    monkeypatch.setattr("nfl.scraping.collect.load_current_depth",
                        lambda season: _depth_frame())


# ── _current_player_map ──────────────────────────────────────────────────────
def test_current_team_uses_latest_team(patched_live):
    current_team, active_ids = up._current_player_map()
    assert current_team["mover"] == "NE"      # trade reflected (not old PHI)
    assert current_team["star"] == "KC"


def test_retiree_excluded_active_includes_roster_lag(patched_live):
    _, active_ids = up._current_player_map()
    assert "star" in active_ids and "mover" in active_ids
    assert "lagging" in active_ids            # kept via depth-chart union
    assert "retiree" not in active_ids        # old last_season, not on a depth chart


def test_player_map_none_when_unavailable(monkeypatch):
    monkeypatch.setattr("nfl.scraping.collect.load_current_players", lambda: None)
    current_team, active_ids = up._current_player_map()
    assert current_team == {} and active_ids is None


# ── _current_depth_map ───────────────────────────────────────────────────────
def test_depth_read_from_current_team(patched_live):
    current_team, _ = up._current_player_map()
    depth = up._current_depth_map(current_team)
    # mover's starter status must come from the NEW team (NE, rank 1), not PHI.
    assert depth["mover"]["is_starter"] == 1
    assert depth["mover"]["depth_rank"] == 1
    assert depth["star"]["is_starter"] == 1
    assert depth["lagging"]["is_starter"] == 1


def test_depth_map_empty_when_unavailable(monkeypatch):
    monkeypatch.setattr("nfl.scraping.collect.load_current_depth",
                        lambda season: None)
    assert up._current_depth_map({"x": "KC"}) == {}


# ── serving override + snap-share fallback ───────────────────────────────────
def _base_row(**over):
    row = {
        "gsis_id": "p1", "games_played_season": 3.0,
        "depth_rank": 3.0, "is_starter": 0, "offense_pct_roll3": 0.2,
        "missing_injury": 1,
    }
    row.update(over)
    return pd.Series(row)


def _patch_serving(monkeypatch, base, player):
    monkeypatch.setattr(serving, "latest_player_row", lambda pid: base)
    monkeypatch.setattr(serving, "find_player", lambda pid: player)
    monkeypatch.setattr(serving, "load_feature_names",
                        lambda: ["depth_rank", "is_starter", "home_away"])
    monkeypatch.setattr(serving, "load_opp_defense", lambda: {})
    monkeypatch.setattr(serving, "load_team_environment", lambda: {})


def test_serving_overrides_depth_from_current(monkeypatch):
    base = _base_row(is_starter=0, depth_rank=4.0)
    player = {"id": "p1", "team": "KC", "depth_rank": 1, "is_starter": 1}
    _patch_serving(monkeypatch, base, player)
    X = serving.build_upcoming_row("p1", "QB", {"team": "KC"})
    assert int(X["is_starter"].iloc[0]) == 1        # current depth wins
    assert float(X["depth_rank"].iloc[0]) == 1.0


def test_serving_snap_share_fallback_marks_starter(monkeypatch):
    # No depth-chart entry (depth_rank/is_starter None), but high recent snap share.
    base = _base_row(is_starter=0, offense_pct_roll3=0.8)
    player = {"id": "p1", "team": "KC", "depth_rank": None, "is_starter": None}
    _patch_serving(monkeypatch, base, player)
    X = serving.build_upcoming_row("p1", "QB", {"team": "KC"})
    assert int(X["is_starter"].iloc[0]) == 1        # snap-share fallback
