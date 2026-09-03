"""Fantasy scoring proofs (spec-mandated)."""
import pytest

from nfl import scoring


def test_one_reception_ppr_adds_one_point():
    assert scoring.fantasy_points({"receptions": 1}, "WR", "ppr") == pytest.approx(1.0)


def test_one_reception_half_ppr_adds_half_point():
    assert scoring.fantasy_points({"receptions": 1}, "WR", "half_ppr") == pytest.approx(0.5)


def test_one_reception_no_ppr_adds_zero():
    assert scoring.fantasy_points({"receptions": 1}, "WR", "no_ppr") == pytest.approx(0.0)


def test_identical_components_correct_under_every_format():
    stats = {"receiving_yards": 100, "receiving_tds": 1, "receptions": 6, "targets": 9}
    # base (no reception pts) = 10 + 6 = 16
    assert scoring.fantasy_points(stats, "WR", "no_ppr") == pytest.approx(16.0)
    assert scoring.fantasy_points(stats, "WR", "half_ppr") == pytest.approx(16.0 + 3.0)
    assert scoring.fantasy_points(stats, "WR", "ppr") == pytest.approx(16.0 + 6.0)


def test_qb_scoring():
    stats = {"passing_yards": 300, "passing_tds": 3, "passing_interceptions": 1,
             "rushing_yards": 20, "rushing_tds": 0}
    # 300*.04 + 3*4 - 1*2 + 20*.1 = 12 + 12 - 2 + 2 = 24
    assert scoring.fantasy_points(stats, "QB", "ppr") == pytest.approx(24.0)


def test_fumble_lost_only_applied_when_present():
    with_fumble = scoring.fantasy_points({"rushing_yards": 50, "fumbles_lost": 1}, "RB", "ppr")
    without = scoring.fantasy_points({"rushing_yards": 50}, "RB", "ppr")
    assert with_fumble == pytest.approx(without - 2.0)


def test_kicker_unchanged_across_formats():
    kick = {"field_goals_made": 2, "extra_points_made": 3}
    ppr = scoring.fantasy_points(kick, "K", "ppr")
    half = scoring.fantasy_points(kick, "K", "half_ppr")
    no = scoring.fantasy_points(kick, "K", "no_ppr")
    assert ppr == half == no


def test_kicker_distance_tiers():
    # 1 FG 0-39 (3) + 1 FG 40-49 (4) + 1 FG 50+ (5) + 2 XP (2) = 14
    stats = {"field_goals_made_0_39": 1, "field_goals_made_40_49": 1,
             "field_goals_made_50_plus": 1, "extra_points_made": 2}
    assert scoring.kicker_fantasy_points(stats) == pytest.approx(14.0)


def test_kicker_simplified_fallback_when_no_buckets():
    # 3 FG * 3 + 1 XP = 10
    assert scoring.kicker_fantasy_points(
        {"field_goals_made": 3, "extra_points_made": 1}) == pytest.approx(10.0)


def test_invalid_scoring_format_rejected():
    with pytest.raises(ValueError):
        scoring.resolve_scoring_format("super_ppr")


def test_reception_multipliers_exact():
    assert scoring.RECEPTION_MULTIPLIERS == {"ppr": 1.0, "half_ppr": 0.5, "no_ppr": 0.0}


def test_stats_that_dont_apply_treated_as_zero():
    # A WR line with no passing stats scores no passing points.
    assert scoring.fantasy_points({"receiving_yards": 50}, "WR", "no_ppr") == pytest.approx(5.0)
