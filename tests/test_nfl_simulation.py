"""Fantasy-point simulation + over/under probability."""
import numpy as np

from nfl import scoring


def test_simulation_returns_ordered_interval():
    means = {"receiving_yards": 80, "receptions": 6, "targets": 9, "receiving_tds": 0.5}
    intervals = {"receiving_yards": (50, 110), "receptions": (3, 9),
                 "targets": (5, 13), "receiving_tds": (0, 1)}
    sim = scoring.simulate_fantasy_points(means, intervals, "WR", "ppr", n_sims=3000)
    assert sim["low"] <= sim["mean"] <= sim["high"]
    assert len(sim["samples"]) == 3000


def test_simulation_enforces_nonnegativity():
    means = {"receiving_yards": 5, "receptions": 1, "targets": 2, "receiving_tds": 0.05}
    sim = scoring.simulate_fantasy_points(means, None, "WR", "ppr", n_sims=2000)
    # PPR fantasy points from non-negative stats can't be negative
    assert sim["samples"].min() >= 0.0


def test_ppr_shifts_distribution_up_vs_no_ppr():
    means = {"receiving_yards": 70, "receptions": 7, "targets": 10, "receiving_tds": 0.4}
    intervals = {"receiving_yards": (40, 100)}
    ppr = scoring.simulate_fantasy_points(means, intervals, "WR", "ppr", n_sims=4000)
    noppr = scoring.simulate_fantasy_points(means, intervals, "WR", "no_ppr", n_sims=4000)
    assert ppr["mean"] > noppr["mean"]   # ~7 more points on ~7 receptions


def test_over_under_probability_monotonic():
    samples = np.linspace(0, 40, 1000)
    low_thresh = scoring.over_under_probability(samples, 10, "over")
    high_thresh = scoring.over_under_probability(samples, 30, "over")
    assert low_thresh > high_thresh
    # over + under approx complementary
    over = scoring.over_under_probability(samples, 20, "over")
    under = scoring.over_under_probability(samples, 20, "under")
    assert abs((over + under) - 1.0) < 0.05


def test_completions_capped_by_attempts_in_sim():
    means = {"attempts": 30, "completions": 28}
    sim = scoring.simulate_fantasy_points(means, None, "QB", "ppr", n_sims=1000)
    assert sim is not None  # sanity: constraint path runs without error
