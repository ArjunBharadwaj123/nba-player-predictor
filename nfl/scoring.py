"""
NFL fantasy scoring
===================
One canonical place for turning *predicted component statistics* into fantasy
points. We never train fantasy-point models — the stat projections are identical
across scoring formats; only the reception conversion changes.

Formats
-------
PPR (points per reception), Half-PPR, and No-PPR differ ONLY by the per-reception
multiplier:

    RECEPTION_MULTIPLIERS = {"ppr": 1.0, "half_ppr": 0.5, "no_ppr": 0.0}

Skill-position scoring (QB/RB/WR/TE)
------------------------------------
    0.04  per passing yard
    4     per passing TD
   -2     per passing interception
    0.1   per rushing yard
    6     per rushing TD
    0.1   per receiving yard
    6     per receiving TD
   (reception multiplier) per reception
   -2     per fumble lost   (only when a fumble prediction is available)

Kicker scoring
--------------
Kicker fantasy points do NOT change between PPR formats and live in a separate,
documented function with named constants (see ``kicker_fantasy_points``).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# ── Scoring formats ────────────────────────────────────────────────────────────
RECEPTION_MULTIPLIERS: dict[str, float] = {
    "ppr": 1.0,
    "half_ppr": 0.5,
    "no_ppr": 0.0,
}

VALID_SCORING_FORMATS = tuple(RECEPTION_MULTIPLIERS.keys())

SCORING_FORMAT_LABELS = {
    "ppr": "PPR",
    "half_ppr": "Half PPR",
    "no_ppr": "No PPR",
}

# ── Skill-position scoring constants ────────────────────────────────────────────
PTS_PER_PASSING_YARD       = 0.04
PTS_PER_PASSING_TD         = 4.0
PTS_PER_INTERCEPTION       = -2.0
PTS_PER_RUSHING_YARD       = 0.1
PTS_PER_RUSHING_TD         = 6.0
PTS_PER_RECEIVING_YARD     = 0.1
PTS_PER_RECEIVING_TD       = 6.0
PTS_PER_FUMBLE_LOST        = -2.0

# ── Kicker scoring constants ────────────────────────────────────────────────────
# Distance-tiered scoring. If the source data cannot reliably separate FGs by
# distance for a serving row, fall back to SIMPLE_FG_POINTS (documented).
PTS_PER_EXTRA_POINT        = 1.0
PTS_FG_0_39                = 3.0
PTS_FG_40_49               = 4.0
PTS_FG_50_PLUS             = 5.0
SIMPLE_FG_POINTS           = 3.0   # used only when distance breakdown unavailable


def resolve_scoring_format(fmt: str | None) -> str:
    """Normalize a scoring-format string. Raises ValueError on unsupported."""
    key = (fmt or "ppr").strip().lower()
    if key not in RECEPTION_MULTIPLIERS:
        raise ValueError(
            f"Unsupported scoring format '{fmt}'. "
            f"Use one of: {', '.join(VALID_SCORING_FORMATS)}."
        )
    return key


def reception_multiplier(fmt: str | None) -> float:
    return RECEPTION_MULTIPLIERS[resolve_scoring_format(fmt)]


def _g(stats: Mapping[str, float], key: str) -> float:
    """Fetch a stat as float, treating missing/None/NaN as 0.0.

    Stats that do not apply to a position are simply absent -> 0, per spec.
    """
    val = stats.get(key)
    if val is None:
        return 0.0
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(f):
        return 0.0
    return f


def skill_fantasy_points(stats: Mapping[str, float], scoring_format: str = "ppr") -> float:
    """Fantasy points for QB/RB/WR/TE from predicted component stats.

    ``stats`` may contain any subset of the component keys; anything missing is
    treated as zero (a stat that doesn't apply to the position). ``fumbles_lost``
    is applied ONLY when present (a real prediction) — an absent fumble stat
    never fabricates a -2.
    """
    mult = reception_multiplier(scoring_format)
    # PTS_PER_INTERCEPTION is already negative (-2.0), so it is added directly.
    pts = (
        _g(stats, "passing_yards")         * PTS_PER_PASSING_YARD
        + _g(stats, "passing_tds")         * PTS_PER_PASSING_TD
        + _g(stats, "passing_interceptions") * PTS_PER_INTERCEPTION
        + _g(stats, "rushing_yards")       * PTS_PER_RUSHING_YARD
        + _g(stats, "rushing_tds")         * PTS_PER_RUSHING_TD
        + _g(stats, "receiving_yards")     * PTS_PER_RECEIVING_YARD
        + _g(stats, "receiving_tds")       * PTS_PER_RECEIVING_TD
        + _g(stats, "receptions")          * mult
    )
    if "fumbles_lost" in stats and stats.get("fumbles_lost") is not None:
        pts += _g(stats, "fumbles_lost") * PTS_PER_FUMBLE_LOST
    return float(pts)


def kicker_fantasy_points(stats: Mapping[str, float]) -> float:
    """Fantasy points for a kicker. Independent of PPR format.

    Prefers distance-tiered FG scoring when distance buckets are present:
        field_goals_made_0_39, field_goals_made_40_49, field_goals_made_50_plus
    Otherwise falls back to a documented flat SIMPLE_FG_POINTS per made FG using
    ``field_goals_made``. Always adds PTS_PER_EXTRA_POINT per ``extra_points_made``.
    """
    ep = _g(stats, "extra_points_made") * PTS_PER_EXTRA_POINT

    has_buckets = any(
        k in stats and stats.get(k) is not None
        for k in ("field_goals_made_0_39", "field_goals_made_40_49",
                  "field_goals_made_50_plus")
    )
    if has_buckets:
        fg = (
            _g(stats, "field_goals_made_0_39")   * PTS_FG_0_39
            + _g(stats, "field_goals_made_40_49") * PTS_FG_40_49
            + _g(stats, "field_goals_made_50_plus") * PTS_FG_50_PLUS
        )
    else:
        # Simplified: distance information not reliably available at serve time.
        fg = _g(stats, "field_goals_made") * SIMPLE_FG_POINTS
    return float(ep + fg)


def fantasy_points(stats: Mapping[str, float],
                   position: str,
                   scoring_format: str = "ppr") -> float:
    """Dispatch fantasy scoring by position. Kickers ignore the scoring format."""
    if str(position).upper() == "K":
        return kicker_fantasy_points(stats)
    return skill_fantasy_points(stats, scoring_format)


# ── Fantasy-point simulation (intervals + over/under) ───────────────────────────
# Fantasy points combine several correlated predicted stats, so we do NOT simply
# apply the scoring formula to each independent lower/upper bound. Instead we
# simulate correlated component stat lines from each target's predicted mean and
# its p15/p85-implied spread, apply physical constraints, score each simulated
# line, and read percentiles / over-rates off the simulated distribution.

def _sigma_from_interval(lo: float, hi: float, mean: float) -> float:
    """Approximate a Gaussian sigma from a p15/p85 interval (~±1.036 sigma)."""
    if hi is not None and lo is not None and hi > lo:
        return max((hi - lo) / 2.07, 1e-6)
    # Fall back to a fraction of the mean when no interval is available.
    return max(abs(mean) * 0.35, 1e-6)


def simulate_fantasy_points(
    means: Mapping[str, float],
    intervals: Mapping[str, tuple[float, float]] | None,
    position: str,
    scoring_format: str = "ppr",
    n_sims: int = 4000,
    correlation: float = 0.35,
    seed: int = 42,
) -> dict:
    """Monte-Carlo fantasy-point distribution from correlated component stats.

    Returns a dict with the simulated fantasy mean, p15/p85 interval, and the
    raw simulated array (so an over/under probability can be read from the same
    draws — see ``over_under_probability``).

    ``correlation`` applies a single shared latent "good game" factor across the
    volume/production stats (documented approximation of the strong positive
    correlation between e.g. targets, receptions and receiving yards).
    """
    rng = np.random.default_rng(seed)
    intervals = intervals or {}

    keys = [k for k in means.keys()]
    if not keys:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "samples": np.zeros(n_sims)}

    # Shared latent factor drives positive correlation between component stats.
    shared = rng.standard_normal(n_sims)
    sims: dict[str, np.ndarray] = {}
    for k in keys:
        mean = float(means[k])
        lo, hi = intervals.get(k, (None, None))
        sigma = _sigma_from_interval(lo, hi, mean)
        idio = rng.standard_normal(n_sims)
        z = np.sqrt(correlation) * shared + np.sqrt(1.0 - correlation) * idio
        draws = mean + sigma * z
        sims[k] = draws

    # Physical constraints: no negative volume/production; TDs/counts are >= 0.
    from nfl.config import NONNEGATIVE_TARGETS
    for k in keys:
        if k in NONNEGATIVE_TARGETS:
            sims[k] = np.clip(sims[k], 0.0, None)
        # completions can't exceed attempts; receptions can't exceed targets.
    if "attempts" in sims and "completions" in sims:
        sims["completions"] = np.minimum(sims["completions"], sims["attempts"])
    if "targets" in sims and "receptions" in sims:
        sims["receptions"] = np.minimum(sims["receptions"], sims["targets"])

    fp = np.zeros(n_sims)
    for i in range(n_sims):
        line = {k: float(sims[k][i]) for k in keys}
        fp[i] = fantasy_points(line, position, scoring_format)

    return {
        "mean": float(np.mean(fp)),
        "low": float(np.percentile(fp, 15)),
        "high": float(np.percentile(fp, 85)),
        "samples": fp,
    }


def over_under_probability(samples: np.ndarray, threshold: float,
                           direction: str = "over") -> float:
    """Probability from simulated draws that fantasy points beat a threshold."""
    if samples is None or len(samples) == 0:
        return 0.0
    if direction == "under":
        p = float(np.mean(samples < threshold))
    else:
        p = float(np.mean(samples > threshold))
    return max(0.01, min(0.99, p))
