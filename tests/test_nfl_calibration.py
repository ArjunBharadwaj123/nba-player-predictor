"""Interval calibration: the widening factor + its effect on coverage."""
import json

import numpy as np
import pytest

from conftest import requires_artifacts, MODELS_SAVED
from nfl.models.evaluate import _calibrate_factor, TARGET_COVERAGE


def test_calibrate_factor_reaches_target_when_band_too_narrow():
    rng = np.random.default_rng(0)
    yt = rng.normal(50, 20, 2000)
    pred = np.full(2000, 50.0)
    # A too-narrow band (±14 ≈ ±0.7σ, ~52% coverage) that a k<=3 stretch can fix
    # (reaching ~70% needs ±1.04σ ≈ ±21, i.e. k ≈ 1.5).
    lo = pred - 14.0
    hi = pred + 14.0
    raw_cov = np.mean((yt >= lo) & (yt <= hi))
    k, cal_cov = _calibrate_factor(yt, pred, lo, hi)
    assert raw_cov < TARGET_COVERAGE          # band really is too tight
    assert k > 1.0                            # so it must widen
    assert cal_cov >= raw_cov                 # and coverage improves
    assert cal_cov >= TARGET_COVERAGE - 0.02  # to about the nominal level


def test_calibrate_factor_no_widening_when_already_covered():
    yt = np.full(500, 50.0)                    # everything inside any band
    pred = np.full(500, 50.0)
    lo, hi = pred - 10, pred + 10
    k, cal_cov = _calibrate_factor(yt, pred, lo, hi)
    assert k == 1.0
    assert cal_cov >= TARGET_COVERAGE


@requires_artifacts
def test_calibration_file_present_and_factors_valid():
    path = MODELS_SAVED / "calibration.json"
    if not path.exists():
        pytest.skip("calibration.json not built yet")
    calib = json.loads(path.read_text())
    assert set(calib).issubset({"QB", "RB", "WR", "TE", "K"})
    for _pos, targets in calib.items():
        for _t, k in targets.items():
            assert 1.0 <= float(k) <= 3.0     # bounded widening factor


@requires_artifacts
def test_calibrated_coverage_not_worse_than_raw():
    path = MODELS_SAVED / "eval_report.json"
    if not path.exists():
        pytest.skip("eval_report.json not built yet")
    rep = json.loads(path.read_text())
    for _pos, pr in rep["positions"].items():
        for _t, m in pr["targets"].items():
            if "interval_coverage_calibrated" in m:
                assert m["interval_coverage_calibrated"] >= m["interval_coverage_p15_p85"]
