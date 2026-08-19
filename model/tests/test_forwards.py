"""Tests for forward zero rates derived from the reported curve."""
import numpy as np
import pandas as pd
import pytest

from model.forwards import FORWARD_LABEL, forward_from_curve, forward_rate


def test_5y5y_is_twice_10y_minus_5y():
    """The canonical case: f(5,10) = 2*y10 - y5 for continuously compounded zeros."""
    assert forward_rate(0.03, 5, 0.04, 10) == pytest.approx(2 * 0.04 - 0.03)


def test_flat_curve_forward_equals_the_flat_level():
    """A flat zero curve implies every forward equals that level (no term structure)."""
    for t1, t2 in [(1, 5), (2, 7), (5, 10), (10, 30)]:
        assert forward_rate(0.025, t1, 0.025, t2) == pytest.approx(0.025)


def test_spot_case_and_bad_range():
    assert forward_rate(np.nan, 0, 0.031, 7) == pytest.approx(0.031)   # t1=0 -> spot
    with pytest.raises(ValueError):
        forward_rate(0.03, 10, 0.04, 10)


def test_forward_is_linear_so_the_decomposition_survives():
    """THE property that makes this cheap: the transform is linear in yields, so the
    forward of (observed - expected) equals forward-observed minus forward-expected.
    A forward term premium therefore needs no re-estimation."""
    rng = np.random.default_rng(0)
    obs5, obs10 = rng.normal(0.03, 0.01, 200), rng.normal(0.04, 0.01, 200)
    exp5, exp10 = rng.normal(0.025, 0.008, 200), rng.normal(0.03, 0.008, 200)
    tp5, tp10 = obs5 - exp5, obs10 - exp10
    f_obs = forward_rate(obs5, 5, obs10, 10)
    f_exp = forward_rate(exp5, 5, exp10, 10)
    f_tp = forward_rate(tp5, 5, tp10, 10)
    np.testing.assert_allclose(f_tp, f_obs - f_exp, atol=1e-15)


def test_forward_from_curve_picks_grid_legs_and_refuses_to_interpolate():
    idx = pd.date_range("2026-01-01", periods=4, freq="D")
    curve = pd.DataFrame({2.0: 0.030, 5.0: 0.035, 10.0: 0.040}, index=idx)
    f = forward_from_curve(curve, start=5, tenor=5)
    assert np.allclose(f.to_numpy(), 2 * 0.040 - 0.035)
    assert f.name == "5y5y"
    # the user's example: 5y tenor starting in 2y -> needs 2y and 7y; 7y is absent
    with pytest.raises(KeyError):
        forward_from_curve(curve, start=2, tenor=5)
    # spot passthrough
    assert np.allclose(forward_from_curve(curve, start=0, tenor=10).to_numpy(), 0.040)


def test_labels():
    assert FORWARD_LABEL(0, 10) == "10y"
    assert FORWARD_LABEL(5, 5) == "5y5y"
    assert FORWARD_LABEL(2, 5) == "2y5y"
