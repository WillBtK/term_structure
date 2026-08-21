"""Offline tests for the Okun level -> inflation-uncertainty map (no network)."""
import numpy as np
import pandas as pd
import pytest

from model.okun import OkunParams, fit_hockey_stick, rolling_level_vol


def test_recovers_a_known_hockey_stick():
    """Synthetic ground truth: if the data IS a hockey stick, the fit must find it."""
    rng = np.random.default_rng(0)
    level = rng.uniform(-0.5, 12.0, 400)
    true = OkunParams(floor=0.60, slope=0.40, kink=1.25)
    vol = true.sigma(level) + rng.normal(0, 0.05, 400)
    got = fit_hockey_stick(level, vol)
    assert got.floor == pytest.approx(0.60, abs=0.05)
    assert got.slope == pytest.approx(0.40, abs=0.05)
    assert got.kink == pytest.approx(1.25, abs=0.5)
    assert got.r2 > 0.95


def test_flat_below_the_kink_and_rising_above():
    """The whole economic point: no extra uncertainty until inflation clears the kink."""
    p = OkunParams(floor=0.6, slope=0.4, kink=1.25)
    assert p.sigma(-1.0) == p.sigma(0.0) == p.sigma(1.25) == pytest.approx(0.6)
    assert p.sigma(3.25) == pytest.approx(0.6 + 0.4 * 2.0)
    # and it is monotone non-decreasing across a scenario path
    path = p.sigma(np.array([0.0, 1.0, 2.0, 4.0, 8.0]))
    assert np.all(np.diff(path) >= 0)


def test_sigma_accepts_scalar_or_path():
    p = OkunParams(floor=0.6, slope=0.4, kink=1.25)
    assert isinstance(p.sigma(3.0), float)
    out = p.sigma(np.array([2.0, 3.0]))
    assert out.shape == (2,)


def test_anchoring_hits_the_observed_reading_without_touching_the_slope():
    """Anchoring re-levels to an observed uncertainty but must not re-shape the map,
    or it could flatter the level->uncertainty link it is supposed to be testing."""
    p = OkunParams(floor=0.6, slope=0.4, kink=1.25, r2=0.8, n=300)
    a = p.anchored_to(observed_vol=0.9, at_level=2.5)
    assert a.sigma(2.5) == pytest.approx(0.9)
    assert a.slope == p.slope and a.kink == p.kink
    # the shift is a pure level move: differences are preserved exactly
    assert a.sigma(5.0) - a.sigma(2.5) == pytest.approx(p.sigma(5.0) - p.sigma(2.5))


def test_rolling_level_vol_shapes_and_units():
    idx = pd.date_range("1990-01-01", periods=200, freq="MS")
    s = pd.Series(np.full(200, 3.0), index=idx)      # perfectly steady 3% inflation
    out = rolling_level_vol(s, window=120)
    assert list(out.columns) == ["level", "vol"]
    assert len(out) == 200 - 120 + 1
    assert out["level"].iloc[-1] == pytest.approx(3.0)
    assert out["vol"].iloc[-1] == pytest.approx(0.0, abs=1e-9)   # steady => no vol


def test_fit_refuses_a_sample_too_small_to_mean_anything():
    with pytest.raises(ValueError, match="at least 8"):
        fit_hockey_stick([1.0, 2.0], [0.5, 0.6])
