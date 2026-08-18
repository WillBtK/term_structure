"""Tests for the shifting-endpoint anchor in the expected-rate functions."""
import numpy as np
import pytest

from model import term_premium as tp


def _setup(T=6):
    K = np.diag([0.2, 0.3, 0.5])
    theta = np.array([0.02, -0.01, 0.0])          # rho'theta = 0.01 (level+slope)
    rho = np.array([1.0, 1.0, 0.0])
    Sigma = np.diag([0.004, 0.006, 0.009])
    grid = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    rng = np.random.default_rng(0)
    states = theta + rng.normal(0, 0.01, size=(T, 3))
    return states, K, theta, rho, Sigma, grid


def test_endpoint_equal_to_const_is_a_noop():
    """endpoint == rho'theta must reproduce the no-endpoint result (continuity)."""
    states, K, theta, rho, Sigma, grid = _setup()
    const = float(rho @ theta)
    base = tp.expected_rate_yields(states, K, theta, rho, grid)
    same = tp.expected_rate_yields(states, K, theta, rho, grid,
                                   endpoint=np.full(len(states), const))
    assert np.allclose(base, same, atol=1e-12)
    # floored version too
    bf = tp.expected_floored_short_rate_yields(states, K, theta, Sigma, -0.02, rho, grid, n_per_year=24)
    sf = tp.expected_floored_short_rate_yields(states, K, theta, Sigma, -0.02, rho, grid,
                                               n_per_year=24, endpoint=np.full(len(states), const))
    assert np.allclose(bf, sf, atol=1e-12)


def test_higher_endpoint_lifts_long_end_more_than_short():
    """Raising the endpoint above rho'theta lifts expected rates, and by more at long
    maturities (the level anchor decays into the average with horizon)."""
    states, K, theta, rho, Sigma, grid = _setup()
    const = float(rho @ theta)
    ep = const + 0.01                              # +100bp neutral
    base = tp.expected_rate_yields(states, K, theta, rho, grid)
    lift = tp.expected_rate_yields(states, K, theta, rho, grid,
                                   endpoint=np.full(len(states), ep))
    d = (lift - base).mean(axis=0)                 # per-maturity mean lift
    assert np.all(d >= -1e-12)                      # everywhere non-negative
    assert d[-1] > d[0] + 1e-4                      # 30y lifted more than 0.5y
    assert abs(d[-1] - 0.01) < 2e-3                 # 30y lift ~ the full +100bp (endpoint reached)
    assert d[0] < 3e-3                              # 0.5y barely moves (near today's short rate)


def test_nan_endpoint_rows_fall_back_to_constant_theta():
    states, K, theta, rho, Sigma, grid = _setup(T=5)
    const = float(rho @ theta)
    ep = np.array([np.nan, const + 0.02, np.nan, const + 0.02, np.nan])
    base = tp.expected_rate_yields(states, K, theta, rho, grid)
    out = tp.expected_rate_yields(states, K, theta, rho, grid, endpoint=ep)
    # NaN rows identical to baseline; finite rows differ
    assert np.allclose(out[0], base[0]) and np.allclose(out[2], base[2]) and np.allclose(out[4], base[4])
    assert not np.allclose(out[1], base[1])
