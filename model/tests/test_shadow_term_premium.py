"""Tests for the shadow-model term premium (expected floored short rate under P).

Decisive check: the Gaussian-call closed form vs a Monte-Carlo simulation of the
P-dynamics with the short rate floored at the ELB. Plus the collapse property
(→ plain-AFNS expectation away from the bound) and the flooring monotonicity.
"""
import numpy as np

from model import term_premium as tp
from model import shadow_rate as sr
from model import ekf

GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
RHO = np.array([1.0, 1.0, 0.0])


def _params():
    K = np.diag([0.06, 0.30, 0.60])
    theta = np.array([0.04, -0.015, 0.0])
    Sigma = np.diag([0.008, 0.010, 0.012])
    return K, theta, Sigma


def test_collapse_to_plain_expectation_away_from_bound():
    K, theta, Sigma = _params()
    rng = np.random.default_rng(0)
    states = theta + 0.02 * rng.standard_normal((20, 3))
    # ELB far below any short rate -> flooring never binds -> plain expectation.
    floored = tp.expected_floored_short_rate_yields(
        states, K, theta, Sigma, elb=-1.0, rho=RHO, grid=GRID, n_per_year=48)
    plain = tp.expected_rate_yields(states, K, theta, RHO, GRID)
    assert np.max(np.abs(floored - plain)) < 5e-5  # < 0.5 bp


def test_flooring_raises_expectation_when_bound_binds():
    K, theta, Sigma = _params()
    # A deeply negative shadow state so the floor bites; max(s,ELB) >= s.
    states = np.array([[-0.05, -0.02, 0.0]])
    floored = tp.expected_floored_short_rate_yields(
        states, K, theta, Sigma, elb=0.0, rho=RHO, grid=GRID, n_per_year=48)
    plain = tp.expected_rate_yields(states, K, theta, RHO, GRID)
    assert np.all(floored >= plain - 1e-9)
    assert np.all(floored >= 0.0 - 1e-9)          # >= ELB everywhere


def test_closed_form_matches_monte_carlo():
    K, theta, Sigma = _params()
    elb = -0.001
    x0 = np.array([0.010, -0.020, 0.005])          # near the bound so flooring matters
    taus = np.array([1.0, 5.0, 10.0])
    cf = tp.expected_floored_short_rate_yields(
        x0[None, :], K, theta, Sigma, elb=elb, rho=RHO, grid=taus, n_per_year=96)[0]

    # Monte-Carlo: exact-OU simulate x on a fine step, floor the short rate, average.
    rng = np.random.default_rng(7)
    n_paths, steps_per_year = 40000, 96
    mc = []
    for tau in taus:
        n = int(round(tau * steps_per_year))
        dt = tau / n
        A, Q = ekf._transition(K, np.diag(Sigma), dt)
        L = np.linalg.cholesky(Q)
        X = np.tile(x0, (n_paths, 1))
        acc = np.zeros(n_paths)
        s0 = RHO @ x0
        acc += 0.5 * np.maximum(s0, elb)           # trapezoid endpoint at u=0
        for k in range(1, n + 1):
            X = theta + (X - theta) @ A.T + rng.standard_normal((n_paths, 3)) @ L.T
            s = X @ RHO
            fl = np.maximum(s, elb)
            acc += (0.5 if k == n else 1.0) * fl
        mc.append((acc * dt / tau).mean())
    mc = np.array(mc)
    # MC standard error ~ vol/sqrt(paths); require agreement to ~1.5 bp.
    assert np.max(np.abs(cf - mc)) < 1.5e-4, f"cf={cf}, mc={mc}"


def test_shadow_tp_equals_plain_tp_far_from_bound():
    # End-to-end: with the bound pushed away, shadow fitted - floored expectation
    # == plain fitted - plain expectation (the splice-continuity guarantee).
    K, theta, Sigma = _params()
    lam = 0.5
    rng = np.random.default_rng(3)
    states = theta + 0.015 * rng.standard_normal((15, 3))
    pre = sr.precompute(lam, Sigma, -1.0, GRID, n_per_year=48)
    fitted = np.array([sr.elb_yields(states[t], pre) for t in range(len(states))])
    floored = tp.expected_floored_short_rate_yields(
        states, K, theta, Sigma, elb=-1.0, rho=RHO, grid=GRID, n_per_year=48)
    shadow_tp = fitted - floored
    # plain: fitted (same, bound away) minus plain expectation
    plain_tp = fitted - tp.expected_rate_yields(states, K, theta, RHO, GRID)
    assert np.max(np.abs(shadow_tp - plain_tp)) < 5e-5
