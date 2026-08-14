"""Pure-math tests for the term-premium decomposition (no engine import).

Only numpy/scipy are needed — ``term_premium`` imports the engine's ``nss`` only
lazily inside ``decompose``/``short_rate_loading``, so these run in the fast
offline suite.
"""
import numpy as np
from model.term_premium import mean_reversion_floor


def test_mean_reversion_floor_diagonal():
    # Near-unit-root level (kappa 0.001), moderate slope/curv; floor at 3.5y HL.
    K = np.diag([0.001, 0.15, 0.5])
    kmin = np.log(2) / 3.5
    Kf = mean_reversion_floor(K, kmin)
    assert np.allclose(Kf, np.diag(np.diag(Kf)))           # stays diagonal (off-diag 0)
    assert Kf[0, 0] == kmin                                 # near-unit-root lifted to floor
    assert Kf[1, 1] == kmin                                 # 0.15 < kmin -> lifted
    assert Kf[2, 2] == 0.5                                  # already faster -> unchanged
    # every factor now reverts at least as fast as the floor
    assert np.all(np.diag(Kf) >= kmin - 1e-12)


def test_mean_reversion_floor_general_real_eigs():
    K = np.array([[0.02, 0.01, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.6]])
    kmin = 0.1
    Kf = mean_reversion_floor(K, kmin)
    ev = np.sort(np.linalg.eigvals(Kf).real)
    assert ev[0] >= kmin - 1e-9                             # slowest mode floored
    assert np.all(np.abs(np.linalg.eigvals(Kf).imag) < 1e-9)  # stays real
import pytest
from scipy.linalg import expm

from term_premium import expected_rate_yields, fitted_yields, term_premium


def _stable_K(rng, n):
    # Diagonally dominant with positive diagonal -> stable mean reversion.
    K = 0.1 * rng.standard_normal((n, n))
    K[np.diag_indices(n)] = rng.uniform(0.3, 1.5, size=n)
    return K


def test_expected_rate_matches_quadrature():
    rng = np.random.default_rng(0)
    n = 5
    K = _stable_K(rng, n)
    theta = rng.normal(0, 0.02, size=n)
    rho = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    states = rng.normal(0, 0.03, size=(4, n))
    grid = np.array([0.25, 1.0, 2.0, 5.0, 10.0])

    closed = expected_rate_yields(states, K, theta, rho, grid)

    # Independent fine-quadrature of (1/tau) integral rho'E[x_{t+s}] ds.
    def quad(x0, tau, nsteps=20000):
        s = np.linspace(0.0, tau, nsteps)
        vals = np.array([rho @ (theta + expm(-K * si) @ (x0 - theta)) for si in s])
        return np.trapz(vals, s) / tau

    for i, x0 in enumerate(states):
        for j, tau in enumerate(grid):
            assert closed[i, j] == pytest.approx(quad(x0, tau), abs=1e-8)


def test_expected_rate_short_maturity_is_short_rate():
    rng = np.random.default_rng(1)
    n = 5
    K = _stable_K(rng, n)
    theta = rng.normal(0, 0.02, size=n)
    rho = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    states = rng.normal(0, 0.03, size=(3, n))
    # As tau -> 0, expected-average-short-rate -> instantaneous short rate rho'x.
    ey = expected_rate_yields(states, K, theta, rho, np.array([1e-6]))
    assert np.allclose(ey[:, 0], states @ rho, atol=1e-4)


def test_expected_rate_long_maturity_approaches_mean():
    rng = np.random.default_rng(2)
    n = 5
    K = _stable_K(rng, n)
    theta = rng.normal(0, 0.02, size=n)
    rho = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
    states = rng.normal(0, 0.03, size=(3, n))
    # As tau -> infinity, the average expected short rate -> rho'theta.
    ey = expected_rate_yields(states, K, theta, rho, np.array([500.0]))
    assert np.allclose(ey[:, 0], rho @ theta, atol=1e-3)


def test_fitted_and_term_premium_identity():
    rng = np.random.default_rng(3)
    T, n, m = 6, 5, 3
    states = rng.normal(0, 0.03, size=(T, n))
    B = rng.normal(size=(m, n))
    a = rng.normal(0, 1e-3, size=m)
    fit = fitted_yields(states, B, a)
    assert fit.shape == (T, m)
    # fitted = B x + a
    assert np.allclose(fit, states @ B.T + a)
    # TP = fitted - expected, so fitted == expected + TP exactly.
    expected = rng.normal(size=(T, m))
    tp = term_premium(fit, expected)
    assert np.allclose(fit, expected + tp)


def test_expected_rate_rejects_nonpositive_maturity():
    K = np.eye(3)
    with pytest.raises(ValueError):
        expected_rate_yields(np.zeros((1, 3)), K, np.zeros(3), np.ones(3), np.array([0.0]))
