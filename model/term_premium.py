"""Term-premium decomposition math for a Gaussian affine (AF)NS model.

The production engine is ``TermStructureModels.jl`` (see ``us/engine.py`` and
``model/engine_julia``). These pure-numpy/scipy functions are our own,
independently-validated implementation of the AFNS decomposition, retained as a
lightweight **cross-check** of the engine's output (they were validated against
synthetic ground truth to 3e-7 bp; see ``validation/FINDINGS_phase1d.md``).

For a Gaussian affine model the nominal yield is

    y(tau) = fitted arbitrage-free yield = B(tau)'x + a(tau)          (Q pricing)

where ``B`` are the factor loadings and ``a(tau)`` the yield-adjustment
(convexity) term. The expectations component is the average expected future
short rate under the **physical (P)** measure,

    y~(tau) = (1/tau) * integral_0^tau E^P_t[r_{t+s}] ds
            = rho'theta + rho' * [ K^{-1}(I - e^{-K tau}) / tau ] * (x_t - theta)

with short rate ``r_t = rho'x_t``, P-mean ``theta`` and P mean-reversion ``K``.
The term premium is the difference ``TP(tau) = y(tau) - y~(tau)``, matching the
SF Fed convention (fitted = expected + term premium). Reference:
Christensen, Diebold & Rudebusch (2011); the expected-short-rate projection is
the standard Gaussian-ATSM result.

All functions here are pure numpy/scipy and unit-testable with no external
dependencies (``model/tests/test_term_premium.py``).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "expected_rate_yields",
    "fitted_yields",
    "term_premium",
]


def expected_rate_yields(
    states: np.ndarray,
    K: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Average expected short rate under P (the expectations component).

    Parameters
    ----------
    states : (T, n) filtered state paths x_t.
    K : (n, n) physical-measure mean-reversion matrix (continuous time).
    theta : (n,) physical-measure long-run mean.
    rho : (n,) short-rate loading, r_t = rho' x_t.
    grid : (m,) maturities in years.

    Returns (T, m) expected-average-short-rate yields (same units as the state /
    short rate — decimal if the model was estimated in decimal).
    """
    states = np.asarray(states, dtype=float)
    K = np.asarray(K, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    rho = np.asarray(rho, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()

    n = K.shape[0]
    I = np.eye(n)
    Kinv = np.linalg.inv(K)
    dev = states - theta  # (T, n)
    const = float(rho @ theta)

    out = np.empty((states.shape[0], grid.size), dtype=float)
    for j, tau in enumerate(grid):
        if tau <= 0:
            raise ValueError("maturities must be positive")
        # (1/tau) * integral_0^tau e^{-K s} ds = K^{-1}(I - e^{-K tau}) / tau
        avg_transition = Kinv @ (I - expm(-K * tau)) / tau  # (n, n)
        loading = rho @ avg_transition  # (n,)
        out[:, j] = const + dev @ loading
    return out


def fitted_yields(states: np.ndarray, B: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Arbitrage-free fitted yields y(tau) = B(tau)'x + a(tau).

    ``B`` is (m, n) factor loadings (rows = maturities), ``a`` is (m,) the
    yield-adjustment term (= -C(tau)/tau). Returns (T, m).
    """
    states = np.asarray(states, dtype=float)
    B = np.asarray(B, dtype=float)
    a = np.asarray(a, dtype=float).ravel()
    return states @ B.T + a


def term_premium(fitted: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """TP = fitted arbitrage-free yield - expected-average-short-rate yield."""
    return np.asarray(fitted, dtype=float) - np.asarray(expected, dtype=float)
