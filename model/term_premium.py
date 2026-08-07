"""Term-premium decomposition for the (AF)NS model.

The vendored engine estimates the state-space model and prices arbitrage-free
yields, but does not split a yield into its expectations and term-premium
components. This module adds that step.

For a Gaussian affine model the nominal yield is

    y(tau) = fitted arbitrage-free yield = B(tau)'x + a(tau)          (Q pricing)

where ``B`` are the Nelson-Siegel factor loadings and ``a(tau) = -C(tau)/tau`` is
the yield-adjustment (convexity) term the engine already computes. The
expectations component is the average expected future short rate under the
**physical (P)** measure,

    y~(tau) = (1/tau) * integral_0^tau E^P_t[r_{t+s}] ds
            = rho'theta + rho' * [ K^{-1}(I - e^{-K tau}) / tau ] * (x_t - theta)

with short rate ``r_t = rho'x_t`` (``rho = lim_{tau->0} B(tau)``), P-mean
``theta`` and P mean-reversion ``K`` (both estimated by the engine). The term
premium is the difference:

    TP(tau) = y(tau) - y~(tau).

This matches the SF Fed convention (fitted = expected + term premium). See
Christensen, Diebold & Rudebusch (2011) for the AFNS yield-adjustment term; the
expected-short-rate projection is the standard Gaussian-ATSM result.

The core functions (``expected_rate_yields``, ``fitted_yields``,
``term_premium``) are pure numpy/scipy and unit-testable without the engine;
``decompose`` wires them to the engine's ``nss`` loadings.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

__all__ = [
    "expected_rate_yields",
    "fitted_yields",
    "term_premium",
    "short_rate_loading",
    "decompose",
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


def short_rate_loading(lambda_: float, lambda_svensson=None, eps: float = 1e-8) -> np.ndarray:
    """Short-rate loading rho = lim_{tau->0} B(tau), read off the NS loadings.

    Avoids hardcoding the factor order: evaluates ``nss.B_matrix`` at a tiny
    maturity. For (AF)NS this yields [1, 1, (1,) 0, 0] — level + slope(s), with
    curvature loadings vanishing at the short end.
    """
    from arbfree_dyn_ns.nss import B_matrix

    row = B_matrix(lambda_, np.array([eps], dtype=float), lambda_svensson).values[0]
    return np.asarray(row, dtype=float)


def decompose(
    states,
    K,
    theta,
    Sigma,
    lambda_,
    grid,
    lambda_svensson=None,
    dates=None,
):
    """Full decomposition from engine parameters -> canonical tidy frame.

    Builds the NS loadings ``B`` and the yield-adjustment ``a = -C/tau`` from the
    engine's ``nss`` module (using the estimated ``Sigma``, ``lambda``), computes
    the fitted yield, the P expected-rate yield, and the term premium, and
    returns a long-format ``pandas.DataFrame`` with columns
    ``date, maturity, term_premium, expected_rate, fitted_yield`` (decimal) —
    the schema consumed by ``validation.compare_afns``.

    ``states`` is (T, n); ``dates`` (len T) supplies the date index (defaults to
    a RangeIndex if omitted).
    """
    import pandas as pd

    from arbfree_dyn_ns.nss import B_matrix, C_vector

    states = np.asarray(states, dtype=float)
    grid = np.asarray(grid, dtype=float).ravel()

    B = B_matrix(lambda_, grid, lambda_svensson).values  # (m, n)
    a = np.asarray(C_vector(lambda_, grid, None, Sigma, lambda_svensson), dtype=float)
    rho = short_rate_loading(lambda_, lambda_svensson)

    fit = fitted_yields(states, B, a)
    exp = expected_rate_yields(states, K, theta, rho, grid)
    tp = term_premium(fit, exp)

    if dates is None:
        dates = np.arange(states.shape[0])
    dates = np.asarray(dates)

    frames = []
    for j, tau in enumerate(grid):
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "maturity": float(tau),
                    "term_premium": tp[:, j],
                    "expected_rate": exp[:, j],
                    "fitted_yield": fit[:, j],
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["date", "maturity", "term_premium", "expected_rate", "fitted_yield"]]
