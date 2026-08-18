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
from scipy.stats import norm

__all__ = [
    "expected_rate_yields",
    "expected_floored_short_rate_yields",
    "fitted_yields",
    "term_premium",
    "mean_reversion_floor",
]


def mean_reversion_floor(K: np.ndarray, kappa_min: float) -> np.ndarray:
    """Floor the P mean-reversion so the expected-rate path reverts at a plausible
    speed, i.e. every factor's mean-reversion rate is at least ``kappa_min``
    (equivalently, its half-life is at most ``ln 2 / kappa_min``).

    Yields-only ML lands near a unit root (the level factor's half-life is
    effectively infinite), so E[avg short over 10y] collapses onto *today's* short
    rate and the term premium degenerates into the term spread — over-tracking the
    policy cycle (the symptom that diverges from the SF Fed Christensen-Rudebusch
    series in a hiking cycle). Imposing a floor restores meaningful reversion to the
    long-run mean theta, so the expected path is anchored and the premium behaves
    like the benchmark. ``kappa_min`` is calibrated (US, vs SF Fed) to a ~3.5-year
    half-life. Only speeds *below* the floor are lifted; already-fast factors are
    unchanged.

    Production K^P is diagonal (independent-factor spec) so the floor is applied
    elementwise; a non-diagonal K is handled by flooring the real parts of its
    eigenvalues and reconstructing.
    """
    K = np.asarray(K, dtype=float)
    if np.allclose(K, np.diag(np.diag(K))):
        return np.diag(np.maximum(np.diag(K), kappa_min))
    w, V = np.linalg.eig(K)
    w = np.where(w.real < kappa_min, kappa_min + 1j * w.imag, w)
    return np.real(V @ np.diag(w) @ np.linalg.inv(V))


def expected_rate_yields(
    states: np.ndarray,
    K: np.ndarray,
    theta: np.ndarray,
    rho: np.ndarray,
    grid: np.ndarray,
    endpoint: "np.ndarray | None" = None,
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

    ep = None if endpoint is None else np.asarray(endpoint, dtype=float).ravel()
    if ep is not None:
        ep = np.where(np.isfinite(ep), ep, const)               # NaN rows -> no shift
    out = np.empty((states.shape[0], grid.size), dtype=float)
    for j, tau in enumerate(grid):
        if tau <= 0:
            raise ValueError("maturities must be positive")
        # (1/tau) * integral_0^tau e^{-K s} ds = K^{-1}(I - e^{-K tau}) / tau
        avg_transition = Kinv @ (I - expm(-K * tau)) / tau  # (n, n)
        loading = rho @ avg_transition  # (n,)
        out[:, j] = const + dev @ loading
        if ep is not None:                                      # shifting endpoint (see floored version)
            out[:, j] += (ep - const) * (1.0 - loading[0])
    return out


def _pmeasure_vol_on_grid(K: np.ndarray, Sigma: np.ndarray, rho: np.ndarray,
                          u: np.ndarray) -> np.ndarray:
    """P-measure conditional vol of the short rate at horizons ``u`` (u[0]=0):

        sigma^P(u) = sqrt( rho' [∫_0^u e^{-K v} ΣΣ' e^{-K' v} dv] rho ).

    Cumulative-trapezoid over ``u`` — O(len(u)) matrix exponentials.
    """
    K = np.asarray(K, dtype=float)
    SS = np.asarray(Sigma, dtype=float) @ np.asarray(Sigma, dtype=float).T
    rho = np.asarray(rho, dtype=float).ravel()
    g = np.empty(u.size)
    for k, t in enumerate(u):
        E = expm(-K * t)
        g[k] = rho @ (E @ SS @ E.T) @ rho
    var = np.zeros(u.size)
    du = np.diff(u)
    var[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * du)
    return np.sqrt(np.clip(var, 0.0, None))


def expected_floored_short_rate_yields(
    states: np.ndarray,
    K: np.ndarray,
    theta: np.ndarray,
    Sigma: np.ndarray,
    elb: float,
    rho: np.ndarray,
    grid: np.ndarray,
    n_per_year: int = 48,
    endpoint: "np.ndarray | None" = None,
) -> np.ndarray:
    """Average expected **floored** short rate under P — the expectations leg of
    the *shadow-rate* term premium.

    The observed short rate is ``r = max(s, ELB)`` with shadow rate ``s = rho'x``.
    Since ``s_{t+u}`` is Gaussian under P with mean ``mu_u`` and sd ``sigma^P_u``,

        E^P_t[max(s_{t+u}, ELB)] = ELB + (mu_u-ELB)Φ(d) + sigma^P_u φ(d),
        d = (mu_u-ELB)/sigma^P_u

    (a Gaussian call struck at ELB — the P-measure analog of the Krippner Q-forward
    in ``model.shadow_rate``). This function integrates it over ``[0,tau]``:
    ``ỹ_lb(tau) = (1/tau) ∫_0^tau E^P[max(s,ELB)] du`` on the observed ``grid``.

    As ``ELB → -∞`` (or rates ≫ ELB) it collapses to ``expected_rate_yields`` — so
    the shadow term premium coincides with the plain-AFNS term premium away from
    the bound (continuity across a splice with a plain/regime model).

    Parameters mirror ``expected_rate_yields`` plus ``Sigma`` (n,n factor vol, the
    same Σ used in the P-dynamics innovation) and ``elb``.
    """
    from model import shadow_rate as sr  # shared fine u-grid + quadrature weights

    states = np.asarray(states, dtype=float)
    K = np.asarray(K, dtype=float)
    theta = np.asarray(theta, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    rho = np.asarray(rho, dtype=float).ravel()
    grid = np.asarray(grid, dtype=float).ravel()

    # Reuse the shadow-rate quadrature scaffolding: u-grid on [0, max(grid)] and
    # per-maturity trapezoid weights W with y(tau_i) = Σ_k W[i,k] f(u_k).
    pre = sr.precompute(0.5, Sigma, elb, grid, n_per_year=n_per_year)
    u, W = pre.u, pre.weights                                   # (Kk,), (m,Kk)

    sigP = _pmeasure_vol_on_grid(K, Sigma, rho, u)              # (Kk,)
    mrow = np.array([rho @ expm(-K * t) for t in u])           # (Kk, n): rho'e^{-Ku}
    dev = states - theta                                        # (T, n)
    const = float(rho @ theta)
    mu = const + dev @ mrow.T                                   # (T, Kk): E^P[s_{t+u}]
    if endpoint is not None:
        # Shifting-endpoint anchor (Kozicki-Tinsley): re-target the steady-state
        # shadow short rate rho'theta to a per-date ``endpoint`` (e.g. a survey/r*+pi*
        # neutral), by shifting the LEVEL-factor mean by delta_t = endpoint_t - rho'theta.
        # Adds delta_t*(1 - rho'e^{-Ku}·e_L): 0 at u=0 (today's short rate untouched) and
        # -> delta_t as u->inf (long-run reverts to ``endpoint``). Leaves Q/fitted yields.
        ep = np.asarray(endpoint, dtype=float).ravel()          # (T,)
        ep = np.where(np.isfinite(ep), ep, const)               # NaN rows -> no shift
        delta = ep - const                                      # (T,)
        mu = mu + delta[:, None] * (1.0 - mrow[:, 0][None, :])

    floored = np.empty_like(mu)
    pos = sigP > 0
    if np.any(~pos):                                            # u=0: sigma=0 -> max
        floored[:, ~pos] = np.maximum(mu[:, ~pos], elb)
    if np.any(pos):
        s = sigP[pos][None, :]
        d = (mu[:, pos] - elb) / s
        floored[:, pos] = elb + (mu[:, pos] - elb) * norm.cdf(d) + s * norm.pdf(d)
    return floored @ W.T                                        # (T, m)


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
