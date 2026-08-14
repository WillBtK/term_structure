"""Extended Kalman filter estimation of the shadow-rate AFNS (US), Christensen-
Rudebusch 2015.

The measurement equation (observed yields as a function of the latent state via
the Krippner option formula, `model.shadow_rate.elb_yields`) is nonlinear, so we
estimate the Gaussian state-space model by quasi-max-likelihood with an EKF that
linearizes the measurement via its analytic Jacobian (`elb_yields_and_jac`).

Specification (AFNS): state x=(L,S,C); P-dynamics OU
    x_t = θ + A (x_{t-1}-θ) + η,  A = e^{-K^P Δ},  η ~ N(0, Q),
    Q = ∫_0^Δ e^{-K^P u} ΣΣ' e^{-K^P' u} du  (conditional covariance).
`K^P` is a **full** 3×3 matrix (CR allow unrestricted P-dynamics; the earlier
diagonal/independent-factor form is the special case `K^P = diag(κ)`). Σ is kept
**diagonal** (so the CR closed-form ω² in shadow_rate applies). Measurement
`y_t = elb_yields(x_t; λ,Σ,ELB,grid) + ε`, `ε ~ N(0, h²I)`; ELB is fixed.

Parameters estimated (17): K^P(9), θ(3), σ(3), λ(1), h(1). σ,λ,h via log for
positivity; K^P unconstrained (stationarity enforced by penalizing unstable A).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from model import shadow_rate as sr

_LOG_FLOOR = 1e-8


@dataclass
class ShadowParams:
    K: np.ndarray          # (3,3) P mean-reversion (full)
    theta: np.ndarray      # (3,) P long-run mean
    sigma: np.ndarray      # (3,) factor vols, >0 (diagonal Σ)
    lambda_: float         # NS decay, >0
    h: float               # measurement-error SD, >0
    elb: float | None = None  # effective lower bound (set when ML-estimated; else
                              # the fixed value passed to negloglik/estimate)

    def pack(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.K, dtype=float).ravel(),          # 9
            np.asarray(self.theta, dtype=float),              # 3
            np.log(np.maximum(self.sigma, _LOG_FLOOR)),       # 3
            [np.log(max(self.lambda_, _LOG_FLOOR)), np.log(max(self.h, _LOG_FLOOR))],
        ])

    @staticmethod
    def unpack(v: np.ndarray) -> "ShadowParams":
        v = np.asarray(v, dtype=float)
        return ShadowParams(
            K=v[0:9].reshape(3, 3),
            theta=v[9:12].copy(),
            sigma=np.exp(v[12:15]),
            lambda_=float(np.exp(v[15])),
            h=float(np.exp(v[16])),
        )

    @classmethod
    def from_diagonal(cls, kappa, theta, sigma, lambda_, h) -> "ShadowParams":
        return cls(K=np.diag(np.asarray(kappa, dtype=float)), theta=np.asarray(theta, float),
                   sigma=np.asarray(sigma, float), lambda_=float(lambda_), h=float(h))


def _transition(K: np.ndarray, sigma: np.ndarray, dt: float):
    """A = e^{-K dt} and Q = conditional covariance over dt (general K)."""
    A = scipy.linalg.expm(-np.asarray(K, dtype=float) * dt)
    Q = sr.cond_cov(K, np.diag(sigma), dt)
    return A, Q


def _stationary(A: np.ndarray) -> bool:
    return float(np.max(np.abs(np.linalg.eigvals(A)))) < 1.0 - 1e-9


def _uncond_cov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Stationary state covariance: solve A P A' - P + Q = 0."""
    try:
        P0 = scipy.linalg.solve_discrete_lyapunov(A, Q)
        if not np.all(np.isfinite(P0)):
            raise ValueError
        return 0.5 * (P0 + P0.T)
    except Exception:
        return np.eye(A.shape[0]) * 1.0  # diffuse fallback


def _measure_update(x, P, y_row, pre, h2: float, var_mult_row=None):
    """One EKF measurement update with per-period missing-data masking.

    Only the finite entries of ``y_row`` are used, so a ragged panel (e.g. the
    10y NaN before it is published in the early GSW sample) is handled by shrinking
    the measurement block for that date. ``var_mult_row`` (m,), if given, is a
    per-maturity multiplier on the measurement variance h^2 for this date (1.0 =
    normal; a large value downweights that maturity — used for the YCC-pegged 10y).
    Returns ``(x_upd, P_upd, ll_inc)``, or ``(None, None, None)`` on a Cholesky
    failure. A fully-missing row leaves the state at its prediction (``ll_inc=0``).
    """
    y_hat, J = sr.elb_yields_and_jac(x, pre)
    mask = np.isfinite(y_row)
    mt = int(mask.sum())
    if mt == 0:
        return x, P, 0.0
    r_diag = h2 * (np.ones(y_row.size) if var_mult_row is None
                   else np.asarray(var_mult_row, dtype=float))
    if mt < y_row.size:
        y_hat, J, y_row, r_diag = y_hat[mask], J[mask], y_row[mask], r_diag[mask]
    v = y_row - y_hat
    S = J @ P @ J.T + np.diag(r_diag)
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return None, None, None
    alpha = np.linalg.solve(L, v)
    ll_inc = -0.5 * (mt * np.log(2.0 * np.pi)
                     + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)
    Kg = P @ J.T @ np.linalg.inv(S)
    return x + Kg @ v, P - Kg @ J @ P, ll_inc


def negloglik(packed, yields, grid, elb, dt, n_per_year: int = 48,
              var_mult=None) -> float:
    """EKF negative log-likelihood for the (full-K^P) shadow-rate AFNS.

    ``yields`` may contain NaNs (ragged maturities); they are masked per date.
    ``var_mult`` (T,m), if given, is a per-date per-maturity multiplier on the
    measurement variance (for the YCC downweight); default is homoskedastic.
    """
    p = ShadowParams.unpack(packed)
    if np.any(p.sigma <= 0) or p.lambda_ <= 0:
        return 1e12
    try:
        A, Q = _transition(p.K, p.sigma, dt)
    except Exception:
        return 1e12
    if not _stationary(A):
        return 1e12
    try:
        pre = sr.precompute(p.lambda_, np.diag(p.sigma), elb, grid, n_per_year=n_per_year)
    except Exception:
        return 1e12

    theta = p.theta
    h2 = p.h**2
    x = theta.copy()
    P = _uncond_cov(A, Q)
    ll = 0.0
    for t in range(yields.shape[0]):
        x = theta + A @ (x - theta)
        P = A @ P @ A.T + Q
        vm = None if var_mult is None else var_mult[t]
        x, P, ll_inc = _measure_update(x, P, yields[t], pre, h2, var_mult_row=vm)
        if ll_inc is None:
            return 1e12
        ll += ll_inc
    return -ll if np.isfinite(ll) else 1e12


def filter_states(p: ShadowParams, yields, grid, elb, dt, n_per_year: int = 48,
                  var_mult=None):
    """Run the EKF at fixed params; return filtered states (T,3) and fits (T,m).

    Handles ragged ``yields`` (NaNs masked per date); ``fits`` are always the
    full-grid model-implied yields (present even where the datum was missing).
    ``var_mult`` (T,m) optionally downweights maturities per date (YCC peg).
    """
    A, Q = _transition(p.K, p.sigma, dt)
    pre = sr.precompute(p.lambda_, np.diag(p.sigma), elb, grid, n_per_year=n_per_year)
    theta = p.theta
    h2 = p.h**2
    x = theta.copy()
    P = _uncond_cov(A, Q)
    m = yields.shape[1]
    xs = np.empty((yields.shape[0], 3))
    fits = np.empty((yields.shape[0], m))
    for t in range(yields.shape[0]):
        x = theta + A @ (x - theta)
        P = A @ P @ A.T + Q
        vm = None if var_mult is None else var_mult[t]
        x_upd, P_upd, ll_inc = _measure_update(x, P, yields[t], pre, h2, var_mult_row=vm)
        if ll_inc is not None:
            x, P = x_upd, P_upd
        xs[t] = x
        fits[t], _ = sr.elb_yields_and_jac(x, pre)
    return xs, fits


def default_init(yields, grid) -> ShadowParams:
    """Data-driven starting parameters (diagonal K^P start).

    Uses ``nanmean`` so a ragged panel (e.g. Japan's 30y, NaN before 1999) does
    not poison the level/slope starting values — a plain mean over a
    partially-missing column is NaN, which makes the whole starting point
    infeasible and the optimizer never gets traction. The long-end level is taken
    from the longest maturity that actually has data, and the slope from the first
    vs last observed maturities.
    """
    col_ok = np.array([np.isfinite(yields[:, j]).any() for j in range(yields.shape[1])])
    last = int(np.max(np.flatnonzero(col_ok)))     # longest observed maturity
    first = int(np.min(np.flatnonzero(col_ok)))    # shortest observed maturity
    level0 = float(np.nanmean(yields[:, last]))
    slope0 = float(np.nanmean(yields[:, first] - yields[:, last]))
    return ShadowParams.from_diagonal(
        kappa=[0.1, 0.3, 0.5], theta=[level0, slope0, 0.0],
        sigma=[0.01, 0.01, 0.01], lambda_=0.5, h=0.001)


def _diag_pack(p: ShadowParams) -> np.ndarray:
    """Reduced 11-vector for the independent-factor model (diagonal K^P):
    [log diag(K), theta, log sigma, log lambda, log h]."""
    return np.concatenate([
        np.log(np.maximum(np.diag(p.K), _LOG_FLOOR)), p.theta,
        np.log(np.maximum(p.sigma, _LOG_FLOOR)),
        [np.log(max(p.lambda_, _LOG_FLOOR)), np.log(max(p.h, _LOG_FLOOR))],
    ])


def _diag_unpack(v: np.ndarray) -> ShadowParams:
    v = np.asarray(v, dtype=float)
    return ShadowParams.from_diagonal(np.exp(v[0:3]), v[3:6], np.exp(v[6:9]),
                                      float(np.exp(v[9])), float(np.exp(v[10])))


def estimate(yields, grid, elb, dt, warm_start: ShadowParams | None = None,
             maxiter: int = 4000, n_per_year: int = 24, restarts: int = 1,
             diagonal_K: bool = True, estimate_elb: bool = False, var_mult=None):
    """QML/EKF estimation via Nelder-Mead with restarts.

    ``diagonal_K=True`` (default) estimates the **independent-factor** AFNS(3)
    (diagonal K^P & Σ, 11 params) — the recommended production spec: robustly
    identified, SF-Fed-consistent, portable across countries. ``diagonal_K=False``
    estimates a full K^P (17 params); in US tests this only adds a marginal,
    economically negligible likelihood gain (identical yield fit / shadow path).

    ``estimate_elb=True`` adds the effective lower bound as a free (unconstrained,
    since it may be negative) parameter — the Christensen-Rudebusch approach to
    *deriving* the ELB from the data rather than calibrating it. The likelihood is
    smooth in ELB via the Krippner option formula. Note the ELB is only
    identified when yields approach the bound (weakly identified for the US, which
    barely hit it; well identified for markets that went clearly negative). The
    fixed ``elb`` argument is used as the ELB when ``estimate_elb=False`` and as
    the ELB starting value when ``estimate_elb=True``. Warm-start to speed
    convergence.

    ``var_mult`` (T,m), if given, is a per-date per-maturity multiplier on the
    measurement variance — used for the Japan YCC downweight (the BoJ-pegged 10y
    is effectively ignored as a market price while the peg is tight).

    Returns ``(params, opt_result, loglik)``; ``params.elb`` carries the ELB used
    (fixed or estimated).
    """
    from scipy.optimize import minimize

    init = warm_start if warm_start is not None else default_init(yields, grid)
    base_pack = _diag_pack if diagonal_K else (lambda p: p.pack())
    base_unpack = _diag_unpack if diagonal_K else ShadowParams.unpack

    if estimate_elb:
        def obj(v):
            return negloglik(base_unpack(v[:-1]).pack(), yields, grid,
                             float(v[-1]), dt, n_per_year, var_mult=var_mult)
        x0 = np.append(base_pack(init), elb)

        def finalize(v):
            p = base_unpack(v[:-1]); p.elb = float(v[-1]); return p
    else:
        def obj(v):
            return negloglik(base_unpack(v).pack(), yields, grid, elb, dt,
                             n_per_year, var_mult=var_mult)
        x0 = base_pack(init)

        def finalize(v):
            p = base_unpack(v); p.elb = elb; return p

    # Bound function evaluations, not just iterations: scipy's Nelder-Mead leaves
    # maxfun=inf when only maxiter is set, so on a flat/degenerate surface the
    # repeated simplex-shrink steps (each ~N evals) run for hours. Capping maxfev
    # bounds worst-case runtime while warm starts keep well-posed fits converging
    # long before the cap.
    res = None
    for _ in range(max(restarts, 1)):
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "maxfev": maxiter,
                                "xatol": 1e-7, "fatol": 1e-5, "adaptive": True})
        if res.success:
            break
        x0 = res.x
    return finalize(res.x), res, -float(res.fun)
