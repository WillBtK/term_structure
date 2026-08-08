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


def negloglik(packed, yields, grid, elb, dt, n_per_year: int = 48) -> float:
    """EKF negative log-likelihood for the (full-K^P) shadow-rate AFNS."""
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
    m = yields.shape[1]
    H = (p.h**2) * np.eye(m)
    x = theta.copy()
    P = _uncond_cov(A, Q)
    ll = 0.0
    const = m * np.log(2.0 * np.pi)
    for t in range(yields.shape[0]):
        x = theta + A @ (x - theta)
        P = A @ P @ A.T + Q
        y_hat, J = sr.elb_yields_and_jac(x, pre)
        v = yields[t] - y_hat
        S = J @ P @ J.T + H
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L, v)
        ll += -0.5 * (const + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)
        Sinv = np.linalg.inv(S)
        Kg = P @ J.T @ Sinv
        x = x + Kg @ v
        P = P - Kg @ J @ P
    return -ll if np.isfinite(ll) else 1e12


def filter_states(p: ShadowParams, yields, grid, elb, dt, n_per_year: int = 48):
    """Run the EKF at fixed params; return filtered states (T,3) and fits (T,m)."""
    A, Q = _transition(p.K, p.sigma, dt)
    pre = sr.precompute(p.lambda_, np.diag(p.sigma), elb, grid, n_per_year=n_per_year)
    theta = p.theta
    m = yields.shape[1]
    H = (p.h**2) * np.eye(m)
    x = theta.copy()
    P = _uncond_cov(A, Q)
    xs = np.empty((yields.shape[0], 3))
    fits = np.empty((yields.shape[0], m))
    for t in range(yields.shape[0]):
        x = theta + A @ (x - theta)
        P = A @ P @ A.T + Q
        y_hat, J = sr.elb_yields_and_jac(x, pre)
        v = yields[t] - y_hat
        S = J @ P @ J.T + H
        Kg = P @ J.T @ np.linalg.inv(S)
        x = x + Kg @ v
        P = P - Kg @ J @ P
        xs[t] = x
        fits[t], _ = sr.elb_yields_and_jac(x, pre)
    return xs, fits


def default_init(yields, grid) -> ShadowParams:
    """Data-driven starting parameters (diagonal K^P start)."""
    level0 = float(np.mean(yields[:, -1]))
    slope0 = float(np.mean(yields[:, 0] - yields[:, -1]))
    return ShadowParams.from_diagonal(
        kappa=[0.1, 0.3, 0.5], theta=[level0, slope0, 0.0],
        sigma=[0.01, 0.01, 0.01], lambda_=0.5, h=0.001)


def estimate(yields, grid, elb, dt, warm_start: ShadowParams | None = None,
             maxiter: int = 4000, n_per_year: int = 24, restarts: int = 2):
    """QML/EKF estimation via Nelder-Mead with restarts. Full K^P (17 params).
    Warm-start from a prior (e.g. diagonal) fit to speed convergence."""
    from scipy.optimize import minimize

    init = warm_start if warm_start is not None else default_init(yields, grid)
    x0 = init.pack()
    res = None
    for _ in range(max(restarts, 1)):
        res = minimize(
            negloglik, x0, args=(yields, grid, elb, dt, n_per_year),
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-7, "fatol": 1e-5, "adaptive": True},
        )
        if res.success:
            break
        x0 = res.x
    return ShadowParams.unpack(res.x), res, -float(res.fun)
