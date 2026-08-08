"""Extended Kalman filter estimation of the shadow-rate AFNS (US), Christensen-
Rudebusch 2015.

The measurement equation (observed yields as a function of the latent state via
the Krippner option formula, `model.shadow_rate.elb_yields`) is nonlinear, so we
estimate the Gaussian state-space model by quasi-max-likelihood with an EKF that
linearizes the measurement via its analytic Jacobian (`elb_yields_and_jac`).

Specification (independent-factor AFNS, matching SF Fed / our Julia setup):
- State x=(L,S,C); P-dynamics OU with **diagonal** mean-reversion `K^P=diag(κ)`,
  long-run mean `θ`, **diagonal** vol `Σ=diag(σ)`.
- Discrete transition over Δ: `x_t = θ + A (x_{t-1}-θ) + η`, `A=diag(e^{-κΔ})`,
  `Q = diag(σ_i²(1-e^{-2κ_iΔ})/(2κ_i))` (closed form; limit σ_i²Δ as κ_i→0).
- Measurement: `y_t = elb_yields(x_t; λ,Σ,ELB,grid) + ε`, `ε~N(0, h²I)`.
- ELB is a fixed calibrated input (not estimated), per the README.

Parameters estimated (11): κ(3), θ(3), σ(3), λ(1), h(1); positivity via log.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import shadow_rate as sr

_LOG_FLOOR = 1e-8


@dataclass
class ShadowParams:
    kappa: np.ndarray      # (3,) P mean-reversion, >0
    theta: np.ndarray      # (3,) P long-run mean
    sigma: np.ndarray      # (3,) factor vols, >0
    lambda_: float         # NS decay, >0
    h: float               # measurement-error SD, >0

    def pack(self) -> np.ndarray:
        return np.concatenate([
            np.log(np.maximum(self.kappa, _LOG_FLOOR)),
            self.theta,
            np.log(np.maximum(self.sigma, _LOG_FLOOR)),
            [np.log(max(self.lambda_, _LOG_FLOOR)), np.log(max(self.h, _LOG_FLOOR))],
        ])

    @staticmethod
    def unpack(v: np.ndarray) -> "ShadowParams":
        v = np.asarray(v, dtype=float)
        return ShadowParams(
            kappa=np.exp(v[0:3]),
            theta=v[3:6].copy(),
            sigma=np.exp(v[6:9]),
            lambda_=float(np.exp(v[9])),
            h=float(np.exp(v[10])),
        )


def _transition(kappa: np.ndarray, sigma: np.ndarray, dt: float):
    """Diagonal A and Q for the OU transition over dt."""
    a = np.exp(-kappa * dt)
    # q_i = sigma_i^2 (1 - e^{-2 k dt}) / (2k); limit sigma_i^2 dt as k->0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            kappa > 1e-10,
            sigma**2 * (1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa),
            sigma**2 * dt,
        )
    return np.diag(a), np.diag(q)


def _uncond_cov(a_diag: np.ndarray, q_diag: np.ndarray) -> np.ndarray:
    """Stationary state covariance P0 for diagonal A,Q: P0_i = q_i/(1-a_i^2)."""
    denom = 1.0 - a_diag**2
    p0 = np.where(denom > 1e-8, q_diag / np.where(denom > 1e-8, denom, 1.0), q_diag * 50.0)
    return np.diag(p0)


def negloglik(packed: np.ndarray, yields: np.ndarray, grid: np.ndarray,
              elb: float, dt: float, n_per_year: int = 48) -> float:
    """EKF negative log-likelihood for the shadow-rate AFNS."""
    p = ShadowParams.unpack(packed)
    Sigma = np.diag(p.sigma)
    try:
        pre = sr.precompute(p.lambda_, Sigma, elb, grid, n_per_year=n_per_year)
    except Exception:
        return 1e12
    A, Q = _transition(p.kappa, p.sigma, dt)
    a_diag = np.diag(A)
    m = yields.shape[1]
    H = (p.h**2) * np.eye(m)

    x = p.theta.copy()
    P = _uncond_cov(a_diag, np.diag(Q))
    theta = p.theta
    ll = 0.0
    const = m * np.log(2.0 * np.pi)
    for t in range(yields.shape[0]):
        # predict
        x = theta + A @ (x - theta)
        P = A @ P @ A.T + Q
        # measurement update (nonlinear via option formula; analytic Jacobian)
        y_hat, J = sr.elb_yields_and_jac(x, pre)
        v = yields[t] - y_hat
        S = J @ P @ J.T + H
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return 1e12
        alpha = np.linalg.solve(L, v)
        logdetS = 2.0 * np.sum(np.log(np.diag(L)))
        ll += -0.5 * (const + logdetS + alpha @ alpha)
        Sinv = np.linalg.inv(S)
        Kg = P @ J.T @ Sinv
        x = x + Kg @ v
        P = P - Kg @ J @ P
    if not np.isfinite(ll):
        return 1e12
    return -ll


def filter_states(p: ShadowParams, yields: np.ndarray, grid: np.ndarray,
                  elb: float, dt: float, n_per_year: int = 48):
    """Run the EKF at fixed params; return filtered states (T,3) and fitted
    yields (T,m)."""
    Sigma = np.diag(p.sigma)
    pre = sr.precompute(p.lambda_, Sigma, elb, grid, n_per_year=n_per_year)
    A, Q = _transition(p.kappa, p.sigma, dt)
    theta = p.theta
    m = yields.shape[1]
    H = (p.h**2) * np.eye(m)
    x = theta.copy()
    P = _uncond_cov(np.diag(A), np.diag(Q))
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


def default_init(yields: np.ndarray, grid: np.ndarray) -> ShadowParams:
    """Sensible data-driven starting parameters (used when no warm start given)."""
    short = yields[:, 0]
    long = yields[:, -1]
    level0 = float(np.mean(long))
    slope0 = float(np.mean(short - long))   # S loads +1 at short end
    return ShadowParams(
        kappa=np.array([0.1, 0.3, 0.5]),
        theta=np.array([level0, slope0, 0.0]),
        sigma=np.array([0.01, 0.01, 0.01]),
        lambda_=0.5,
        h=0.001,
    )


def estimate(yields: np.ndarray, grid: np.ndarray, elb: float, dt: float,
             warm_start: ShadowParams | None = None, maxiter: int = 2000,
             n_per_year: int = 24, restarts: int = 2):
    """QML/EKF estimation via Nelder-Mead with restarts.

    Nelder-Mead is robust to the option-formula kink at the bound; restarting it
    from the previous optimum (standard practice) escapes premature simplex
    collapse. Returns (ShadowParams, last result, loglik).
    """
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
        x0 = res.x  # restart simplex from current best
    return ShadowParams.unpack(res.x), res, -float(res.fun)
