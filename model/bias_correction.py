"""Bauer-Rudebusch-Wu (2012) bootstrap bias-correction of the P-dynamics.

Finite-sample estimates of the VAR(1) persistence ``Phi = e^{-K^P Δ}`` are biased
toward *too little* persistence (Kilian 1998). In a term-structure model that
understates how slowly the short rate is expected to mean-revert, so it
understates the term premium — most severely in a highly persistent sample such
as the 1970s. This module bias-corrects ``Phi`` via Kilian's
bootstrap-after-bootstrap on the VAR(1) fitted to the filtered factors, applies
the stationarity adjustment, and maps the result back to a continuous-time
``K^P`` for the expectations decomposition (`model.term_premium`).

Only the **persistence** is corrected; the unconditional mean ``θ`` (well
estimated by the sample mean) is held fixed, matching BRW — the caller keeps the
ML ``θ`` and swaps only ``K^P``.

Portable and yields-only: the alternative robustness fix — survey-anchored
expectations (Kim-Wright / D'Amico-Kim-Wei) — is not uniformly available across
the four markets; this is. References: Bauer, Rudebusch & Wu (2012, JBES);
Kilian (1998, REStat); Pope (1990).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import logm

__all__ = ["fit_var1", "phi_to_kp", "kp_to_phi", "bias_correct", "BiasCorrectResult"]


def _spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def fit_var1(states: np.ndarray) -> dict:
    """OLS VAR(1) ``x_t = c + Φ x_{t-1} + e_t`` on the filtered factors.

    Returns ``{Phi, c, Omega, resid, theta}`` with ``theta=(I-Φ)^{-1} c`` the
    implied unconditional mean.
    """
    X = np.asarray(states, dtype=float)
    n = X.shape[1]
    Y = X[1:]                                       # (T-1, n)
    Z = np.column_stack([np.ones(len(Y)), X[:-1]])  # (T-1, n+1)
    B, *_ = np.linalg.lstsq(Z, Y, rcond=None)       # (n+1, n): rows [c; Φ']
    c = B[0]
    Phi = B[1:].T
    resid = Y - Z @ B
    dof = max(len(Y) - (n + 1), 1)
    Omega = resid.T @ resid / dof
    theta = np.linalg.solve(np.eye(n) - Phi, c)
    return {"Phi": Phi, "c": c, "Omega": Omega, "resid": resid, "theta": theta}


def kp_to_phi(K: np.ndarray, dt: float) -> np.ndarray:
    """Discrete persistence ``Φ = e^{-K Δ}``."""
    from scipy.linalg import expm
    return expm(-np.asarray(K, dtype=float) * dt)


def phi_to_kp(Phi: np.ndarray, dt: float) -> np.ndarray:
    """Continuous-time mean-reversion ``K = -log(Φ)/Δ`` (real part; guarded)."""
    L = logm(np.asarray(Phi, dtype=float))
    if np.iscomplexobj(L):
        L = L.real
    return -L / dt


def _bootstrap_mean_phi(Phi, c, resid, T, n_boot, rng):
    """Mean OLS ``Φ*`` over ``n_boot`` residual-resampled VAR(1) paths of length T."""
    n = Phi.shape[0]
    E = resid - resid.mean(axis=0)          # centred innovations
    ne = len(E)
    theta = np.linalg.solve(np.eye(n) - Phi, c)
    acc = np.zeros((n, n))
    for _ in range(n_boot):
        x = np.empty((T, n))
        x[0] = theta                        # start at the unconditional mean
        idx = rng.integers(0, ne, size=T - 1)
        for t in range(1, T):
            x[t] = c + Phi @ x[t - 1] + E[idx[t - 1]]
        acc += fit_var1(x)["Phi"]
    return acc / n_boot


@dataclass
class BiasCorrectResult:
    Phi_hat: np.ndarray        # OLS persistence
    Phi_bc: np.ndarray         # bias-corrected persistence
    K_hat: np.ndarray          # continuous-time K^P (OLS)
    K_bc: np.ndarray           # continuous-time K^P (bias-corrected)
    theta: np.ndarray          # unconditional mean (held fixed)
    delta: float               # Kilian stationarity shrinkage actually applied
    rho_hat: float             # spectral radius before correction
    rho_bc: float              # spectral radius after correction
    n_boot: int


def bias_correct(states: np.ndarray, dt: float, n_boot: int = 1000,
                 seed: int = 0) -> BiasCorrectResult:
    """Kilian bias-corrected VAR(1) persistence for the filtered factors.

    Bootstrap bias ``Δ = mean(Φ*) - Φ̂``; corrected ``Φ̃ = Φ̂ - δΔ`` with ``δ``
    reduced from 1 (Kilian's stationarity adjustment) until the spectral radius is
    below 1. Only persistence is corrected; ``θ`` is held at the OLS/ML value.
    """
    X = np.asarray(states, dtype=float)
    T = len(X)
    base = fit_var1(X)
    Phi_hat, c, resid, theta = base["Phi"], base["c"], base["resid"], base["theta"]
    rng = np.random.default_rng(seed)

    mean_phi = _bootstrap_mean_phi(Phi_hat, c, resid, T, n_boot, rng)
    bias = mean_phi - Phi_hat

    # Kilian (1998) stationarity-adjusted correction.
    delta = 1.0
    Phi_bc = Phi_hat - delta * bias
    if _spectral_radius(Phi_bc) >= 1.0:
        delta = 1.0
        while delta > 0.0 and _spectral_radius(Phi_hat - delta * bias) >= 1.0:
            delta -= 0.01
        delta = max(delta, 0.0)
        Phi_bc = Phi_hat - delta * bias

    return BiasCorrectResult(
        Phi_hat=Phi_hat, Phi_bc=Phi_bc,
        K_hat=phi_to_kp(Phi_hat, dt), K_bc=phi_to_kp(Phi_bc, dt),
        theta=theta, delta=float(delta),
        rho_hat=_spectral_radius(Phi_hat), rho_bc=_spectral_radius(Phi_bc),
        n_boot=int(n_boot),
    )
