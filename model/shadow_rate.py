"""Christensen-Rudebusch / Krippner shadow-rate AFNS math (3-factor B-AFNS).

Pure numpy/scipy. This module implements the option-based shadow-rate extension
of the arbitrage-free Nelson-Siegel model (Christensen & Rudebusch 2015/2016,
building on Krippner 2013): the observed short rate is floored at an effective
lower bound (ELB), `r_t = max(s_t, ELB)`, where `s_t` is the Gaussian *shadow*
short rate. Away from the bound the model collapses to plain AFNS.

State `x = (L, S, C)` (level, slope, curvature). Conventions:
- **Shadow short rate** `s = L + S` (loading `rho = [1, 1, 0]`).
- **Shadow instantaneous forward** `f_s(tau) = b(tau)'x + a(tau)` with NS forward
  loadings `b(tau) = [1, e^{-λτ}, λτ e^{-λτ}]` and forward convexity
  `a(tau) = -1/2 · B(tau)' ΣΣ' B(tau)`, where `B(tau)` is the AFNS bond-price
  loading `[τ, (1-e^{-λτ})/λ, (1-e^{-λτ})/λ - τ e^{-λτ}]` (so that
  `y(τ)=B(τ)'x/τ - A(τ)/τ` and `f = d(τy)/dτ`).
- **Krippner ELB forward:**
      f_lb(tau) = ELB + (f_s(tau) - ELB)·Φ(z) + ω(tau)·φ(z),  z = (f_s - ELB)/ω(tau)
  As `ω→0`: `f_lb → max(f_s, ELB)`. As `ELB→-∞`: `f_lb → f_s` (plain AFNS).
- **ELB yield:** `y_lb(τ) = (1/τ) ∫_0^τ f_lb(u) du` (numeric quadrature).

ω(tau) — the conditional volatility of the shadow short rate at horizon τ under
the risk-neutral (Q) dynamics. Verified against Christensen & Rudebusch, "Est-
imating Shadow-Rate Term Structure Models with Near-Zero Yields" (FRB SF WP
2013-07): their eq. for the shadow forward (their eq. 8) has loadings
`[1, e^{-λτ}, λτ e^{-λτ}]`, the forward yield-adjustment `A_f = -∂A/∂T =
-½ B'ΣΣ'B`, the ELB forward `f_lb = f·Φ(f/ω)+ω·φ(f/ω)` (bound=0; generalized to
ELB≠0 per CR 2016), and the closed-form ω²(τ) on p.12. For the independent-
factor (diagonal Σ) B-AFNS(3) we use, that closed form is

      ω(τ)² = σ11²·τ + σ22²·(1-e^{-2λτ})/(2λ)
            + σ33²·[(1-e^{-2λτ})/(4λ) - ½τe^{-2λτ} - ½λτ²e^{-2λτ}]

(`omega_closed_form`), which for diagonal Σ equals `ρ'V(τ)ρ`,
`V(τ)=∫_0^τ e^{-K^Q u}ΣΣ'e^{-K^Q' u}du`, `K^Q=[[0,0,0],[0,λ,-λ],[0,0,λ]]`
(`omega_on_grid`) — the two agree in tests. (For a *full* Σ, CR's ω² carries
extra cross terms; use the numeric path there.)

All quantities except `f_s` depend only on parameters (λ, Σ, ELB), not on the
state — so `precompute(...)` builds them once per parameter vector and
`elb_yields(x, pre)` is cheap per date (used inside the EKF).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm

RHO = np.array([1.0, 1.0, 0.0])  # shadow short rate s = L + S


# --------------------------------------------------------------------------- #
# AFNS loadings and convexity                                                  #
# --------------------------------------------------------------------------- #
def forward_loadings(lambda_: float, tau: np.ndarray) -> np.ndarray:
    """NS instantaneous-forward loadings b(tau) = [1, e^{-λτ}, λτ e^{-λτ}].

    Returns array shape (len(tau), 3).
    """
    tau = np.asarray(tau, dtype=float)
    e = np.exp(-lambda_ * tau)
    return np.stack([np.ones_like(tau), e, lambda_ * tau * e], axis=-1)


def price_loadings(lambda_: float, tau: np.ndarray) -> np.ndarray:
    """AFNS bond-price loadings B(tau) = [τ, (1-e^{-λτ})/λ,
    (1-e^{-λτ})/λ - τ e^{-λτ}]. Returns shape (len(tau), 3)."""
    tau = np.asarray(tau, dtype=float)
    e = np.exp(-lambda_ * tau)
    slope = (1.0 - e) / lambda_
    return np.stack([tau, slope, slope - tau * e], axis=-1)


def forward_convexity(lambda_: float, tau: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Forward-rate convexity a(tau) = -1/2 · B(tau)' ΣΣ' B(tau). Shape (len(tau),)."""
    B = price_loadings(lambda_, tau)          # (m, 3)
    SS = np.asarray(Sigma) @ np.asarray(Sigma).T
    # quadratic form per row: -0.5 * B_i' SS B_i
    return -0.5 * np.einsum("mi,ij,mj->m", B, SS, B)


def KQ_afns(lambda_: float) -> np.ndarray:
    """Risk-neutral mean-reversion of AFNS: [[0,0,0],[0,λ,-λ],[0,0,λ]]."""
    return np.array([[0.0, 0.0, 0.0], [0.0, lambda_, -lambda_], [0.0, 0.0, lambda_]])


def cond_cov(K: np.ndarray, Sigma: np.ndarray, tau: float, n: int = 80) -> np.ndarray:
    """Conditional covariance V(tau) = ∫_0^τ e^{-K u} ΣΣ' e^{-K' u} du (Simpson)."""
    K = np.asarray(K, dtype=float)
    SS = np.asarray(Sigma, dtype=float) @ np.asarray(Sigma, dtype=float).T
    if tau <= 0:
        return np.zeros_like(SS)
    us = np.linspace(0.0, tau, 2 * n + 1)
    integrand = np.empty((us.size, *SS.shape))
    for i, u in enumerate(us):
        E = expm(-K * u)
        integrand[i] = E @ SS @ E.T
    # composite Simpson weights
    w = np.ones(us.size)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    h = (us[1] - us[0]) / 3.0
    return h * np.tensordot(w, integrand, axes=(0, 0))


def omega(lambda_: float, Sigma: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Conditional vol ω(tau) of the shadow short rate under Q. Shape (len(tau),).

    ω(tau)^2 = rho' V(tau) rho with V from the AFNS K^Q. ω(0)=0. Reference
    implementation (per-point ``cond_cov``); ``omega_on_grid`` is the fast path
    used by ``precompute``.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    KQ = KQ_afns(lambda_)
    out = np.empty(tau.shape)
    for i, t in enumerate(tau):
        V = cond_cov(KQ, Sigma, float(t))
        out[i] = np.sqrt(max(RHO @ V @ RHO, 0.0))
    return out


def omega_closed_form(lambda_: float, Sigma: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """ω(tau) via the Christensen-Rudebusch closed form (WP 2013-07, p.12), for
    the **independent-factor** (diagonal Σ) B-AFNS(3):

        ω(τ)² = σ11²·τ + σ22²·(1-e^{-2λτ})/(2λ)
              + σ33²·[(1-e^{-2λτ})/(4λ) - ½τe^{-2λτ} - ½λτ²e^{-2λτ}].

    Exact and fast (no quadrature). For a *full* lower-triangular Σ, CR's ω²
    includes additional cross terms that differ from ρ'V(τ)ρ; use the numeric
    ``omega_on_grid`` there. We estimate with diagonal Σ, so this is exact.
    """
    tau = np.asarray(tau, dtype=float)
    s = np.diag(np.asarray(Sigma, dtype=float))
    l = float(lambda_)
    e2 = np.exp(-2.0 * l * tau)
    v = (
        s[0] ** 2 * tau
        + s[1] ** 2 * (1.0 - e2) / (2.0 * l)
        + s[2] ** 2 * ((1.0 - e2) / (4.0 * l) - 0.5 * tau * e2 - 0.5 * l * tau**2 * e2)
    )
    return np.sqrt(np.clip(v, 0.0, None))


def _is_diagonal(M: np.ndarray, tol: float = 1e-12) -> bool:
    M = np.asarray(M, dtype=float)
    return bool(np.all(np.abs(M - np.diag(np.diag(M))) <= tol))


def omega_on_grid(lambda_: float, Sigma: np.ndarray, u: np.ndarray) -> np.ndarray:
    """ω on a sorted grid `u` (with u[0]=0) via one cumulative-trapezoid pass.

    ω(u_k)^2 = ∫_0^{u_k} g(t) dt with g(t) = rho' e^{-K^Q t} ΣΣ' e^{-K^Q' t} rho.
    Equivalent to `omega(lambda_, Sigma, u)` but O(len(u)) matrix exponentials
    instead of O(len(u)·n). Matches `omega` to quadrature tolerance.
    """
    u = np.asarray(u, dtype=float)
    KQ = KQ_afns(lambda_)
    SS = np.asarray(Sigma, dtype=float) @ np.asarray(Sigma, dtype=float).T
    g = np.empty(u.size)
    for k, t in enumerate(u):
        E = expm(-KQ * t)
        g[k] = RHO @ (E @ SS @ E.T) @ RHO
    var = np.zeros(u.size)
    du = np.diff(u)
    var[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * du)
    return np.sqrt(np.clip(var, 0.0, None))


# --------------------------------------------------------------------------- #
# Krippner option-based ELB forward and yields                                #
# --------------------------------------------------------------------------- #
def elb_forward(f_shadow: np.ndarray, omega_tau: np.ndarray, elb: float) -> np.ndarray:
    """Krippner ELB-constrained forward:
    f_lb = ELB + (f_s - ELB)Φ(z) + ω φ(z), z=(f_s-ELB)/ω.

    Handles ω→0 as the limit max(f_s, ELB) elementwise.
    """
    f_shadow = np.asarray(f_shadow, dtype=float)
    omega_tau = np.asarray(omega_tau, dtype=float)
    out = np.maximum(f_shadow, elb)  # ω->0 limit
    pos = omega_tau > 0
    if np.any(pos):
        d = (f_shadow[pos] - elb) / omega_tau[pos]
        out[pos] = elb + (f_shadow[pos] - elb) * norm.cdf(d) + omega_tau[pos] * norm.pdf(d)
    return out


@dataclass
class ShadowPrecompute:
    """Parameter-dependent quantities (independent of the state x), built once
    per parameter vector and reused across all dates in the EKF."""
    grid: np.ndarray            # observed maturities (years), shape (m,)
    u: np.ndarray               # fine forward grid over [0, max(grid)], shape (K,)
    b_u: np.ndarray             # forward loadings at u, (K, 3)
    a_u: np.ndarray             # forward convexity at u, (K,)
    omega_u: np.ndarray         # ω at u, (K,)
    weights: np.ndarray         # (m, K) quadrature weights so y(τ_i)=Σ_k W[i,k] f_lb(u_k)
    elb: float


def precompute(lambda_: float, Sigma: np.ndarray, elb: float,
               grid: np.ndarray, n_per_year: int = 48) -> ShadowPrecompute:
    """Build the state-independent pieces for the ELB yield map on `grid`.

    Uses a shared fine forward grid `u` on [0, max(grid)] and trapezoidal
    weights per maturity so that y(τ_i) = (1/τ_i) ∫_0^{τ_i} f_lb(u) du.
    """
    grid = np.asarray(grid, dtype=float)
    tau_max = float(grid.max())
    K = max(int(np.ceil(tau_max * n_per_year)) + 1, 8)
    u = np.linspace(0.0, tau_max, K)
    b_u = forward_loadings(lambda_, u)
    a_u = forward_convexity(lambda_, u, Sigma)
    # Exact CR closed form for diagonal (independent-factor) Σ; numeric otherwise.
    omega_u = (omega_closed_form(lambda_, Sigma, u) if _is_diagonal(Sigma)
               else omega_on_grid(lambda_, Sigma, u))

    # Per-maturity trapezoidal weights over u up to τ_i, divided by τ_i.
    W = np.zeros((grid.size, K))
    for i, tau in enumerate(grid):
        mask = u <= tau + 1e-12
        uu = u[mask]
        if uu[-1] < tau - 1e-9:  # ensure endpoint at tau
            uu = np.append(uu, tau)
        # trapezoid weights on uu, mapped back to the u index positions
        w_local = np.zeros(uu.size)
        du = np.diff(uu)
        w_local[:-1] += du / 2.0
        w_local[1:] += du / 2.0
        # place onto full grid (last node may be the appended tau -> nearest idx)
        idx = np.where(mask)[0]
        m = min(idx.size, uu.size)
        W[i, idx[:m]] += w_local[:m] / tau
        if uu.size > idx.size:  # appended tau endpoint: attribute to last real node
            W[i, idx[-1]] += w_local[-1] / tau
    return ShadowPrecompute(grid=grid, u=u, b_u=b_u, a_u=a_u, omega_u=omega_u,
                            weights=W, elb=elb)


def shadow_forward(x: np.ndarray, pre: ShadowPrecompute) -> np.ndarray:
    """Shadow instantaneous forward f_s(u) = b(u)'x + a(u) on the fine grid."""
    return pre.b_u @ np.asarray(x, dtype=float) + pre.a_u


def elb_yields(x: np.ndarray, pre: ShadowPrecompute) -> np.ndarray:
    """ELB-constrained zero yields at `pre.grid` for state x. Shape (m,)."""
    f_s = shadow_forward(x, pre)
    f_lb = elb_forward(f_s, pre.omega_u, pre.elb)
    return pre.weights @ f_lb


def elb_yields_and_jac(x: np.ndarray, pre: ShadowPrecompute):
    """ELB yields and their Jacobian w.r.t. the state x, for the EKF.

    The Krippner forward has the clean derivative d f_lb/d f_s = Φ(z), so
    ∂y/∂x = W · (Φ(z_u) ⊙ b(u)). Returns (y (m,), J (m,3)).
    """
    f_s = shadow_forward(x, pre)                         # (K,)
    pos = pre.omega_u > 0
    phi_cdf = np.where(f_s > pre.elb, 1.0, 0.0)          # ω->0 step limit
    if np.any(pos):
        z = (f_s[pos] - pre.elb) / pre.omega_u[pos]
        phi_cdf[pos] = norm.cdf(z)
    f_lb = elb_forward(f_s, pre.omega_u, pre.elb)
    y = pre.weights @ f_lb
    J = pre.weights @ (phi_cdf[:, None] * pre.b_u)       # (m, 3)
    return y, J


def shadow_short_rate(x: np.ndarray) -> float:
    """Shadow short rate s = L + S = rho'x."""
    return float(RHO @ np.asarray(x, dtype=float))
