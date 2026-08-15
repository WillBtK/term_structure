"""Joint nominal/real arbitrage-free Nelson-Siegel (AFNS) model — US spike.

A single no-arbitrage pricing kernel prices *both* the nominal and real curves, so
the inflation-risk premium (IRP) is model-consistent rather than the difference of
two independently-estimated fits (the current `us/inflation.py` "subtract" approach,
whose IRP depends on a matched-sample fudge — see `validation/FINDINGS_inflation.md`).

Spec (Christensen-Lopez-Rudebusch 2010 style), 4-factor state x = [L_N, L_R, S, C]:
  * nominal short rate  r_N = L_N + S     (rho_N = [1,0,1,0])
  * real short rate     r_R = L_R + S     (rho_R = [0,1,1,0])
  * nominal yields are a 3-factor AFNS in (L_N, S, C); real yields a 3-factor AFNS
    in (L_R, S, C) — shared slope/curvature and a shared NS decay λ. With a
    **diagonal** Σ each curve's Q-loadings/convexity are exactly the standard AFNS
    ones, so we reuse `shadow_rate` per block: the **nominal** block carries the
    Krippner ELB option (it hits the ZLB), the **real** block is plain affine
    (real cash has no floor -> elb = -inf).
  * P-dynamics: OU with **diagonal** K^P over the joint 4-factor state (identified;
    the cross-curve linkage is the shared S,C factors + the one kernel). Separate
    measurement-error scales h_N, h_R (real is far less liquid).

Reuses `shadow_rate.precompute/elb_yields_and_jac` (per block) and the
dimension-agnostic `term_premium` decomposition. Estimation is QML via an EKF, as
for the single-curve model (`model/ekf.py`).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.optimize

from model import shadow_rate as sr
from model import term_premium as tp

_LOG_FLOOR = 1e-8
RHO_N = np.array([1.0, 0.0, 1.0, 0.0])      # nominal short = L_N + S
RHO_R = np.array([0.0, 1.0, 1.0, 0.0])      # real short   = L_R + S
_NOM_SUB = [0, 2, 3]                         # (L_N, S, C) columns of the 4-state
_REAL_SUB = [1, 2, 3]                        # (L_R, S, C)
_REAL_ELB = -1.0                             # real curve: plain AFNS (no floor)


@dataclass
class JointParams:
    kappa: np.ndarray      # (4,) diagonal P mean-reversion, >0  [L_N,L_R,S,C]
    theta: np.ndarray      # (4,) P long-run mean
    sigma: np.ndarray      # (4,) factor vols, >0 (diagonal Σ)
    lambda_: float         # shared NS decay, >0
    h_n: float             # nominal measurement-error SD, >0
    h_r: float             # real measurement-error SD, >0
    elb_n: float = -0.0035  # nominal effective lower bound (fixed)

    def pack(self) -> np.ndarray:
        return np.concatenate([
            np.log(np.maximum(self.kappa, _LOG_FLOOR)),           # 4
            np.asarray(self.theta, float),                        # 4
            np.log(np.maximum(self.sigma, _LOG_FLOOR)),           # 4
            [np.log(max(self.lambda_, _LOG_FLOOR)),
             np.log(max(self.h_n, _LOG_FLOOR)),
             np.log(max(self.h_r, _LOG_FLOOR))],                  # 3
        ])

    @staticmethod
    def unpack(v: np.ndarray, elb_n: float) -> "JointParams":
        v = np.asarray(v, dtype=float)
        return JointParams(
            kappa=np.exp(v[0:4]), theta=v[4:8].copy(), sigma=np.exp(v[8:12]),
            lambda_=float(np.exp(v[12])), h_n=float(np.exp(v[13])),
            h_r=float(np.exp(v[14])), elb_n=elb_n)

    @property
    def K(self) -> np.ndarray:
        return np.diag(self.kappa)


# ------------------------------------------------------------------ measurement
def _precompute(p: JointParams, grid_n, grid_r, n_per_year=48):
    """AFNS precompute per block (nominal floored at elb_n; real plain)."""
    sig = np.asarray(p.sigma, float)
    pre_n = sr.precompute(p.lambda_, np.diag(sig[_NOM_SUB]), p.elb_n,
                          np.asarray(grid_n, float), n_per_year)
    pre_r = sr.precompute(p.lambda_, np.diag(sig[_REAL_SUB]), _REAL_ELB,
                          np.asarray(grid_r, float), n_per_year)
    return pre_n, pre_r


def _measure(x, pre_n, pre_r, m_n):
    """Stacked [nominal; real] yields and Jacobian wrt the 4-factor state."""
    yn, Jn = sr.elb_yields_and_jac(x[_NOM_SUB], pre_n)
    yr, Jr = sr.elb_yields_and_jac(x[_REAL_SUB], pre_r)
    y = np.concatenate([yn, yr])
    J = np.zeros((y.shape[0], 4))
    J[:m_n][:, _NOM_SUB] = Jn
    J[m_n:][:, _REAL_SUB] = Jr
    return y, J


def _transition(p: JointParams, dt: float):
    K = p.K
    A = scipy.linalg.expm(-K * dt)
    Q = sr.cond_cov(K, np.diag(p.sigma), dt)
    return A, Q


# --------------------------------------------------------------------- filter
def filter_states(p: JointParams, Yn, Yr, grid_n, grid_r, dt, n_per_year=48):
    """EKF over the joint panel. Yn (T,m_n), Yr (T,m_r) may contain NaN (ragged).
    Returns filtered states xs (T,4), stacked fitted yields (T,m_n+m_r), loglik."""
    Yn = np.asarray(Yn, float); Yr = np.asarray(Yr, float)
    T, m_n = Yn.shape; m_r = Yr.shape[1]
    pre_n, pre_r = _precompute(p, grid_n, grid_r, n_per_year)
    A, Q = _transition(p, dt)
    r_diag = np.concatenate([np.full(m_n, p.h_n ** 2), np.full(m_r, p.h_r ** 2)])
    # init at unconditional moments
    x = p.theta.copy()
    try:
        P = scipy.linalg.solve_discrete_lyapunov(A, Q)
    except Exception:                                             # noqa: BLE001
        P = np.eye(4)
    xs = np.empty((T, 4)); fits = np.empty((T, m_n + m_r)); ll = 0.0
    for t in range(T):
        # predict
        xp = p.theta + A @ (x - p.theta)
        Pp = A @ P @ A.T + Q
        # measure (nonlinear) + mask NaN
        y_row = np.concatenate([Yn[t], Yr[t]])
        y_hat, J = _measure(xp, pre_n, pre_r, m_n)
        fits[t] = y_hat
        mask = np.isfinite(y_row)
        if not mask.any():
            x, P = xp, Pp; xs[t] = x; continue
        Jm = J[mask]; v = y_row[mask] - y_hat[mask]
        S = Jm @ Pp @ Jm.T + np.diag(r_diag[mask])
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return xs, fits, -1e12
        Kg = Pp @ Jm.T @ Sinv
        x = xp + Kg @ v
        P = (np.eye(4) - Kg @ Jm) @ Pp
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (mask.sum() * np.log(2 * np.pi) + logdet + v @ Sinv @ v)
        xs[t] = x
    return xs, fits, float(ll)


def _stationary(p: JointParams, dt: float) -> bool:
    A = scipy.linalg.expm(-p.K * dt)
    return np.max(np.abs(np.linalg.eigvals(A))) < 1.0


def negloglik(v, Yn, Yr, grid_n, grid_r, dt, elb_n, n_per_year=48) -> float:
    p = JointParams.unpack(v, elb_n)
    if not np.all(np.isfinite(v)) or not _stationary(p, dt):
        return 1e12
    try:
        _, _, ll = filter_states(p, Yn, Yr, grid_n, grid_r, dt, n_per_year)
    except Exception:                                             # noqa: BLE001
        return 1e12
    return 1e12 if not np.isfinite(ll) else -ll


def estimate(Yn, Yr, grid_n, grid_r, dt, elb_n=-0.0035, warm: JointParams | None = None,
             maxiter=4000, n_per_year=24):
    """QML estimate of the joint model (Nelder-Mead). Warm-start from `warm` if given."""
    if warm is None:
        warm = _default_init(Yn, Yr)
    v0 = warm.pack()
    res = scipy.optimize.minimize(
        negloglik, v0, args=(Yn, Yr, grid_n, grid_r, dt, elb_n, n_per_year),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "maxfev": maxiter, "xatol": 1e-5, "fatol": 1e-4})
    p = JointParams.unpack(res.x, elb_n)
    return p, res


def _default_init(Yn, Yr) -> JointParams:
    ln = np.nanmean(Yn[:, -1]); lr = np.nanmean(Yr[:, -1])       # long-end level proxies
    return JointParams(kappa=np.array([0.3, 0.3, 0.5, 0.8]),
                       theta=np.array([ln, lr, 0.0, 0.0]),
                       sigma=np.array([0.01, 0.01, 0.01, 0.01]),
                       lambda_=0.5, h_n=0.001, h_r=0.002, elb_n=-0.0035)


def _expected_floored_yields(states, K, theta, Sigma, elb, rho, grid, n_per_year=24):
    """E^P[(1/τ)∫_0^τ max(s_{t+u}, ELB) du] for short rate s=rho'x, dimension-agnostic.

    μ_u = rho'θ + rho'e^{-K u}(x-θ); σ_u² = rho'·cond_cov_P(K,Σ,u)·rho;
    E[max(s,ELB)] = (μ_u-ELB)Φ(d) + σ_u φ(d), d=(μ_u-ELB)/σ_u. (The 3-factor
    `term_premium.expected_floored_short_rate_yields` hardcodes the AFNS λ/convexity,
    so it can't be reused for the joint 4-factor state — hence this generic version.)
    """
    from scipy.linalg import expm
    from scipy.stats import norm
    states = np.asarray(states, float); rho = np.asarray(rho, float)
    theta = np.asarray(theta, float); grid = np.asarray(grid, float)
    dev = states - theta
    tau_max = float(grid.max())
    Ku = max(int(np.ceil(tau_max * n_per_year)) + 1, 8)
    u = np.linspace(0.0, tau_max, Ku)
    floored = np.empty((states.shape[0], Ku))
    for j, uu in enumerate(u):
        g = expm(-K * uu).T @ rho                       # μ_u = rho'θ + dev·g
        mu = rho @ theta + dev @ g
        if uu <= 0:
            var = 0.0
        else:
            var = float(rho @ sr.cond_cov(K, Sigma, uu) @ rho)
        s = np.sqrt(max(var, 0.0))
        if s < 1e-12:
            floored[:, j] = np.maximum(mu, elb)
        else:
            d = (mu - elb) / s
            floored[:, j] = (mu - elb) * norm.cdf(d) + s * norm.pdf(d) + elb
    # (1/τ)∫_0^τ floored du via cumulative trapezoid at each grid maturity
    out = np.empty((states.shape[0], grid.size))
    for i, tau in enumerate(grid):
        mask = u <= tau + 1e-12
        uu = u[mask]; fu = floored[:, mask]
        integ = np.trapz(fu, uu, axis=1)
        out[:, i] = integ / tau
    return out


# --------------------------------------------------------------- decomposition
def decompose(p: JointParams, states, grid_n, grid_r, mr_halflife: float | None = 3.5,
              n_per_year=24) -> dict:
    """Nominal/real fitted yields, expected-rate paths, term premia, IRP and
    expected inflation from filtered `states` (T,4). Applies the mean-reversion
    floor to the P-dynamics for the expectations legs (as the production series)."""
    states = np.asarray(states, float)
    K = p.K
    K_exp = tp.mean_reversion_floor(K, np.log(2) / mr_halflife) if mr_halflife else K
    Sig = np.diag(p.sigma)
    gn = np.asarray(grid_n, float); gr = np.asarray(grid_r, float)
    # fitted yields (Q pricing) per block
    pre_n, pre_r = _precompute(p, gn, gr, n_per_year)
    fit_n = np.array([sr.elb_yields(x[_NOM_SUB], pre_n) for x in states])
    fit_r = np.array([sr.elb_yields(x[_REAL_SUB], pre_r) for x in states])
    # expected average short rate (P): nominal floored at elb_n, real plain affine
    exp_n = _expected_floored_yields(states, K_exp, p.theta, Sig,
                                     p.elb_n, RHO_N, gn, n_per_year)
    exp_r = tp.expected_rate_yields(states, K_exp, p.theta, RHO_R, gr)
    tp_n = fit_n - exp_n
    tp_r = fit_r - exp_r
    # IRP and expected inflation on the common grid (intersection of gn, gr)
    common = np.array([g for g in gr if np.isclose(gn, g).any()])
    idx_n = [int(np.where(np.isclose(gn, g))[0][0]) for g in common]
    idx_r = [int(np.where(np.isclose(gr, g))[0][0]) for g in common]
    irp = tp_n[:, idx_n] - tp_r[:, idx_r]
    exp_infl = exp_n[:, idx_n] - exp_r[:, idx_r]           # E[avg nominal-real short]
    breakeven = fit_n[:, idx_n] - fit_r[:, idx_r]
    return {"grid_n": gn, "grid_r": gr, "common": common,
            "fitted_n": fit_n, "fitted_r": fit_r, "exp_n": exp_n, "exp_r": exp_r,
            "tp_n": tp_n, "tp_r": tp_r, "irp": irp,
            "expected_inflation": exp_infl, "breakeven": breakeven}
