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
_NOM_SUB = [0, 2, 3]                         # (L_N, S, C) columns of the state
_REAL_SUB = [1, 2, 3]                        # (L_R, S, C)
_REAL_ELB = -1.0                             # real curve: plain AFNS (no floor)
LIQ_GAMMA = 1.0                             # fixed decay of the TIPS-liquidity loading

# 5th factor (optional) = TIPS liquidity ℓ. It loads on the REAL (TIPS) measurement
# ONLY, with a maturity-decaying shape β(τ)=(1-e^{-γτ})/(γτ) — distinct from the flat
# real level L_R (so it is identified) — capturing the illiquidity that makes observed
# TIPS yields sit above the model real yield. It does NOT enter the pricing kernel or
# the expectations legs, so the "clean" real yield/TP excludes it (a reduced-form,
# measurement-side liquidity wedge; DKW's is priced — a documented approximation).


def _rho_n(n):  # nominal short = L_N + S
    r = np.zeros(n); r[0] = 1.0; r[2] = 1.0; return r


def _rho_r(n):  # real short = L_R + S
    r = np.zeros(n); r[1] = 1.0; r[2] = 1.0; return r


def _liq_loading(grid, gamma=LIQ_GAMMA):
    tau = np.asarray(grid, float)
    return (1.0 - np.exp(-gamma * tau)) / (gamma * tau)


# Backwards-compatible 4-factor rho constants (used where n is known to be 4).
RHO_N = _rho_n(4)
RHO_R = _rho_r(4)


@dataclass
class JointParams:
    kappa: np.ndarray      # (n,) diagonal P mean-reversion, >0  [L_N,L_R,S,C,(ℓ)]
    theta: np.ndarray      # (n,) P long-run mean
    sigma: np.ndarray      # (n,) factor vols, >0 (diagonal Σ)
    lambda_: float         # shared NS decay, >0
    h_n: float             # nominal measurement-error SD, >0
    h_r: float             # real measurement-error SD, >0
    elb_n: float = -0.0035  # nominal effective lower bound (fixed)
    h_s: float = 0.005     # survey (ATSIX) measurement-error SD, >0 (used iff anchored)

    @property
    def n(self) -> int:
        return len(self.kappa)

    @property
    def liq(self) -> bool:
        return self.n == 5     # 5th factor is the TIPS liquidity ℓ

    def pack(self, survey: bool = False) -> np.ndarray:
        base = np.concatenate([
            np.log(np.maximum(self.kappa, _LOG_FLOOR)),           # n
            np.asarray(self.theta, float),                        # n
            np.log(np.maximum(self.sigma, _LOG_FLOOR)),           # n
            [np.log(max(self.lambda_, _LOG_FLOOR)),
             np.log(max(self.h_n, _LOG_FLOOR)),
             np.log(max(self.h_r, _LOG_FLOOR))],                  # 3
        ])
        if survey:                                                # +1 survey error
            base = np.concatenate([base, [np.log(max(self.h_s, _LOG_FLOOR))]])
        return base

    @staticmethod
    def unpack(v: np.ndarray, elb_n: float, n: int = 4,
               survey: bool = False) -> "JointParams":
        v = np.asarray(v, dtype=float)
        h_s = float(np.exp(v[3 * n + 3])) if survey else 0.005
        return JointParams(
            kappa=np.exp(v[0:n]), theta=v[n:2 * n].copy(), sigma=np.exp(v[2 * n:3 * n]),
            lambda_=float(np.exp(v[3 * n])), h_n=float(np.exp(v[3 * n + 1])),
            h_r=float(np.exp(v[3 * n + 2])), elb_n=elb_n, h_s=h_s)

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


def _measure(x, pre_n, pre_r, m_n, beta_r=None):
    """Stacked [nominal; real] yields and Jacobian wrt the n-factor state. When
    ``beta_r`` (real liquidity loading, length m_r) is given, the real block adds
    β(τ)·ℓ with ℓ = x[4] and a 5th Jacobian column."""
    n = x.shape[0]
    yn, Jn = sr.elb_yields_and_jac(x[_NOM_SUB], pre_n)
    yr, Jr = sr.elb_yields_and_jac(x[_REAL_SUB], pre_r)
    if beta_r is not None:
        yr = yr + beta_r * x[4]
    y = np.concatenate([yn, yr])
    J = np.zeros((y.shape[0], n))
    J[:m_n][:, _NOM_SUB] = Jn
    J[m_n:][:, _REAL_SUB] = Jr
    if beta_r is not None:
        J[m_n:, 4] = beta_r
    return y, J


def _transition(p: JointParams, dt: float):
    K = p.K
    A = scipy.linalg.expm(-K * dt)
    Q = sr.cond_cov(K, np.diag(p.sigma), dt)
    return A, Q


# --------------------------------------------------------------------- filter
def filter_states(p: JointParams, Yn, Yr, grid_n, grid_r, dt, n_per_year=48,
                  Ysurv=None, grid_surv=None, mr_halflife: float | None = 3.5):
    """EKF over the joint panel. Yn (T,m_n), Yr (T,m_r) may contain NaN (ragged).
    Returns filtered states xs (T,n), stacked fitted yields (T,m_n+m_r), loglik.

    ``Ysurv`` (T,m_s) optionally supplies survey (ATSIX) expected-inflation on
    ``grid_surv`` (years); those rows are appended to the measurement (with error SD
    ``p.h_s``) so the P-expected inflation *level* is anchored — identifying the IRP
    that yields-only cannot. Survey rows are NaN-masked like the yields (ragged OK)."""
    Yn = np.asarray(Yn, float); Yr = np.asarray(Yr, float)
    T, m_n = Yn.shape; m_r = Yr.shape[1]
    n = p.n
    use_surv = Ysurv is not None
    beta_r = _liq_loading(grid_r) if p.liq else None
    pre_n, pre_r = _precompute(p, grid_n, grid_r, n_per_year)
    A, Q = _transition(p, dt)
    r_diag = np.concatenate([np.full(m_n, p.h_n ** 2), np.full(m_r, p.h_r ** 2)])
    if use_surv:
        Ysurv = np.asarray(Ysurv, float)
        pre_s = _survey_precompute(p, grid_surv, mr_halflife)
        m_s = Ysurv.shape[1]
        r_diag = np.concatenate([r_diag, np.full(m_s, p.h_s ** 2)])
    # init at unconditional moments
    x = p.theta.copy()
    try:
        P = scipy.linalg.solve_discrete_lyapunov(A, Q)
    except Exception:                                             # noqa: BLE001
        P = np.eye(n)
    xs = np.empty((T, n)); fits = np.empty((T, m_n + m_r)); ll = 0.0
    for t in range(T):
        # predict
        xp = p.theta + A @ (x - p.theta)
        Pp = A @ P @ A.T + Q
        # measure (nonlinear) + mask NaN
        y_hat, J = _measure(xp, pre_n, pre_r, m_n, beta_r)
        fits[t] = y_hat                                    # yields only (survey excluded)
        if use_surv:
            ys, Js = _survey_measure(xp, pre_s)
            y_hat = np.concatenate([y_hat, ys]); J = np.vstack([J, Js])
            y_row = np.concatenate([Yn[t], Yr[t], Ysurv[t]])
        else:
            y_row = np.concatenate([Yn[t], Yr[t]])
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
        P = (np.eye(n) - Kg @ Jm) @ Pp
        sign, logdet = np.linalg.slogdet(S)
        ll += -0.5 * (mask.sum() * np.log(2 * np.pi) + logdet + v @ Sinv @ v)
        xs[t] = x
    return xs, fits, float(ll)


def _stationary(p: JointParams, dt: float) -> bool:
    A = scipy.linalg.expm(-p.K * dt)
    return np.max(np.abs(np.linalg.eigvals(A))) < 1.0


def negloglik(v, Yn, Yr, grid_n, grid_r, dt, elb_n, n_per_year=48,
              Ysurv=None, grid_surv=None, mr_halflife=3.5, survey=False) -> float:
    extra = 4 if survey else 3
    n = (len(v) - extra) // 3
    p = JointParams.unpack(v, elb_n, n, survey=survey)
    if not np.all(np.isfinite(v)) or not _stationary(p, dt):
        return 1e12
    try:
        _, _, ll = filter_states(p, Yn, Yr, grid_n, grid_r, dt, n_per_year,
                                 Ysurv=Ysurv, grid_surv=grid_surv, mr_halflife=mr_halflife)
    except Exception:                                             # noqa: BLE001
        return 1e12
    return 1e12 if not np.isfinite(ll) else -ll


def estimate(Yn, Yr, grid_n, grid_r, dt, elb_n=-0.0035, warm: JointParams | None = None,
             maxiter=4000, n_per_year=24, liq=False,
             Ysurv=None, grid_surv=None, mr_halflife=3.5):
    """QML estimate of the joint model (Nelder-Mead). ``liq=True`` adds the TIPS
    liquidity factor (5-factor). ``Ysurv`` (with ``grid_surv``) anchors the model's
    expected-inflation term structure to survey (ATSIX) observations, identifying the
    IRP level. Warm-start from `warm` if given."""
    survey = Ysurv is not None
    if warm is None:
        warm = _default_init(Yn, Yr, liq=liq)
    v0 = warm.pack(survey=survey)
    n = warm.n
    res = scipy.optimize.minimize(
        negloglik, v0,
        args=(Yn, Yr, grid_n, grid_r, dt, elb_n, n_per_year,
              Ysurv, grid_surv, mr_halflife, survey),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "maxfev": maxiter, "xatol": 1e-5, "fatol": 1e-4})
    p = JointParams.unpack(res.x, elb_n, n, survey=survey)
    return p, res


def _default_init(Yn, Yr, liq=False) -> JointParams:
    ln = np.nanmean(Yn[:, -1]); lr = np.nanmean(Yr[:, -1])       # long-end level proxies
    if liq:
        return JointParams(kappa=np.array([0.3, 0.3, 0.5, 0.8, 0.6]),
                           theta=np.array([ln, lr, 0.0, 0.0, 0.0]),
                           sigma=np.array([0.01, 0.01, 0.01, 0.01, 0.005]),
                           lambda_=0.5, h_n=0.001, h_r=0.002, elb_n=-0.0035)
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


# ------------------------------------------------------------ survey (ATSIX) anchor
def _survey_precompute(p: JointParams, grid_common, mr_halflife: float | None = 3.5,
                       n_per_year=24) -> dict:
    """Precompute the u-grid loadings mapping a state to the model's expected-inflation
    term structure on ``grid_common`` (the survey/ATSIX horizons, in years), so the
    per-date survey measurement + its analytic Jacobian are cheap inside the EKF.

    Model expected inflation over [t,t+τ] = E^P[avg floored nominal short] −
    E^P[avg real short] — the SAME object `decompose` reports, using the same
    mean-reversion-floored K_exp. The nominal leg is floored (Krippner option, so a
    smooth nonlinearity in the state); the real leg is plain affine.
    """
    n = p.n
    K_exp = tp.mean_reversion_floor(p.K, np.log(2) / mr_halflife) if mr_halflife else p.K
    Sig = np.diag(p.sigma)
    rho_n, rho_r = _rho_n(n), _rho_r(n)
    grid = np.asarray(grid_common, float)
    tau_max = float(grid.max())
    Ku = max(int(np.ceil(tau_max * n_per_year)) + 1, 8)
    u = np.linspace(0.0, tau_max, Ku)
    du = u[1] - u[0]
    SS = Sig @ Sig.T
    # Accumulate e^{-K_exp u} along the uniform grid (E_{j} = e^{-K_exp du} · E_{j-1})
    # instead of an expm per point, and cumulative-trapezoid the covariance integrand
    # V(u)=∫_0^u e^{-K s}SSe^{-K's}ds — turning ~Ku·(2·80+1) expm calls into ~Ku matmuls
    # (this precompute runs inside the estimation loop, so speed matters). The reported
    # series still use the accurate per-point `_expected_floored_yields` in decompose.
    E_step = scipy.linalg.expm(-K_exp * du)
    gn = np.empty((Ku, n)); gr = np.empty((Ku, n)); sn = np.empty(Ku)
    E = np.eye(n)
    V = np.zeros((n, n)); prev_M = SS.copy()                  # integrand at u=0 is SS
    for j in range(Ku):
        gn[j] = E.T @ rho_n                       # ∂μ^N_u/∂x = e^{-K_exp u}'ρ_N
        gr[j] = E.T @ rho_r
        M = E @ SS @ E.T                          # integrand e^{-Ku}SSe^{-K'u}
        if j > 0:
            V = V + 0.5 * du * (prev_M + M)       # cumulative trapezoid
        prev_M = M
        sn[j] = np.sqrt(max(float(rho_n @ V @ rho_n), 0.0))
        E = E_step @ E
    masks = [u <= tau + 1e-12 for tau in grid]
    return {"u": u, "gn": gn, "gr": gr, "sn": sn, "grid": grid, "masks": masks,
            "rho_n": rho_n, "rho_r": rho_r, "theta": np.asarray(p.theta, float),
            "elb_n": p.elb_n, "n": n}


def _survey_measure(x, pre_s: dict):
    """Model expected-inflation term structure at state ``x`` on the survey grid, and
    its analytic Jacobian wrt the n-factor state. For the floored nominal leg,
    d E[max(s,ELB)]/dμ = Φ(d) exactly, so the Jacobian is closed-form given the
    precomputed loadings."""
    from scipy.stats import norm
    u = pre_s["u"]; gn = pre_s["gn"]; gr = pre_s["gr"]; sn = pre_s["sn"]
    theta = pre_s["theta"]; elb = pre_s["elb_n"]; n = pre_s["n"]
    dev = x - theta
    mu_n = pre_s["rho_n"] @ theta + gn @ dev
    mu_r = pre_s["rho_r"] @ theta + gr @ dev
    small = sn < 1e-12
    sd = np.where(small, 1.0, sn)
    d = np.where(small, 0.0, (mu_n - elb) / sd)
    floored = np.where(small, np.maximum(mu_n, elb),
                       (mu_n - elb) * norm.cdf(d) + sn * norm.pdf(d) + elb)
    phi_n = np.where(small, (mu_n >= elb).astype(float), norm.cdf(d))  # d floored/dμ_n
    grid = pre_s["grid"]; masks = pre_s["masks"]
    pred = np.empty(len(grid)); J = np.empty((len(grid), n))
    for i, (tau, m) in enumerate(zip(grid, masks)):
        uu = u[m]
        pred[i] = (np.trapz(floored[m], uu) - np.trapz(mu_r[m], uu)) / tau
        J[i] = (np.trapz(phi_n[m][:, None] * gn[m], uu, axis=0)
                - np.trapz(gr[m], uu, axis=0)) / tau
    return pred, J


# --------------------------------------------------------------- decomposition
def decompose(p: JointParams, states, grid_n, grid_r, mr_halflife: float | None = 3.5,
              n_per_year=24) -> dict:
    """Nominal/real fitted yields, expected-rate paths, term premia, IRP and
    expected inflation from filtered `states` (T,4). Applies the mean-reversion
    floor to the P-dynamics for the expectations legs (as the production series)."""
    states = np.asarray(states, float)
    nfac = p.n
    K = p.K
    K_exp = tp.mean_reversion_floor(K, np.log(2) / mr_halflife) if mr_halflife else K
    Sig = np.diag(p.sigma)
    gn = np.asarray(grid_n, float); gr = np.asarray(grid_r, float)
    rho_n, rho_r = _rho_n(nfac), _rho_r(nfac)
    # fitted yields (Q pricing) per block. The real block is the CLEAN model real
    # yield (AFNS on L_R,S,C) — the liquidity wedge is added separately below.
    pre_n, pre_r = _precompute(p, gn, gr, n_per_year)
    fit_n = np.array([sr.elb_yields(x[_NOM_SUB], pre_n) for x in states])
    fit_r = np.array([sr.elb_yields(x[_REAL_SUB], pre_r) for x in states])   # clean
    # expected average short rate (P): nominal floored at elb_n, real plain affine
    exp_n = _expected_floored_yields(states, K_exp, p.theta, Sig,
                                     p.elb_n, rho_n, gn, n_per_year)
    exp_r = tp.expected_rate_yields(states, K_exp, p.theta, rho_r, gr)
    tp_n = fit_n - exp_n
    tp_r = fit_r - exp_r                                   # CLEAN real term premium
    # TIPS liquidity premium (observed real yield = clean + liquidity), if modelled
    if p.liq:
        beta_r = _liq_loading(gr)
        liq_prem = states[:, 4][:, None] * beta_r[None, :]   # (T, m_r)
    else:
        liq_prem = np.zeros_like(fit_r)
    obs_fit_r = fit_r + liq_prem
    # IRP and expected inflation on the common grid (intersection of gn, gr)
    common = np.array([g for g in gr if np.isclose(gn, g).any()])
    idx_n = [int(np.where(np.isclose(gn, g))[0][0]) for g in common]
    idx_r = [int(np.where(np.isclose(gr, g))[0][0]) for g in common]
    irp = tp_n[:, idx_n] - tp_r[:, idx_r]
    exp_infl = exp_n[:, idx_n] - exp_r[:, idx_r]           # E[avg nominal-real short]
    breakeven = fit_n[:, idx_n] - obs_fit_r[:, idx_r]     # observed inflation compensation
    return {"grid_n": gn, "grid_r": gr, "common": common,
            "fitted_n": fit_n, "fitted_r": fit_r, "obs_fitted_r": obs_fit_r,
            "exp_n": exp_n, "exp_r": exp_r, "tp_n": tp_n, "tp_r": tp_r, "irp": irp,
            "liquidity_premium": liq_prem[:, idx_r], "expected_inflation": exp_infl,
            "breakeven": breakeven}
