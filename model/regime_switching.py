"""Markov-switching AFNS for regime-dependent term premia (Kim 1994 filter).

Motivation. The term premium in the 1970s stagflation is exactly the case where a
single stationary model is least reliable — the highly persistent, high-volatility
inflation regime is averaged together with the calm post-1990 sample. This module
lets the **physical dynamics switch** across a small number of latent Markov
regimes, so the high-inflation regime gets its own mean-reversion, mean, and
volatility — and therefore its own term-premium level.

Specification (regime-switching AFNS, common-Q loadings)
--------------------------------------------------------
State ``x=(L,S,C)``. Regimes ``s=0..S-1`` follow a Markov chain with transition
matrix ``P`` (``P[i,j]=Pr(S_t=j | S_{t-1}=i)``). Per regime:

    x_t = θ_s + A_s (x_{t-1}-θ_s) + η,  A_s=e^{-K^P_s Δ},  η~N(0,Q_s),
    Q_s = ∫_0^Δ e^{-K^P_s u} Σ_s Σ_s' e^{-K^P_s' u} du     (P-dynamics switch)

Measurement is the **plain AFNS** affine map (linear — no ELB; historical rates
are high), sharing the NS decay ``λ`` across regimes:

    y_t = B x_t + a_s + ε,   ε~N(0, h²I)

where ``B`` (yield loadings, common because ``λ`` is common) and the tiny yield
convexity ``a_s`` come from ``model.shadow_rate.precompute`` with the bound pushed
away. Because the arbitrage-free cross-section (``B``) is common, term premia are
comparable across regimes; only the expectations leg (via ``K^P_s, θ_s``) and the
volatility (``Σ_s``) switch. This treats regime-switching risk as unpriced — a
deliberate tractability/interpretability choice; the priced-regime extension is
Dai-Singleton-Yang (2007).

Estimation is exact-likelihood QML via the Kim (1994) filter (Hamilton regime
filter + Kalman filter + Kim collapsing), in log-space with masking for ragged
early maturities. Base case ``S=2`` (calm / high-inflation); ``S=3`` supported
with an equal-split off-diagonal transition parameterisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.special import expit, logit, logsumexp

from model import ekf
from model import shadow_rate as sr
from model import term_premium as tp

_FLOOR = 1e-8
# Bound pushed to -100%: the Krippner floor never binds, so the AFNS measurement
# is exactly linear (plain AFNS). Historical (1961+) short rates are far above it.
PLAIN_ELB = -1.0


@dataclass
class RSParams:
    kappa: np.ndarray     # (S, 3) diagonal K^P mean-reversion per regime
    theta: np.ndarray     # (S, 3) P long-run mean per regime
    sigma: np.ndarray     # (S, 3) diagonal Σ per regime
    lambda_: float        # common NS decay
    h: float              # common measurement-error SD
    p_stay: np.ndarray    # (S,) diagonal stay-probabilities of P

    @property
    def n_regimes(self) -> int:
        return self.kappa.shape[0]

    def transition(self) -> np.ndarray:
        """Transition matrix P[i,j]=Pr(S_t=j|S_{t-1}=i); off-diagonals split equally."""
        S = self.n_regimes
        P = np.empty((S, S))
        for i in range(S):
            off = (1.0 - self.p_stay[i]) / (S - 1) if S > 1 else 0.0
            P[i, :] = off
            P[i, i] = self.p_stay[i]
        return P

    def pack(self) -> np.ndarray:
        return np.concatenate([
            np.log(np.maximum(self.kappa.ravel(), _FLOOR)),    # 3S
            self.theta.ravel(),                                # 3S
            np.log(np.maximum(self.sigma.ravel(), _FLOOR)),    # 3S
            [np.log(max(self.lambda_, _FLOOR)), np.log(max(self.h, _FLOOR))],
            logit(np.clip(self.p_stay, 1e-4, 1 - 1e-4)),       # S
        ])

    @staticmethod
    def unpack(v: np.ndarray, S: int) -> "RSParams":
        v = np.asarray(v, dtype=float)
        i = 0
        kappa = np.exp(v[i:i + 3 * S]).reshape(S, 3); i += 3 * S
        theta = v[i:i + 3 * S].reshape(S, 3).copy(); i += 3 * S
        sigma = np.exp(v[i:i + 3 * S]).reshape(S, 3); i += 3 * S
        lambda_ = float(np.exp(v[i])); h = float(np.exp(v[i + 1])); i += 2
        p_stay = expit(v[i:i + S])
        return RSParams(kappa=kappa, theta=theta, sigma=sigma,
                        lambda_=lambda_, h=h, p_stay=p_stay)


def _stationary_dist(P: np.ndarray) -> np.ndarray:
    """Stationary regime distribution (left eigenvector of P for eigenvalue 1)."""
    S = P.shape[0]
    w, V = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(w - 1.0)))
    pi = np.real(V[:, k])
    pi = np.abs(pi)
    tot = pi.sum()
    return pi / tot if tot > 0 else np.full(S, 1.0 / S)


def _regime_pieces(p: RSParams, grid: np.ndarray, dt: float, n_per_year: int):
    """Per-regime measurement (B, a) and transition (A, Q, uncond cov)."""
    S = p.n_regimes
    B, a, A, Q, P0 = [], [], [], [], []
    for s in range(S):
        Sig = np.diag(p.sigma[s])
        pre = sr.precompute(p.lambda_, Sig, PLAIN_ELB, grid, n_per_year=n_per_year)
        B.append(pre.weights @ pre.b_u)         # (m,3) yield loadings (common up to λ)
        a.append(pre.weights @ pre.a_u)         # (m,) tiny yield convexity
        K = np.diag(p.kappa[s])
        A_s = expm(-K * dt)
        Q_s = sr.cond_cov(K, Sig, dt)
        A.append(A_s); Q.append(Q_s); P0.append(ekf._uncond_cov(A_s, Q_s))
    return B, a, A, Q, P0


def _kalman_branch(x_i, P_i, A_j, Q_j, theta_j, B_j, a_j, y_row, h2):
    """Predict from regime i's posterior, switch to regime j's dynamics, update.

    Returns (x_upd, P_upd, loglik) with per-period NaN masking; loglik is -inf
    (as np.nan sentinel handled by caller) on a Cholesky failure.
    """
    xp = theta_j + A_j @ (x_i - theta_j)
    Pp = A_j @ P_i @ A_j.T + Q_j
    y_hat = B_j @ xp + a_j
    mask = np.isfinite(y_row)
    mt = int(mask.sum())
    if mt == 0:
        return xp, Pp, 0.0
    Bm = B_j[mask]; yh = y_hat[mask]; yr = y_row[mask]
    v = yr - yh
    Smat = Bm @ Pp @ Bm.T + h2 * np.eye(mt)
    try:
        L = np.linalg.cholesky(Smat)
    except np.linalg.LinAlgError:
        return xp, Pp, -np.inf
    alpha = np.linalg.solve(L, v)
    ll = -0.5 * (mt * np.log(2.0 * np.pi) + 2.0 * np.sum(np.log(np.diag(L))) + alpha @ alpha)
    Kg = Pp @ Bm.T @ np.linalg.inv(Smat)
    return xp + Kg @ v, Pp - Kg @ Bm @ Pp, ll


def kim_filter(p: RSParams, yields, grid, dt, n_per_year: int = 24, smooth: bool = False):
    """Kim (1994) filter. Returns a dict with loglik and (optionally) smoothed
    regime probabilities and regime-conditional filtered states.
    """
    yields = np.asarray(yields, dtype=float)
    grid = np.asarray(grid, dtype=float)
    S = p.n_regimes
    T = yields.shape[0]
    h2 = p.h ** 2
    B, a, A, Q, P0 = _regime_pieces(p, grid, dt, n_per_year)
    Pmat = p.transition()
    logP = np.log(np.clip(Pmat, 1e-300, None))

    prob = _stationary_dist(Pmat)                 # P(S_0 | Y_0)
    x = [p.theta[s].copy() for s in range(S)]     # regime-conditional state
    Pc = [P0[s].copy() for s in range(S)]

    ll = 0.0
    filt_prob = np.empty((T, S))
    pred_prob = np.empty((T, S))                  # P(S_t=j | Y_{t-1}) for smoother
    x_hist = np.empty((T, S, 3))
    P_hist = np.empty((T, S, 3, 3))

    for t in range(T):
        log_joint = np.full((S, S), -np.inf)      # (i, j)
        x_ij = np.empty((S, S, 3))
        P_ij = np.empty((S, S, 3, 3))
        with np.errstate(divide="ignore"):
            log_prev = np.log(np.clip(prob, 1e-300, None))
        for i in range(S):
            for j in range(S):
                xu, Pu, llij = _kalman_branch(x[i], Pc[i], A[j], Q[j], p.theta[j],
                                              B[j], a[j], yields[t], h2)
                x_ij[i, j] = xu; P_ij[i, j] = Pu
                log_joint[i, j] = log_prev[i] + logP[i, j] + llij
        log_tot = logsumexp(log_joint)
        ll += log_tot
        w = np.exp(log_joint - log_tot)           # P(S_{t-1}=i, S_t=j | Y_t)
        marg = w.sum(axis=0)                       # P(S_t=j | Y_t)
        pred_prob[t] = (prob @ Pmat)              # P(S_t=j | Y_{t-1})

        # Kim collapsing over the previous regime i.
        new_x = np.zeros((S, 3)); new_P = np.zeros((S, 3, 3))
        for j in range(S):
            if marg[j] <= 0:
                new_x[j] = x_ij[:, j].mean(axis=0)
                new_P[j] = P_ij[:, j].mean(axis=0)
                continue
            wj = w[:, j] / marg[j]
            xj = (wj[:, None] * x_ij[:, j]).sum(axis=0)
            Pj = np.zeros((3, 3))
            for i in range(S):
                d = xj - x_ij[i, j]
                Pj += wj[i] * (P_ij[i, j] + np.outer(d, d))
            new_x[j] = xj; new_P[j] = Pj
        x, Pc = [new_x[s] for s in range(S)], [new_P[s] for s in range(S)]
        prob = marg
        filt_prob[t] = marg
        x_hist[t] = new_x
        P_hist[t] = new_P

    out = {"loglik": float(ll), "filtered_prob": filt_prob, "states": x_hist,
           "pred_prob": pred_prob, "transition": Pmat}

    if smooth:
        sm = np.empty((T, S))
        sm[-1] = filt_prob[-1]
        for t in range(T - 2, -1, -1):
            for j in range(S):
                s = 0.0
                for k in range(S):
                    denom = pred_prob[t + 1, k]
                    if denom > 0:
                        s += sm[t + 1, k] * filt_prob[t, j] * Pmat[j, k] / denom
                sm[t, j] = s
            tot = sm[t].sum()
            if tot > 0:
                sm[t] /= tot
        out["smoothed_prob"] = sm
    return out


def negloglik(packed, yields, grid, dt, S, n_per_year: int = 24) -> float:
    p = RSParams.unpack(packed, S)
    if np.any(p.sigma <= 0) or np.any(p.kappa <= 0) or p.lambda_ <= 0:
        return 1e12
    try:
        A = [expm(-np.diag(p.kappa[s]) * dt) for s in range(S)]
        for A_s in A:
            if np.max(np.abs(np.linalg.eigvals(A_s))) >= 1.0 - 1e-9:
                return 1e12
        res = kim_filter(p, yields, grid, dt, n_per_year=n_per_year)
    except Exception:
        return 1e12
    ll = res["loglik"]
    return -ll if np.isfinite(ll) else 1e12


def regime_term_premium(p: RSParams, states_by_regime: np.ndarray, grid: np.ndarray,
                        n_per_year: int = 24):
    """Regime-conditional fitted / expected / term-premium yields.

    ``states_by_regime`` is (T, S, 3) (e.g. ``kim_filter(...)['states']``).
    Returns dict of (T, S, m) arrays: ``fitted``, ``expected``, ``term_premium``
    — the decomposition *as if* the economy were in each regime at that date.
    """
    grid = np.asarray(grid, dtype=float)
    T, S = states_by_regime.shape[0], p.n_regimes
    m = grid.size
    fitted = np.empty((T, S, m)); expected = np.empty((T, S, m))
    for s in range(S):
        pre = sr.precompute(p.lambda_, np.diag(p.sigma[s]), PLAIN_ELB, grid,
                            n_per_year=n_per_year)
        B = pre.weights @ pre.b_u; a = pre.weights @ pre.a_u
        xs = states_by_regime[:, s, :]                          # (T,3)
        fitted[:, s, :] = xs @ B.T + a
        expected[:, s, :] = tp.expected_rate_yields(
            xs, np.diag(p.kappa[s]), p.theta[s], sr.RHO, grid)
    return {"fitted": fitted, "expected": expected,
            "term_premium": fitted - expected}


def default_init(yields, grid, S: int = 2) -> RSParams:
    """Data-driven start: regimes share a level/slope guess but differ in
    persistence & vol (a calm regime and a persistent, high-vol regime)."""
    y = np.asarray(yields, dtype=float)
    level0 = float(np.nanmean(y[:, -1]))
    slope0 = float(np.nanmean(y[:, 0] - y[:, -1]))
    # regime 0 = calm (faster reversion, lower vol); regime 1 = high-inflation.
    kappa = np.array([[0.20, 0.50, 0.80], [0.03, 0.20, 0.40]][:S])
    sigma = np.array([[0.008, 0.010, 0.012], [0.015, 0.020, 0.025]][:S])
    theta = np.tile([level0, slope0, 0.0], (S, 1))
    p_stay = np.full(S, 0.97)
    return RSParams(kappa=kappa, theta=theta, sigma=sigma,
                    lambda_=0.5, h=0.001, p_stay=p_stay)


def estimate(yields, grid, dt, S: int = 2, warm_start: "RSParams | None" = None,
             maxiter: int = 8000, n_per_year: int = 24, restarts: int = 2, seed: int = 0):
    """QML/Kim-filter estimation via Nelder-Mead with restarts.

    Multi-start guards against the multi-modal regime likelihood; label-switching
    is resolved post-hoc by ordering regimes by persistence (see ``order_regimes``).
    """
    from scipy.optimize import minimize

    init = warm_start if warm_start is not None else default_init(yields, grid, S)
    x0 = init.pack()
    obj = lambda v: negloglik(v, yields, grid, dt, S, n_per_year)
    rng = np.random.default_rng(seed)
    res = None
    for r in range(max(restarts, 1)):
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-6, "fatol": 1e-4,
                                "adaptive": True})
        if res.success:
            break
        x0 = res.x + 0.05 * rng.standard_normal(res.x.size)
    return order_regimes(RSParams.unpack(res.x, S)), res, -float(res.fun)


def order_regimes(p: RSParams) -> RSParams:
    """Canonical label ordering: regime 0 = most persistent (smallest mean
    level-factor reversion), so labels are comparable across fits/multi-starts."""
    order = np.argsort(p.kappa[:, 0])   # ascending κ_L: slowest-reverting first
    return RSParams(kappa=p.kappa[order], theta=p.theta[order], sigma=p.sigma[order],
                    lambda_=p.lambda_, h=p.h, p_stay=p.p_stay[order])
