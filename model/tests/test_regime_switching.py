"""Tests for the Markov-switching AFNS (Kim filter) on synthetic ground truth.

Includes the Kim-collapsing correctness battery (Phase-4 step-2 validation): the
Kim filter is compared against an EXACT path-enumeration filter that reuses the
identical per-branch Kalman math (``rs._kalman_branch``) — so the only difference
is Kim's collapse. A collapsing bug (wrong axis/weights, i<->j swap) shows up as a
divergence in loglik or filtered regime probabilities. See
``validation/FINDINGS_kim_validation.md``.
"""
import numpy as np
import pytest
from scipy.special import logsumexp

from model import regime_switching as rs

GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
DT = 1.0 / 12


def _exact_filter(p, yields, grid, dt, n_per_year=24):
    """Exact regime filter by full path enumeration (no collapsing).

    Mirrors kim_filter's conventions: initial regime i has weight log(pi_i) and
    state (theta_i, P0_i); each observation extends every path by a branch (i->j)
    via rs._kalman_branch (the SAME math kim_filter uses). O(S^{T+1}) — short T
    only. Returns (loglik, filtered_prob[T,S]).
    """
    B, a, A, Q, P0 = rs._regime_pieces(p, grid, dt, n_per_year)
    Pmat = p.transition()
    logP = np.log(np.clip(Pmat, 1e-300, None))
    pi = rs._stationary_dist(Pmat)
    S, T, h2 = p.n_regimes, len(yields), p.h ** 2
    active = [(i, p.theta[i].copy(), P0[i].copy(), float(np.log(pi[i]))) for i in range(S)]
    filt = np.zeros((T, S))
    for t in range(T):
        new = []
        for (i, x, P, logw) in active:
            for j in range(S):
                xu, Pu, ll = rs._kalman_branch(x, P, A[j], Q[j], p.theta[j],
                                               B[j], a[j], yields[t], h2)
                new.append((j, xu, Pu, logw + logP[i, j] + ll))
        active = new
        tot = logsumexp(np.array([w for *_, w in active]))
        for (j, _, _, w) in active:
            filt[t, j] += np.exp(w - tot)
    return float(logsumexp(np.array([w for *_, w in active]))), filt


def _true_params():
    # regime 0 = persistent/high-vol (stagflation-like); regime 1 = calm.
    return rs.RSParams(
        kappa=np.array([[0.03, 0.20, 0.40], [0.25, 0.60, 0.90]]),
        theta=np.array([[0.08, -0.02, 0.0], [0.04, -0.01, 0.0]]),
        sigma=np.array([[0.016, 0.020, 0.026], [0.007, 0.009, 0.011]]),
        lambda_=0.5, h=0.0006, p_stay=np.array([0.97, 0.98]),
    )


def _simulate(T=300, seed=0):
    rng = np.random.default_rng(seed)
    p = _true_params()
    B, a, A, Q, P0 = rs._regime_pieces(p, GRID, DT, 24)
    # regime sequence: calm -> persistent -> calm (a stagflation episode in the middle)
    reg = np.array([1] * (T // 3) + [0] * (T // 3) + [1] * (T - 2 * (T // 3)))
    x = p.theta[reg[0]].copy()
    X = np.empty((T, 3)); Y = np.empty((T, GRID.size))
    for t in range(T):
        s = reg[t]
        Lq = np.linalg.cholesky(Q[s])
        x = p.theta[s] + A[s] @ (x - p.theta[s]) + Lq @ rng.standard_normal(3)
        X[t] = x
        Y[t] = B[s] @ x + a[s] + p.h * rng.standard_normal(GRID.size)
    return p, reg, X, Y


def test_pack_unpack_roundtrip():
    p = _true_params()
    q = rs.RSParams.unpack(p.pack(), S=2)
    assert np.allclose(q.kappa, p.kappa) and np.allclose(q.theta, p.theta)
    assert np.allclose(q.sigma, p.sigma)
    assert np.isclose(q.lambda_, p.lambda_) and np.isclose(q.h, p.h)
    assert np.allclose(q.p_stay, p.p_stay, atol=1e-6)


def test_transition_and_stationary():
    p = _true_params()
    P = p.transition()
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0, 0] == pytest.approx(0.97) and P[1, 1] == pytest.approx(0.98)
    pi = rs._stationary_dist(P)
    assert np.allclose(pi @ P, pi, atol=1e-10) and pi.sum() == pytest.approx(1.0)


def test_kim_filter_recovers_regime_sequence():
    p, reg, X, Y = _simulate()
    res = rs.kim_filter(p, Y, GRID, DT, smooth=True)
    sm = res["smoothed_prob"]
    # classify by argmax and compare to the true regime sequence.
    pred = sm.argmax(axis=1)
    accuracy = float(np.mean(pred == reg))
    assert accuracy > 0.9
    assert np.isfinite(res["loglik"])


def test_negloglik_minimized_near_truth():
    p, reg, X, Y = _simulate()
    rng = np.random.default_rng(1)
    nll_true = rs.negloglik(p.pack(), Y, GRID, DT, 2)
    npar = p.pack().size
    perturbed = [rs.negloglik(p.pack() + 0.15 * rng.standard_normal(npar), Y, GRID, DT, 2)
                 for _ in range(5)]
    assert np.isfinite(nll_true)
    assert nll_true < min(perturbed)


def test_ragged_maturities_handled():
    p, reg, X, Y = _simulate()
    Yr = Y.copy()
    Yr[: len(Y) // 4, -1] = np.nan   # early 10y missing
    res = rs.kim_filter(p, Yr, GRID, DT)
    assert np.isfinite(res["loglik"])


def test_regime_conditional_tp_shapes_and_ordering():
    p, reg, X, Y = _simulate()
    res = rs.kim_filter(p, Y, GRID, DT)
    dec = rs.regime_term_premium(p, res["states"], GRID)
    assert dec["term_premium"].shape == (len(Y), 2, GRID.size)
    assert dec["fitted"].shape == dec["expected"].shape == dec["term_premium"].shape
    # The two regimes must imply materially different 10y term premia (the whole
    # point of regime-switching); the sign of the gap depends on the K^P-vs-K^Q
    # interplay and is not a clean invariant, so we only require a real difference.
    tp10 = dec["term_premium"][:, :, -1]
    assert abs(np.nanmean(tp10[:, 0]) - np.nanmean(tp10[:, 1])) > 5e-4  # > 5 bp


def test_order_regimes_sorts_by_persistence():
    p = _true_params()
    ordered = rs.order_regimes(p)
    assert ordered.kappa[0, 0] <= ordered.kappa[1, 0]


# --- Kim-collapsing correctness battery (Phase-4 step-2 validation) --------- #

def test_kim_matches_exact_path_enumeration():
    # Well-separated regimes so the collapse is tight; Kim must match the exact
    # (no-collapse) filter in both loglik and filtered regime probabilities.
    p, reg, X, Y = _simulate(T=12, seed=12)
    ll_ex, filt_ex = _exact_filter(p, Y, GRID, DT)
    km = rs.kim_filter(p, Y, GRID, DT)
    assert abs(ll_ex - km["loglik"]) < 1e-2
    assert np.max(np.abs(filt_ex - km["filtered_prob"])) < 5e-3


def test_kim_degenerate_identical_regimes_is_flat_and_exact():
    # Identical regimes: no information to switch -> filtered probs pinned to the
    # stationary distribution, and Kim == exact.
    p0 = _true_params()
    pid = rs.RSParams(kappa=np.vstack([p0.kappa[0], p0.kappa[0]]),
                      theta=np.vstack([p0.theta[0], p0.theta[0]]),
                      sigma=np.vstack([p0.sigma[0], p0.sigma[0]]),
                      lambda_=p0.lambda_, h=p0.h, p_stay=np.array([0.9, 0.9]))
    _, _, _, Y = _simulate(T=40, seed=5)
    km = rs.kim_filter(pid, Y, GRID, DT)
    pi = rs._stationary_dist(pid.transition())
    assert np.max(np.abs(km["filtered_prob"] - pi[None, :])) < 1e-10
    ll_ex, _ = _exact_filter(pid, Y[:10], GRID, DT)
    assert abs(ll_ex - rs.kim_filter(pid, Y[:10], GRID, DT)["loglik"]) < 1e-6


def test_kim_loglik_relabel_invariant():
    p, reg, X, Y = _simulate(T=40, seed=7)
    p_swap = rs.RSParams(kappa=p.kappa[::-1], theta=p.theta[::-1], sigma=p.sigma[::-1],
                         lambda_=p.lambda_, h=p.h, p_stay=p.p_stay[::-1])
    a = rs.kim_filter(p, Y, GRID, DT)["loglik"]
    b = rs.kim_filter(p_swap, Y, GRID, DT)["loglik"]
    assert abs(a - b) < 1e-8
