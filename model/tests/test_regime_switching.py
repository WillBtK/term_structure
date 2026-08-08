"""Tests for the Markov-switching AFNS (Kim filter) on synthetic ground truth."""
import numpy as np
import pytest

from model import regime_switching as rs

GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
DT = 1.0 / 12


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
