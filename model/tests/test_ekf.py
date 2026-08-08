"""Tests for the shadow-rate EKF (synthetic ground truth)."""
import numpy as np
import pytest

from model import shadow_rate as sr
from model import ekf

DT = 1.0 / 12
ELB = -0.0035
GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def _true_params():
    return ekf.ShadowParams.from_diagonal(
        kappa=[0.06, 0.30, 0.60],
        theta=[0.04, -0.015, 0.0],
        sigma=[0.008, 0.010, 0.012],
        lambda_=0.5, h=0.0006,
    )


def _simulate(T=360, seed=0):
    rng = np.random.default_rng(seed)
    p = _true_params()
    pre = sr.precompute(p.lambda_, np.diag(p.sigma), ELB, GRID)
    A, Q = ekf._transition(p.K, p.sigma, DT)
    Lq = np.linalg.cholesky(Q)
    X = np.empty((T, 3)); x = p.theta.copy()
    for t in range(T):
        x = p.theta + A @ (x - p.theta) + Lq @ rng.standard_normal(3)
        X[t] = x
    # ZLB episode: drive the slope down so the shadow short rate goes negative
    X[T // 3: 2 * T // 3, 1] += np.linspace(0, -0.05, 2 * T // 3 - T // 3)
    Y = np.array([sr.elb_yields(X[t], pre) for t in range(T)]) + p.h * rng.standard_normal((T, GRID.size))
    return p, X, Y


def test_transition_diagonal_case():
    kappa = np.array([0.05, 0.3, 0.6]); sigma = np.array([0.008, 0.01, 0.012])
    A, Q = ekf._transition(np.diag(kappa), sigma, DT)
    # For diagonal K, A = diag(e^{-k dt}).
    assert np.allclose(A, np.diag(np.exp(-kappa * DT)), atol=1e-10)
    # Q matches the diagonal OU closed form q_i = s_i^2 (1-e^{-2k dt})/(2k).
    q = sigma**2 * (1 - np.exp(-2 * kappa * DT)) / (2 * kappa)
    assert np.allclose(np.diag(Q), q, atol=1e-9)


def test_diag_pack_roundtrip_and_estimate_keeps_K_diagonal():
    # The reduced 11-vector (independent-factor spec) round-trips, and estimate()
    # with diagonal_K=True (the production default) yields a diagonal K^P.
    p = _true_params()
    v = ekf._diag_pack(p)
    assert v.size == 11
    q = ekf._diag_unpack(v)
    assert np.allclose(np.diag(q.K), np.diag(p.K))
    assert np.allclose(q.K, np.diag(np.diag(q.K)))          # off-diagonal exactly 0
    assert np.allclose(q.theta, p.theta) and np.allclose(q.sigma, p.sigma)
    assert np.isclose(q.lambda_, p.lambda_) and np.isclose(q.h, p.h)
    # A short estimate() run stays in the diagonal family.
    _, X, Y = _simulate(T=120)
    est, res, ll = ekf.estimate(Y, GRID, ELB, DT, warm_start=p, maxiter=50,
                                n_per_year=12, restarts=1, diagonal_K=True)
    assert np.allclose(est.K, np.diag(np.diag(est.K)))       # remains diagonal
    assert np.isfinite(ll)


def test_full_K_transition_and_uncond_cov():
    # Non-diagonal but stable K^P: transition + Lyapunov P0 are well-formed.
    K = np.array([[0.10, 0.05, 0.00], [0.00, 0.30, -0.10], [0.00, 0.00, 0.50]])
    sigma = np.array([0.008, 0.010, 0.012])
    A, Q = ekf._transition(K, sigma, DT)
    assert ekf._stationary(A)
    P0 = ekf._uncond_cov(A, Q)
    assert np.allclose(A @ P0 @ A.T - P0 + Q, 0.0, atol=1e-8)  # discrete Lyapunov
    assert np.allclose(P0, P0.T, atol=1e-10)


def test_negloglik_minimized_near_truth():
    p, X, Y = _simulate()
    rng = np.random.default_rng(1)
    nll_true = ekf.negloglik(p.pack(), Y, GRID, ELB, DT)
    npar = p.pack().size
    perturbed = [ekf.negloglik(ekf.ShadowParams.unpack(p.pack() + 0.2 * rng.standard_normal(npar)).pack(),
                               Y, GRID, ELB, DT) for _ in range(5)]
    assert np.isfinite(nll_true)
    assert nll_true < min(perturbed)


def test_filter_recovers_shadow_rate_dynamics():
    p, X, Y = _simulate()
    xs, fits = ekf.filter_states(p, Y, GRID, ELB, DT)
    s_true = X[:, 0] + X[:, 1]
    s_filt = xs[:, 0] + xs[:, 1]
    # Turning points recovered (level is weakly identified deep below the bound).
    assert np.corrcoef(s_filt, s_true)[0, 1] > 0.9
    # Yields fit well.
    assert np.mean(np.abs(fits - Y)) * 1e4 < 15.0  # < 15 bp
