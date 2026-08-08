"""Tests for the shadow-rate EKF (synthetic ground truth)."""
import numpy as np
import pytest

from model import shadow_rate as sr
from model import ekf

DT = 1.0 / 12
ELB = -0.0035
GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def _true_params():
    return ekf.ShadowParams(
        kappa=np.array([0.06, 0.30, 0.60]),
        theta=np.array([0.04, -0.015, 0.0]),
        sigma=np.array([0.008, 0.010, 0.012]),
        lambda_=0.5, h=0.0006,
    )


def _simulate(T=360, seed=0):
    rng = np.random.default_rng(seed)
    p = _true_params()
    pre = sr.precompute(p.lambda_, np.diag(p.sigma), ELB, GRID)
    A, Q = ekf._transition(p.kappa, p.sigma, DT)
    Lq = np.linalg.cholesky(Q)
    X = np.empty((T, 3)); x = p.theta.copy()
    for t in range(T):
        x = p.theta + A @ (x - p.theta) + Lq @ rng.standard_normal(3)
        X[t] = x
    # ZLB episode: drive the slope down so the shadow short rate goes negative
    X[T // 3: 2 * T // 3, 1] += np.linspace(0, -0.05, 2 * T // 3 - T // 3)
    Y = np.array([sr.elb_yields(X[t], pre) for t in range(T)]) + p.h * rng.standard_normal((T, GRID.size))
    return p, X, Y


def test_transition_closed_form_matches_numeric():
    kappa = np.array([0.05, 0.3, 0.6]); sigma = np.array([0.008, 0.01, 0.012])
    A, Q = ekf._transition(kappa, sigma, DT)
    # A = diag(e^{-k dt})
    assert np.allclose(np.diag(A), np.exp(-kappa * DT))
    # Q vs numeric OU conditional covariance for diagonal K
    Qn = sr.cond_cov(np.diag(kappa), np.diag(sigma), DT)
    assert np.allclose(Q, Qn, atol=1e-10)


def test_negloglik_minimized_near_truth():
    p, X, Y = _simulate()
    rng = np.random.default_rng(1)
    nll_true = ekf.negloglik(p.pack(), Y, GRID, ELB, DT)
    perturbed = [ekf.negloglik(ekf.ShadowParams.unpack(p.pack() + 0.3 * rng.standard_normal(11)).pack(),
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
