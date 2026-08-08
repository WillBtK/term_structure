"""Tests for BRW bootstrap bias-correction of the P-dynamics."""
import numpy as np

from model import bias_correction as bc


def _simulate_var1(Phi, c, Omega, T, seed):
    rng = np.random.default_rng(seed)
    n = Phi.shape[0]
    L = np.linalg.cholesky(Omega)
    theta = np.linalg.solve(np.eye(n) - Phi, c)
    x = np.empty((T, n))
    x[0] = theta
    for t in range(1, T):
        x[t] = c + Phi @ x[t - 1] + L @ rng.standard_normal(n)
    return x


def test_fit_var1_recovers_known_dynamics():
    Phi = np.array([[0.95, 0.0], [0.10, 0.80]])
    c = np.array([0.01, 0.00])
    Omega = np.diag([1e-4, 4e-4])
    X = _simulate_var1(Phi, c, Omega, T=20000, seed=1)
    est = bc.fit_var1(X)
    assert np.allclose(est["Phi"], Phi, atol=0.02)


def test_phi_kp_roundtrip():
    K = np.array([[0.05, 0.0, 0.0], [0.0, 0.30, 0.0], [0.0, 0.0, 0.60]])
    dt = 1.0 / 12
    Phi = bc.kp_to_phi(K, dt)
    K2 = bc.phi_to_kp(Phi, dt)
    assert np.allclose(K, K2, atol=1e-9)


def test_bias_correction_increases_persistence_and_stays_stationary():
    # Near-unit-root, short sample -> OLS is biased low; correction pushes up.
    Phi = np.array([[0.99, 0.0], [0.0, 0.97]])
    c = np.array([0.99 * 0.0, 0.0]) + np.array([0.0004, 0.0])  # small drift
    Omega = np.diag([1e-4, 1e-4])
    X = _simulate_var1(Phi, c, Omega, T=300, seed=3)
    res = bc.bias_correct(X, dt=1.0 / 12, n_boot=200, seed=7)
    # Correction raises persistence (closer to the true near-unit-root).
    assert res.rho_bc > res.rho_hat
    # And keeps it in the stationary region.
    assert res.rho_bc < 1.0
    # theta held fixed at the OLS value.
    assert np.allclose(res.theta, bc.fit_var1(X)["theta"])


def test_bias_correction_deterministic():
    Phi = np.array([[0.9, 0.0], [0.0, 0.85]])
    c = np.array([0.001, 0.0])
    X = _simulate_var1(Phi, c, np.diag([1e-4, 1e-4]), T=250, seed=5)
    a = bc.bias_correct(X, dt=1.0 / 12, n_boot=100, seed=11)
    b = bc.bias_correct(X, dt=1.0 / 12, n_boot=100, seed=11)
    assert np.allclose(a.Phi_bc, b.Phi_bc)
