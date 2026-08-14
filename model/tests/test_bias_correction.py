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


def test_diagonal_correction_keeps_K_real_and_diagonal():
    # On near-unit-root, collinear factors an unrestricted correction can yield a
    # full K^P with COMPLEX eigenvalues (distorting the expected-rate path). The
    # diagonal mode keeps Phi (and K) diagonal with real, positive mean-reversion.
    rng = np.random.default_rng(0)
    # two highly persistent, strongly correlated factors (collinear, ZLB-like)
    T = 360
    z = np.zeros((T, 2))
    for t in range(1, T):
        shock = rng.standard_normal()
        z[t, 0] = 0.995 * z[t - 1, 0] + 0.01 * shock
        z[t, 1] = 0.99 * z[t - 1, 1] + 0.01 * (0.9 * shock + 0.1 * rng.standard_normal())
    full = bc.bias_correct(z, dt=1.0 / 12, n_boot=300, seed=1, diagonal=False)
    diag = bc.bias_correct(z, dt=1.0 / 12, n_boot=300, seed=1, diagonal=True)
    # diagonal K is exactly diagonal and real
    assert np.allclose(diag.K_bc, np.diag(np.diag(diag.K_bc)))
    assert np.all(np.isreal(np.linalg.eigvals(diag.K_bc)))
    assert np.all(np.diag(diag.K_bc) > 0)                 # positive mean-reversion
    # diagonal fit_var1 constrains off-diagonals to exactly zero
    assert bc.fit_var1(z, diagonal=True)["Phi"][0, 1] == 0.0


def test_auto_mode_falls_back_to_diagonal_on_complex_dynamics():
    # Collinear near-unit-root factors -> unrestricted correction goes complex ->
    # auto mode must fall back to diagonal (real, diagonal K).
    rng = np.random.default_rng(0)
    T = 360
    z = np.zeros((T, 2))
    for t in range(1, T):
        s = rng.standard_normal()
        z[t, 0] = 0.995 * z[t - 1, 0] + 0.01 * s
        z[t, 1] = 0.99 * z[t - 1, 1] + 0.01 * (0.9 * s + 0.1 * rng.standard_normal())
    full = bc.bias_correct(z, dt=1.0 / 12, n_boot=300, seed=1, diagonal=False)
    auto = bc.bias_correct(z, dt=1.0 / 12, n_boot=300, seed=1, diagonal="auto")
    if bc._kp_has_complex_dynamics(full.K_bc):        # the case this guard is for
        assert auto.mode == "diagonal"
        assert np.allclose(auto.K_bc, np.diag(np.diag(auto.K_bc)))
        assert np.all(np.isreal(np.linalg.eigvals(auto.K_bc)))


def test_auto_mode_keeps_full_when_well_behaved():
    # Well-separated stationary factors -> unrestricted K is real -> auto keeps full
    # (this is what preserves the US cross-factor SF-Fed match).
    Phi = np.array([[0.95, 0.02], [0.0, 0.80]])
    c = np.array([0.002, 0.0])
    X = _simulate_var1(Phi, c, np.diag([1e-4, 1e-4]), T=600, seed=2)
    auto = bc.bias_correct(X, dt=1.0 / 12, n_boot=300, seed=1, diagonal="auto")
    assert auto.mode == "full"
    assert not np.allclose(auto.Phi_bc, np.diag(np.diag(auto.Phi_bc)))  # off-diagonal kept


def test_bias_correction_deterministic():
    Phi = np.array([[0.9, 0.0], [0.0, 0.85]])
    c = np.array([0.001, 0.0])
    X = _simulate_var1(Phi, c, np.diag([1e-4, 1e-4]), T=250, seed=5)
    a = bc.bias_correct(X, dt=1.0 / 12, n_boot=100, seed=11)
    b = bc.bias_correct(X, dt=1.0 / 12, n_boot=100, seed=11)
    assert np.allclose(a.Phi_bc, b.Phi_bc)
