"""Correctness tests for the joint nominal/real AFNS (no slow estimation).

Checks the decisive, fast properties: the two-block measurement Jacobian structure,
the EKF tracking known-parameter synthetic data, and the exact accounting identities
of the decomposition (breakeven = expected inflation + IRP; TP = fitted - expected).
"""
import numpy as np
import pytest

from model import joint_afns as ja


def _params():
    return ja.JointParams(
        kappa=np.array([0.25, 0.30, 0.6, 0.9]),
        theta=np.array([0.035, 0.012, -0.01, 0.0]),
        sigma=np.array([0.008, 0.007, 0.010, 0.012]),
        lambda_=0.55, h_n=0.0008, h_r=0.0015, elb_n=-0.0035)


def _simulate(p, T=240, seed=0):
    grid_n = np.array([1., 2., 3., 5., 7., 10.])
    grid_r = np.array([2., 3., 5., 7., 10.])          # ragged: real has no 1y
    A, Q = ja._transition(p, 1 / 12)
    L = np.linalg.cholesky(Q + 1e-14 * np.eye(4))
    rng = np.random.default_rng(seed)
    pre_n, pre_r = ja._precompute(p, grid_n, grid_r)
    x = p.theta.copy()
    Yn, Yr, xs = [], [], []
    for _ in range(T):
        x = p.theta + A @ (x - p.theta) + L @ rng.standard_normal(4)
        y, _ = ja._measure(x, pre_n, pre_r, len(grid_n))
        yn = y[:len(grid_n)] + p.h_n * rng.standard_normal(len(grid_n))
        yr = y[len(grid_n):] + p.h_r * rng.standard_normal(len(grid_r))
        Yn.append(yn); Yr.append(yr); xs.append(x.copy())
    return np.array(Yn), np.array(Yr), np.array(xs), grid_n, grid_r


def test_measurement_jacobian_block_structure():
    p = _params()
    pre_n, pre_r = ja._precompute(p, [2., 10.], [2., 10.])
    x = p.theta + np.array([0.01, -0.01, 0.02, 0.03])
    _, J = ja._measure(x, pre_n, pre_r, 2)
    # nominal rows (first 2) have zero sensitivity to L_R (col 1); real rows to L_N (col 0)
    assert np.allclose(J[:2, 1], 0.0)
    assert np.allclose(J[2:, 0], 0.0)
    # nominal loads on L_N (col 0) and real on L_R (col 1)
    assert np.all(np.abs(J[:2, 0]) > 1e-6)
    assert np.all(np.abs(J[2:, 1]) > 1e-6)


def test_filter_tracks_known_params():
    p = _params()
    Yn, Yr, xs, gn, gr = _simulate(p, T=180)
    xf, fits, ll = ja.filter_states(p, Yn, Yr, gn, gr, 1 / 12)
    assert np.isfinite(ll)
    # fitted nominal yields track observed to within a few bp (known params)
    rmse = np.sqrt(np.nanmean((fits[:, :len(gn)] - Yn) ** 2))
    assert rmse < 0.0040, rmse                       # < 40 bp (obs noise h_n=8bp + transient)
    # filtered state correlates with the true simulated state (level factors)
    assert np.corrcoef(xf[:, 0], xs[:, 0])[0, 1] > 0.9


def test_decomposition_identities():
    p = _params()
    Yn, Yr, xs, gn, gr = _simulate(p, T=120)
    xf, _, _ = ja.filter_states(p, Yn, Yr, gn, gr, 1 / 12)
    d = ja.decompose(p, xf, gn, gr, mr_halflife=3.5)
    # TP = fitted - expected, exactly, per curve
    assert np.allclose(d["tp_n"], d["fitted_n"] - d["exp_n"])
    assert np.allclose(d["tp_r"], d["fitted_r"] - d["exp_r"])
    # breakeven = expected inflation + IRP, exactly, on the common grid
    assert np.allclose(d["breakeven"], d["expected_inflation"] + d["irp"], atol=1e-12)
    # common grid is the intersection (real has no 1y -> excluded)
    assert 1.0 not in d["common"] and 10.0 in d["common"]
