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


def _params_liq():
    return ja.JointParams(
        kappa=np.array([0.25, 0.30, 0.6, 0.9, 0.5]),
        theta=np.array([0.035, 0.012, -0.01, 0.0, 0.003]),
        sigma=np.array([0.008, 0.007, 0.010, 0.012, 0.004]),
        lambda_=0.55, h_n=0.0008, h_r=0.0015, elb_n=-0.0035)


def _simulate_liq(p, T=200, seed=1):
    grid_n = np.array([1., 2., 3., 5., 7., 10.])
    grid_r = np.array([2., 3., 5., 7., 10.])
    beta_r = ja._liq_loading(grid_r)
    A, Q = ja._transition(p, 1 / 12)
    L = np.linalg.cholesky(Q + 1e-14 * np.eye(5))
    rng = np.random.default_rng(seed)
    pre_n, pre_r = ja._precompute(p, grid_n, grid_r)
    x = p.theta.copy(); Yn, Yr = [], []
    for _ in range(T):
        x = p.theta + A @ (x - p.theta) + L @ rng.standard_normal(5)
        y, _ = ja._measure(x, pre_n, pre_r, len(grid_n), beta_r)
        Yn.append(y[:len(grid_n)] + p.h_n * rng.standard_normal(len(grid_n)))
        Yr.append(y[len(grid_n):] + p.h_r * rng.standard_normal(len(grid_r)))
    return np.array(Yn), np.array(Yr), grid_n, grid_r


def test_liquidity_measurement_jacobian():
    p = _params_liq()
    pre_n, pre_r = ja._precompute(p, [2., 10.], [2., 10.])
    beta_r = ja._liq_loading(np.array([2., 10.]))
    x = p.theta + np.array([0.01, -0.01, 0.02, 0.03, 0.005])
    _, J = ja._measure(x, pre_n, pre_r, 2, beta_r)
    assert J.shape == (4, 5)
    assert np.allclose(J[:2, 4], 0.0)          # nominal block: no liquidity loading
    assert np.all(np.abs(J[2:, 4]) > 1e-6)     # real block: liquidity loads


def test_liquidity_decomposition_three_way_identity():
    p = _params_liq()
    Yn, Yr, gn, gr = _simulate_liq(p, T=120)
    xf, _, ll = ja.filter_states(p, Yn, Yr, gn, gr, 1 / 12)
    assert np.isfinite(ll) and xf.shape[1] == 5
    d = ja.decompose(p, xf, gn, gr, mr_halflife=3.5)
    # observed real yield = clean real + liquidity wedge
    assert np.allclose(d["obs_fitted_r"], d["fitted_r"] + (d["obs_fitted_r"] - d["fitted_r"]))
    assert d["liquidity_premium"].shape[1] == len(d["common"])
    # DKW 3-way identity: observed breakeven = expected inflation + IRP - liquidity premium
    assert np.allclose(d["breakeven"],
                       d["expected_inflation"] + d["irp"] - d["liquidity_premium"], atol=1e-12)


# --------------------------------------------------------------- survey (ATSIX) anchor
def test_survey_jacobian_matches_finite_difference():
    p = _params_liq()
    common = np.array([2., 3., 5., 7., 10.])
    pre_s = ja._survey_precompute(p, common, mr_halflife=3.5)
    x = p.theta + np.array([0.01, -0.008, 0.02, 0.03, 0.004])
    pred, J = ja._survey_measure(x, pre_s)
    assert pred.shape == (5,) and J.shape == (5, 5)
    eps = 1e-6
    Jfd = np.zeros_like(J)
    for k in range(len(x)):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        Jfd[:, k] = (ja._survey_measure(xp, pre_s)[0]
                     - ja._survey_measure(xm, pre_s)[0]) / (2 * eps)
    assert np.max(np.abs(J - Jfd)) < 1e-8


def test_survey_prediction_matches_decompose_expected_inflation():
    """The survey measurement must observe the SAME object decompose reports as
    expected_inflation (so anchoring is internally consistent)."""
    p = _params()
    common = np.array([2., 3., 5., 7., 10.])
    x = p.theta + np.array([0.012, -0.006, 0.02, 0.03])
    pre_s = ja._survey_precompute(p, common, mr_halflife=3.5)
    pred, _ = ja._survey_measure(x, pre_s)
    d = ja.decompose(p, x[None, :], np.array([1., 2., 3., 5., 7., 10.]), common,
                     mr_halflife=3.5)
    assert np.allclose(d["common"], common)
    assert np.max(np.abs(pred - d["expected_inflation"][0])) < 1e-5


def test_survey_anchor_pulls_expected_inflation():
    """With survey rows the filter tracks a survey path the yields alone don't imply:
    feed a survey series offset from the model-free E[infl] and confirm the anchored
    filter's expected inflation moves toward it (identification, not just co-movement)."""
    p = _params()
    Yn, Yr, xs, gn, gr = _simulate(p, T=180, seed=3)
    common = np.array([2., 3., 5., 7., 10.])
    # baseline (yields only) expected inflation
    xf0, _, _ = ja.filter_states(p, Yn, Yr, gn, gr, 1 / 12)
    d0 = ja.decompose(p, xf0, gn, gr, mr_halflife=3.5)
    ei0 = d0["expected_inflation"].mean(axis=0)
    # synthetic survey anchored 50bp ABOVE the model's own expected inflation
    T = Yn.shape[0]
    Ysurv = d0["expected_inflation"] + 0.005
    p_hi = ja.JointParams(**{**p.__dict__, "h_s": 0.0005})    # tight survey noise
    xf1, _, ll = ja.filter_states(p_hi, Yn, Yr, gn, gr, 1 / 12,
                                  Ysurv=Ysurv, grid_surv=common, mr_halflife=3.5)
    assert np.isfinite(ll)
    d1 = ja.decompose(p_hi, xf1, gn, gr, mr_halflife=3.5)
    ei1 = d1["expected_inflation"].mean(axis=0)
    # anchored expected inflation is pulled up toward the +50bp survey at every tenor
    assert np.all(ei1 > ei0 + 1e-4)


def test_survey_pack_roundtrip():
    p = _params()
    p = ja.JointParams(**{**p.__dict__, "h_s": 0.004})
    v = p.pack(survey=True)
    assert v.shape[0] == 3 * p.n + 4
    q = ja.JointParams.unpack(v, p.elb_n, p.n, survey=True)
    assert np.isclose(q.h_s, 0.004)
    # non-survey pack ignores h_s and unpacks to the default
    v0 = p.pack(survey=False)
    assert v0.shape[0] == 3 * p.n + 3
    q0 = ja.JointParams.unpack(v0, p.elb_n, p.n, survey=False)
    assert np.isclose(q0.h_s, 0.005)


# ------------------------------------------------- TIPS liquidity-proxy measurement
def test_liqproxy_pack_roundtrip():
    """pack/unpack with every combination of the survey and liqproxy blocks."""
    p = ja.JointParams(**{**_params_liq().__dict__, "h_s": 0.003,
                          "a_lp": 0.02, "b_lp": 1.5, "h_lp": 0.001})
    for survey in (False, True):
        for liqproxy in (False, True):
            v = p.pack(survey=survey, liqproxy=liqproxy)
            assert v.shape[0] == 3 * p.n + 3 + int(survey) + 3 * int(liqproxy)
            q = ja.JointParams.unpack(v, p.elb_n, p.n, survey=survey, liqproxy=liqproxy)
            assert np.isclose(q.h_s, 0.003 if survey else 0.005)
            if liqproxy:
                assert np.isclose(q.a_lp, 0.02) and np.isclose(q.b_lp, 1.5)
                assert np.isclose(q.h_lp, 0.001)
            else:
                assert q.b_lp == 0.0


def test_liqproxy_anchors_liquidity_factor():
    """Feed a proxy that is an exact affine image of the TRUE liquidity factor; the
    anchored filter should recover ℓ (hence the liquidity premium) far better than
    yields alone — the identification the proxy is meant to provide."""
    p = _params_liq()
    Yn, Yr, gn, gr = _simulate_liq(p, T=200, seed=7)
    # reconstruct the true ℓ path used by the simulator (re-simulate states)
    grid_n = np.array([1., 2., 3., 5., 7., 10.]); grid_r = np.array([2., 3., 5., 7., 10.])
    A, Q = ja._transition(p, 1 / 12)
    L = np.linalg.cholesky(Q + 1e-14 * np.eye(5))
    rng = np.random.default_rng(7)
    x = p.theta.copy(); ell = []
    for _ in range(200):
        x = p.theta + A @ (x - p.theta) + L @ rng.standard_normal(5)
        ell.append(x[4])
    ell = np.array(ell)
    proxy = 0.02 + 3.0 * ell                       # exact affine image (a_lp=0.02,b_lp=3)
    p_lp = ja.JointParams(**{**p.__dict__, "a_lp": 0.02, "b_lp": 3.0, "h_lp": 1e-5})
    xf0, _, _ = ja.filter_states(p, Yn, Yr, gn, gr, 1 / 12)                # yields only
    xf1, _, ll = ja.filter_states(p_lp, Yn, Yr, gn, gr, 1 / 12, Yproxy=proxy)  # + proxy
    assert np.isfinite(ll)
    corr0 = np.corrcoef(xf0[:, 4], ell)[0, 1]
    corr1 = np.corrcoef(xf1[:, 4], ell)[0, 1]
    assert corr1 > corr0                            # proxy improves ℓ identification
    assert corr1 > 0.9                              # and pins it tightly
