"""Tests for the shadow-rate AFNS math (pure numpy/scipy, no external deps)."""
import numpy as np
import pytest

from model import shadow_rate as sr

LAM = 0.5
SIGMA = np.diag([0.011, 0.010, 0.020])
GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
ELB = -0.0035


def test_forward_and_price_loading_shapes_and_limits():
    tau = np.array([1e-8, 1.0, 10.0])
    b = sr.forward_loadings(LAM, tau)
    assert b.shape == (3, 3)
    # tau->0: b -> [1,1,0]
    assert np.allclose(b[0], [1.0, 1.0, 0.0], atol=1e-6)
    B = sr.price_loadings(LAM, tau)
    assert np.allclose(B[0], [0.0, 0.0, 0.0], atol=1e-6)  # B(0)=0


def test_elb_forward_limits():
    fs = np.array([0.05, 0.001, -0.05])
    om = np.array([0.005, 0.005, 0.005])
    # ELB -> -inf collapses to shadow
    assert np.allclose(sr.elb_forward(fs, om, -1e6), fs, atol=1e-8)
    # omega -> 0 gives max(fs, ELB)
    assert np.allclose(sr.elb_forward(np.array([0.05, -0.05]), np.array([0.0, 0.0]), ELB),
                       [0.05, ELB], atol=1e-12)
    # fs far below ELB -> ~ELB
    assert sr.elb_forward(np.array([-0.05]), np.array([0.004]), ELB)[0] == pytest.approx(ELB, abs=5e-4)
    # ELB forward is always >= ELB
    assert np.all(sr.elb_forward(fs, om, ELB) >= ELB - 1e-12)


def test_omega_grid_matches_reference_and_monotone():
    u = np.linspace(0.0, 10.0, 60)
    fast = sr.omega_on_grid(LAM, SIGMA, u)
    ref = sr.omega(LAM, SIGMA, u)
    assert np.allclose(fast, ref, atol=5e-4)
    assert fast[0] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.diff(fast) >= -1e-9)  # non-decreasing


def test_collapse_to_plain_afns_at_high_rates():
    # High shadow short rate: ELB yields collapse to unbounded (shadow) yields.
    # Collapse is tight at the short/medium end; a tiny wedge persists at the very
    # long end because the level factor is a Q-random-walk (large ω at 10y) -
    # this is genuine CR shadow-rate behaviour, not a bug.
    x = np.array([0.06, 0.00, 0.00])  # s = 6%
    pre_elb = sr.precompute(LAM, SIGMA, ELB, GRID)
    pre_free = sr.precompute(LAM, SIGMA, -1e6, GRID)
    diff_bp = np.abs(sr.elb_yields(x, pre_elb) - sr.elb_yields(x, pre_free)) * 1e4
    short_med = GRID <= 5.0
    assert np.max(diff_bp[short_med]) < 1.0     # negligible through 5y
    assert np.max(diff_bp) < 5.0                # small even at 10y
    assert diff_bp.argmax() == GRID.size - 1     # wedge largest at the long end


def test_short_end_floored_below_bound():
    x = np.array([0.005, -0.020, 0.010])  # s = -1.5% (below ELB)
    pre_elb = sr.precompute(LAM, SIGMA, ELB, GRID)
    pre_free = sr.precompute(LAM, SIGMA, -1e6, GRID)
    y_elb = sr.elb_yields(x, pre_elb)
    y_free = sr.elb_yields(x, pre_free)
    assert y_elb[0] > y_free[0]           # ELB lifts the short end
    assert y_elb[0] > ELB - 1e-4          # short-end yield ~ at/above ELB


def test_jacobian_matches_numeric():
    pre = sr.precompute(LAM, SIGMA, ELB, GRID)
    x = np.array([0.02, -0.01, 0.005])
    y, J = sr.elb_yields_and_jac(x, pre)
    eps = 1e-6
    Jn = np.empty_like(J)
    for k in range(3):
        xp = x.copy(); xp[k] += eps
        xm = x.copy(); xm[k] -= eps
        Jn[:, k] = (sr.elb_yields(xp, pre) - sr.elb_yields(xm, pre)) / (2 * eps)
    assert np.allclose(J, Jn, atol=1e-6)
    assert np.allclose(y, sr.elb_yields(x, pre))


def test_shadow_short_rate():
    assert sr.shadow_short_rate(np.array([0.03, -0.01, 0.5])) == pytest.approx(0.02)
