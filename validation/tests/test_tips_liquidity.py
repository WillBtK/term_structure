"""TIPS 'noise' liquidity gauge: the maturity taper makes it robust to basket
composition changes (the 2026-01-14 step was one near-maturity outlier crossing a
hard cutoff)."""
import numpy as np

from validation import tips_liquidity as tl


def test_bond_weight_taper():
    # 0 at/below the floor, 1 above the taper top, monotone non-decreasing between.
    assert tl._bond_weight(np.array([tl._MIN_RESID_MAT]))[0] == 0.0
    assert tl._bond_weight(np.array([tl._TAPER_HI]))[0] == 1.0
    assert tl._bond_weight(np.array([10.0]))[0] == 1.0
    assert tl._bond_weight(np.array([tl._MAX_RESID_MAT]))[0] == 0.0     # long cutoff
    grid = np.linspace(tl._MIN_RESID_MAT, tl._TAPER_HI, 50)
    w = tl._bond_weight(grid)
    assert np.all(np.diff(w) >= -1e-12)                                  # monotone up


def test_composition_robustness():
    """A near-floor outlier bond leaving the basket must NOT step the gauge."""
    rng = np.random.default_rng(0)
    tau = np.linspace(2.0, 20.0, 20)
    beta = np.array([0.02, -0.01, 0.005])
    y = tl._ns_design(tau) @ beta + rng.normal(0, 2e-4, tau.size)       # clean curve
    # add one distorted bond just above the floor (a ~30 bp outlier, like the 01-14 case)
    tau_out = np.array([tl._MIN_RESID_MAT + 0.01])
    y_out = tl._ns_design(tau_out) @ beta + 30e-4
    with_out = tl._rms_residual(np.r_[tau, tau_out], np.r_[y, y_out])
    without = tl._rms_residual(tau, y)
    # the outlier is down-weighted to ~0, so its presence barely moves the gauge
    assert abs(with_out - without) < 0.2e-4                             # < 0.2 bp step
