"""TIPS 'noise' liquidity gauge.

Two properties are pinned here: the maturity taper that makes the gauge robust to
basket composition changes (the 2026-01-14 step was one near-maturity outlier crossing
a hard cutoff), and the coupon parsed out of the FRED series title, which the shared
fit regresses out because yield to maturity is not a function of maturity alone.
"""
import numpy as np
import pytest

from validation import tips_liquidity as tl
from validation.noise_core import residual_se, svensson_design


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
    beta = np.array([0.02, -0.01, 0.005, 0.0])
    y = svensson_design(tau, 0.7, 0.15) @ beta + rng.normal(0, 2e-4, tau.size)
    # one distorted bond just above the floor (a ~30 bp outlier, like the 01-14 case)
    tau_out = np.array([tl._MIN_RESID_MAT + 0.01])
    y_out = svensson_design(tau_out, 0.7, 0.15) @ beta + 30e-4
    both_tau, both_y = np.r_[tau, tau_out], np.r_[y, y_out]
    with_out = residual_se(both_tau, both_y, weights=tl._bond_weight(both_tau))
    without = residual_se(tau, y, weights=tl._bond_weight(tau))
    # the outlier is down-weighted to ~0, so its presence barely moves the gauge
    assert abs(with_out - without) < 0.2e-4                             # < 0.2 bp step


@pytest.mark.parametrize("title,expected", [
    ("10-Year 3-3/8% Treasury Inflation-Indexed Note, Due 1/15/2007", 3.375),
    ("10-Year 2% Treasury Inflation-Indexed Note, Due 1/15/2014", 2.0),
    ("30-Year 1-3/4% Treasury Inflation-Indexed Bond, Due 2/15/2045", 1.75),
    ("Treasury Inflation-Indexed Note, Due 1/15/2030", float("nan")),
])
def test_coupon_parsed_from_the_fred_title(title, expected):
    got = tl.parse_coupon(title)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)
    # the leading "10-Year"/"30-Year" must not be mistaken for a coupon
    assert not (got == 10.0 or got == 30.0)
