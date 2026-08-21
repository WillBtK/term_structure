"""The ten-year-equivalent weighting, and the duration it rests on.

The property that matters for the supply channel is the LAST test: a portfolio of bills
and a portfolio of long bonds can hold identical par and wildly different duration risk.
That gap is the entire reason par-share measures fail to identify QE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.bond_math import macaulay_duration, modified_duration
from supply.tenyear import (ladder_durations, reference_duration,
                            ten_year_equivalents, tye_share)

ASOF = pd.Timestamp("2020-01-15")


def flat(y):
    """A flat continuously compounded zero curve at ``y`` decimal."""
    return lambda t: np.full_like(np.asarray(t, dtype=float), y)


def test_zero_coupon_duration_is_time_to_maturity():
    mat = ASOF + pd.DateOffset(years=5)
    d = macaulay_duration(ASOF, mat, 0.0, 0.03)
    assert abs(d - (mat - ASOF).days / 365.25) < 1e-9


def test_coupon_bond_duration_is_below_its_maturity():
    mat = ASOF + pd.DateOffset(years=10)
    d = macaulay_duration(ASOF, mat, 5.0, 0.05)
    assert 7.0 < d < 9.99


def test_duration_falls_with_coupon_and_with_yield():
    mat = ASOF + pd.DateOffset(years=20)
    assert (macaulay_duration(ASOF, mat, 2.0, 0.04)
            > macaulay_duration(ASOF, mat, 8.0, 0.04))
    assert (macaulay_duration(ASOF, mat, 5.0, 0.02)
            > macaulay_duration(ASOF, mat, 5.0, 0.08))


def test_modified_duration_is_the_price_sensitivity():
    mat = ASOF + pd.DateOffset(years=10)
    from model.bond_math import price_from_ytm
    y, bump = 0.04, 1e-6
    p0 = price_from_ytm(ASOF, mat, 4.0, y)
    p1 = price_from_ytm(ASOF, mat, 4.0, y + bump)
    numeric = -(p1 - p0) / bump / p0            # clean-price sensitivity
    analytic = modified_duration(ASOF, mat, 4.0, y)
    assert abs(numeric - analytic) < 0.02       # differ only by accrued, which is 0 here


def test_a_portfolio_of_the_benchmark_has_tye_equal_to_its_par():
    curve = flat(0.03)
    ref = reference_duration(ASOF, curve)
    assert ten_year_equivalents([100.0], [ref], ref) == 100.0


def test_bills_and_bonds_with_equal_par_carry_very_different_tye():
    curve = flat(0.03)
    ref = reference_duration(ASOF, curve)
    bills = pd.DataFrame({"maturity_date": [ASOF + pd.DateOffset(months=3)],
                          "coupon_pct": [0.0], "amount": [1000.0]})
    bonds = pd.DataFrame({"maturity_date": [ASOF + pd.DateOffset(years=30)],
                          "coupon_pct": [3.0], "amount": [1000.0]})
    t_bill = ten_year_equivalents(bills.amount, ladder_durations(bills, ASOF, curve), ref)
    t_bond = ten_year_equivalents(bonds.amount, ladder_durations(bonds, ASOF, curve), ref)
    assert t_bill < 0.05 * t_bond               # same par, ~20x the duration risk
    assert t_bond > 2.0 * 1000.0                # a 30y is worth >2 ten-year notes


def test_tye_share_is_100_when_the_holder_owns_everything():
    curve = flat(0.03)
    lad = pd.DataFrame({"maturity_date": [ASOF + pd.DateOffset(years=n) for n in (2, 10, 30)],
                        "coupon_pct": [2.0, 3.0, 3.5], "amount": [500.0, 700.0, 300.0]})
    r = tye_share(lad, lad, ASOF, curve)
    assert abs(r["tye_share"] - 100.0) < 1e-9
    assert abs(r["par_share"] - 100.0) < 1e-9


def test_tye_share_diverges_from_par_share_when_the_holder_is_long_duration():
    """The QE case: same par share, much larger duration share."""
    curve = flat(0.03)
    out = pd.DataFrame({"maturity_date": [ASOF + pd.DateOffset(years=n) for n in (1, 30)],
                        "coupon_pct": [2.0, 3.5], "amount": [1000.0, 1000.0]})
    held = pd.DataFrame({"maturity_date": [ASOF + pd.DateOffset(years=30)],
                         "coupon_pct": [3.5], "amount": [500.0]})
    r = tye_share(out, held, ASOF, curve)
    assert abs(r["par_share"] - 25.0) < 1e-9
    assert r["tye_share"] > 45.0                # duration share is far above par share
