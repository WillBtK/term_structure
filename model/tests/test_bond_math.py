"""Offline tests for coupon-bond math (no network).

The gauge these feed reads a few basis points of cross-sectional scatter, so a
systematic error in accrual or discounting would show up as "illiquidity". Each test
below pins a property that has an answer independent of the implementation.
"""
import numpy as np
import pandas as pd
import pytest

from model.bond_math import (_priced_schedule, accrued_interest, coupon_dates, cashflows,
                             disc_from_zero, price_from_curve, price_from_ytm,
                             ytm_from_price, yields_from_prices)


def test_schedule_runs_back_from_maturity_on_the_maturity_day():
    d = coupon_dates("2026-08-19", "2030-02-15")
    assert list(d.strftime("%m-%d").unique()) == ["02-15", "08-15"]
    assert d[-1] == pd.Timestamp("2030-02-15")
    assert d[0] > pd.Timestamp("2026-08-19")
    # settling 19 Aug 2026 the 15 Aug coupon has already gone, so 7 remain
    assert len(d) == 7


def test_accrued_is_zero_on_a_coupon_date_and_half_a_coupon_mid_period():
    # settle exactly on a coupon date: the schedule rolls to the next one, nothing accrued
    assert accrued_interest("2027-02-15", "2030-02-15", 4.0) == pytest.approx(0.0)
    # 2026-05-16 sits 90 of 181 days into the 15 Feb -> 15 Aug period
    a = accrued_interest("2026-05-16", "2030-02-15", 4.0)
    assert a == pytest.approx(4.0 / 2 * 90 / 181, rel=1e-12)


def test_par_bond_prices_at_100_when_the_yield_equals_the_coupon():
    # on a coupon date, with no accrual, a bond yielding its coupon is worth par
    p = price_from_ytm("2027-02-15", "2032-02-15", 4.0, 0.04)
    assert p == pytest.approx(100.0, abs=0.02)


def test_price_and_ytm_are_exact_inverses():
    for coupon, y in [(0.0, 0.045), (2.5, 0.01), (6.5, 0.055), (1.25, -0.004)]:
        p = price_from_ytm("2026-08-19", "2034-05-15", coupon, y)
        assert ytm_from_price(p, "2026-08-19", "2034-05-15", coupon) == pytest.approx(y, abs=1e-10)


def test_price_falls_as_yield_rises():
    args = ("2026-08-19", "2036-11-15", 3.0)
    ps = [price_from_ytm(*args, y) for y in (0.01, 0.03, 0.05, 0.07)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_flat_continuous_curve_reprices_to_the_equivalent_semiannual_yield():
    """A flat continuously-compounded zero curve at c must price a bond at the
    semiannual yield 2*(exp(c/2)-1) — the standard compounding conversion. This is the
    join between the Svensson curve (continuous) and quoted yields (semiannual); if it
    were wrong every bond would carry the same spurious residual."""
    c = 0.04
    disc = disc_from_zero(lambda t: np.full_like(np.asarray(t, dtype=float), c))
    p_curve = price_from_curve("2026-08-19", "2033-05-15", 3.5, disc)
    y_equiv = 2.0 * (np.exp(c / 2.0) - 1.0)
    p_ytm = price_from_ytm("2026-08-19", "2033-05-15", 3.5, y_equiv)
    assert p_curve == pytest.approx(p_ytm, abs=2e-3)


def test_zero_coupon_matches_the_discount_factor():
    t_years = (pd.Timestamp("2031-08-19") - pd.Timestamp("2026-08-19")).days / 365.25
    disc = disc_from_zero(lambda t: np.full_like(np.asarray(t, dtype=float), 0.03))
    p = price_from_curve("2026-08-19", "2031-08-19", 0.0, disc)
    assert p == pytest.approx(100.0 * np.exp(-0.03 * t_years), abs=1e-9)


def test_bad_rows_return_nan_instead_of_raising():
    # a matured bond, a nonsense price, and a price no yield in the bracket can produce
    assert np.isnan(ytm_from_price(99.0, "2026-08-19", "2020-01-01", 4.0))
    assert np.isnan(ytm_from_price(0.0, "2026-08-19", "2030-01-01", 4.0))
    assert np.isnan(ytm_from_price(np.nan, "2026-08-19", "2030-01-01", 4.0))


def test_yields_from_prices_vectorises_and_survives_one_bad_row():
    panel = pd.DataFrame({
        "date": ["2026-08-19"] * 3,
        "maturity": ["2030-02-15", "2034-05-15", "2019-01-01"],
        "coupon": [4.0, 2.5, 3.0],
        "price": [101.5, 92.0, 99.0],
    })
    out = yields_from_prices(panel)
    assert out["yield"].notna().sum() == 2
    assert np.isnan(out["yield"].iloc[2])
    # a bond below par yields more than its coupon; above par, less
    assert out["yield"].iloc[1] > 0.025
    assert out["yield"].iloc[0] < 0.04


def test_the_hoisted_schedule_matches_the_public_pieces():
    """``_priced_schedule`` exists only so the root-finder does not rebuild the coupon
    calendar on every iteration. It must stay bit-identical to the public functions it
    replaced inside that loop, or the speedup silently becomes a different answer."""
    settle, maturity, cpn = pd.Timestamp("2026-08-18"), pd.Timestamp("2053-11-15"), 4.75
    t, cf, accrued = _priced_schedule(settle, maturity, cpn)
    t_ref, cf_ref = cashflows(settle, maturity, cpn)
    assert np.allclose(t, t_ref, rtol=0, atol=0)
    assert np.allclose(cf, cf_ref, rtol=0, atol=0)
    assert accrued == accrued_interest(settle, maturity, cpn)


def test_yield_inversion_round_trips():
    """Price -> yield -> price must return the input; this is what pins the inversion
    after the objective was rewritten to work off a pre-built schedule."""
    settle, maturity, cpn = pd.Timestamp("2026-08-18"), pd.Timestamp("2034-05-15"), 4.125
    for price in (85.0, 98.523438, 107.25):
        y = ytm_from_price(price, settle, maturity, cpn)
        assert price_from_ytm(settle, maturity, cpn, y) == pytest.approx(price, abs=1e-9)
