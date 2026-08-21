"""The driver model's arithmetic and its anchoring rule."""
from __future__ import annotations

from model.tp_drivers import DRIVERS, TARGET_KINK, DriverModel

M = DriverModel(intercept=0.5,
                coef={"expected_inflation": 1.2, "inflation_uncertainty": 0.4,
                      "foreign_official": -0.06, "fed_holdings": -0.07,
                      "debt_gdp": 0.02})
BASE = dict(expected_inflation=2.3, inflation_uncertainty=1.3,
            foreign_official=12.0, fed_holdings=15.0, debt_gdp=99.0)


def test_premium_is_flat_at_and_below_the_inflation_target():
    a = M.premium(**{**BASE, "expected_inflation": 1.0})
    b = M.premium(**{**BASE, "expected_inflation": TARGET_KINK})
    assert abs(a - b) < 1e-12


def test_every_sign_is_the_one_theory_requires():
    up = lambda k, d: M.premium(**{**BASE, k: BASE[k] + d}) - M.premium(**BASE)
    assert up("expected_inflation", 1.0) > 0
    assert up("inflation_uncertainty", 1.0) > 0
    assert up("debt_gdp", 10.0) > 0
    assert up("foreign_official", 5.0) < 0        # price-insensitive holder absorbs
    assert up("fed_holdings", 5.0) < 0


def test_scenario_starts_from_todays_observed_premium():
    """Anchoring: an unchanged scenario must return exactly today's premium, residual
    and all -- the model supplies the change, never the level."""
    assert abs(M.scenario(BASE, BASE, tp_today=1.63) - 1.63) < 1e-12


def test_scenario_moves_by_the_model_change_not_to_its_fair_value():
    hot = {**BASE, "expected_inflation": 3.3, "inflation_uncertainty": 2.3}
    moved = M.scenario(BASE, hot, tp_today=1.63)
    expected = 1.63 + (M.premium(**hot) - M.premium(**BASE))
    assert abs(moved - expected) < 1e-12
    assert moved > 1.63


def test_a_shrinking_official_bid_raises_the_premium_on_its_own():
    """Inflation unchanged at target, but the price-insensitive buyers step back."""
    calm = {**BASE, "expected_inflation": 2.0, "inflation_uncertainty": 0.8}
    drained = {**calm, "foreign_official": 8.0, "fed_holdings": 9.0, "debt_gdp": 120.0}
    assert M.scenario(calm, drained, tp_today=1.0) > 1.0
