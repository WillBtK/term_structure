"""The US-plus-spread structure: anchoring, adjustment handling, and fit hygiene."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.tp_drivers import DEBT_COEF, DRIVERS, TARGET_KINK, DriverModel
from model.tp_relative import RelativeModel, _relative, fit_relative

US = dict(expected_inflation=2.3, inflation_uncertainty=1.3,
          price_insensitive=27.0, debt_gdp=99.0)
OWN = dict(expected_inflation=2.0, inflation_uncertainty=1.7,
           price_insensitive=38.0, debt_gdp=101.0)
BASE = _relative(OWN, US)
M = RelativeModel(country="XX", intercept=0.2, beta_us=0.9,
                  coef={"expected_inflation": 0.9, "price_insensitive": -0.05,
                        "debt_gdp": 0.06},
                  adjust={"inflation_uncertainty": 0.39}, base=dict(BASE),
                  borrowed=("inflation_uncertainty",))
CAL = RelativeModel(country="YY", intercept=0.2, beta_us=0.7,
                    coef={"price_insensitive": -0.014},
                    adjust={"expected_inflation": 1.25, "inflation_uncertainty": 0.39,
                            "debt_gdp": DEBT_COEF},
                    base=dict(BASE), debt_calibrated=True)


def test_relative_drivers_kink_each_side_before_differencing():
    r = _relative({**OWN, "expected_inflation": 1.4},
                  {**US, "expected_inflation": 1.8})
    assert r["expected_inflation"] == 0.0          # both below target, no difference
    r2 = _relative({**OWN, "expected_inflation": 3.0}, US)
    assert abs(r2["expected_inflation"] - (1.0 - 0.3)) < 1e-12


def test_adjustments_are_worth_nothing_at_todays_values():
    """An unidentified channel must not shift the level -- only respond to a change."""
    plain = (M.intercept + M.beta_us * 1.63
             + sum(M.coef[k] * BASE[k] for k in M.coef))
    assert abs(M.premium(US, OWN, 1.63) - plain) < 1e-12
    plain_c = (CAL.intercept + CAL.beta_us * 1.63
               + sum(CAL.coef[k] * BASE[k] for k in CAL.coef))
    assert abs(CAL.premium(US, OWN, 1.63) - plain_c) < 1e-12


def test_an_unchanged_scenario_returns_todays_premium_exactly():
    for m in (M, CAL):
        got = m.scenario(US, US, OWN, OWN, tp_us_today=1.63, tp_us_scenario=1.63,
                         tp_today=1.39)
        assert abs(got - 1.39) < 1e-12


def test_the_us_premium_carries_through_at_the_estimated_beta():
    got = M.scenario(US, US, OWN, OWN, tp_us_today=1.63, tp_us_scenario=2.63,
                     tp_today=1.39)
    assert abs(got - (1.39 + M.beta_us)) < 1e-12


def test_a_calibrated_fiscal_channel_still_responds_to_the_scenario():
    """Japan's case: unidentified in its own history, but a widening deficit against the
    US must still raise the premium."""
    wider = {**OWN, "debt_gdp": OWN["debt_gdp"] + 20.0}
    got = CAL.scenario(US, US, OWN, wider, tp_us_today=1.36, tp_us_scenario=1.36,
                       tp_today=1.36)
    assert abs(got - (1.36 + 20.0 * DEBT_COEF)) < 1e-12


def test_a_borrowed_inflation_channel_responds_to_own_inflation_only():
    hot = {**OWN, "expected_inflation": OWN["expected_inflation"] + 1.0}
    got = CAL.scenario(US, US, OWN, hot, tp_us_today=1.36, tp_us_scenario=1.36,
                       tp_today=1.36)
    assert got > 1.36
    us_hot = {**US, "expected_inflation": US["expected_inflation"] + 1.0}
    both = CAL.scenario(US, us_hot, OWN, hot, tp_us_today=1.36, tp_us_scenario=1.36,
                        tp_today=1.36)
    assert abs(both - 1.36) < 1e-12               # relative move is nil, US TP held


def _pair(n=260, seed=1, perverse_debt=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2004-10-01", periods=n, freq="MS")
    p = pd.DataFrame(index=idx)
    p["us_tp"] = np.cumsum(rng.normal(0, 0.08, n)) + 1.5
    p["rel_expected_inflation"] = rng.normal(0.1, 0.3, n)
    p["rel_inflation_uncertainty"] = rng.normal(0.2, 0.3, n)
    p["rel_price_insensitive"] = rng.normal(10.0, 6.0, n)
    p["rel_debt_gdp"] = np.linspace(-20.0, 20.0, n)
    p["tp"] = (0.2 + 0.9 * p["us_tp"] + 0.9 * p["rel_expected_inflation"]
               + 0.4 * p["rel_inflation_uncertainty"]
               - 0.05 * p["rel_price_insensitive"]
               + (-0.06 if perverse_debt else 0.06) * p["rel_debt_gdp"]
               + rng.normal(0, 0.05, n))
    return p


REF = DriverModel(intercept=0.5,
                  coef={"expected_inflation": 1.25, "inflation_uncertainty": 0.39,
                        "price_insensitive": -0.065, "debt_gdp": 0.014})


def test_fit_recovers_the_us_beta_and_the_relative_slopes():
    m = fit_relative("SYN", _pair(), REF)
    assert abs(m.beta_us - 0.9) < 0.08
    assert abs(m.coef["price_insensitive"] + 0.05) < 0.02
    assert not m.debt_calibrated
    assert m.r2 > 0.9


def test_a_perverse_relative_debt_sample_is_calibrated_not_fitted():
    m = fit_relative("SYN", _pair(perverse_debt=True), REF)
    assert m.debt_calibrated
    assert m.adjust["debt_gdp"] == DEBT_COEF
    assert "debt_gdp" not in m.coef


def test_the_fit_never_contains_an_adjustment_channel():
    """Adjustments are measured from today, so including them in the historical fit
    would impose swings that never happened. They must stay out of coef."""
    m = fit_relative("SYN", _pair(perverse_debt=True), REF)
    assert not set(m.coef) & set(m.adjust)
    assert m.r2 > 0.5
