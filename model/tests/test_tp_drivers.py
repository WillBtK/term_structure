"""The driver model's arithmetic, its anchoring rule, and the calibrated fiscal term."""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.tp_drivers import (DEBT_COEF, DRIVERS, SIGNS, TARGET_KINK, DriverModel,
                              design, fit_market)

COEF = {"expected_inflation": 1.2, "inflation_uncertainty": 0.4,
        "price_insensitive": -0.065, "debt_gdp": 0.014}
M = DriverModel(intercept=0.5, coef=COEF, debt_fitted=True)
CALIB = DriverModel(intercept=0.5, coef={**COEF, "debt_gdp": DEBT_COEF},
                    debt_fitted=False, debt_base=240.0)
BASE = dict(expected_inflation=2.3, inflation_uncertainty=1.3,
            price_insensitive=27.0, debt_gdp=99.0)


def test_premium_is_flat_at_and_below_the_inflation_target():
    a = M.premium(**{**BASE, "expected_inflation": 1.0})
    b = M.premium(**{**BASE, "expected_inflation": TARGET_KINK})
    assert abs(a - b) < 1e-12


def test_every_sign_is_the_one_theory_requires():
    up = lambda k, d: M.premium(**{**BASE, k: BASE[k] + d}) - M.premium(**BASE)
    for k, s in ((k, SIGNS[k]) for k in DRIVERS):
        assert np.sign(up(k, 5.0)) == s, k


def test_scenario_starts_from_todays_observed_premium():
    """Anchoring: an unchanged scenario must return exactly today's premium, residual
    and all -- the model supplies the change, never the level."""
    assert abs(M.scenario(BASE, BASE, tp_today=1.63) - 1.63) < 1e-12
    assert abs(CALIB.scenario(BASE, BASE, tp_today=1.36) - 1.36) < 1e-12


def test_scenario_moves_by_the_model_change_not_to_its_fair_value():
    hot = {**BASE, "expected_inflation": 3.3, "inflation_uncertainty": 2.3}
    moved = M.scenario(BASE, hot, tp_today=1.63)
    assert abs(moved - (1.63 + M.premium(**hot) - M.premium(**BASE))) < 1e-12
    assert moved > 1.63


def test_a_shrinking_official_bid_raises_the_premium_on_its_own():
    """Inflation unchanged at target, but the price-insensitive buyers step back."""
    calm = {**BASE, "expected_inflation": 2.0, "inflation_uncertainty": 0.8}
    drained = {**calm, "price_insensitive": 15.0, "debt_gdp": 120.0}
    assert M.scenario(calm, drained, tp_today=1.0) > 1.0


def test_calibrated_fiscal_term_is_measured_from_todays_debt_ratio():
    """Where debt is calibrated it contributes nothing at the base and exactly
    DEBT_COEF per point thereafter -- so it never distorts the historical level."""
    at_base = {**BASE, "debt_gdp": CALIB.debt_base}
    assert abs(CALIB.premium(**at_base)
               - (CALIB.intercept
                  + COEF["expected_inflation"] * (BASE["expected_inflation"] - TARGET_KINK)
                  + COEF["inflation_uncertainty"] * BASE["inflation_uncertainty"]
                  + COEF["price_insensitive"] * BASE["price_insensitive"])) < 1e-12
    ten_more = {**at_base, "debt_gdp": CALIB.debt_base + 10.0}
    assert abs(CALIB.premium(**ten_more) - CALIB.premium(**at_base)
               - 10.0 * DEBT_COEF) < 1e-12


def test_design_kinks_inflation_and_leaves_the_rest_alone():
    p = pd.DataFrame({"expected_inflation": [1.0, 2.0, 3.5],
                      "inflation_uncertainty": [0.5, 1.0, 1.5],
                      "price_insensitive": [10.0, 20.0, 30.0],
                      "debt_gdp": [80.0, 90.0, 100.0]})
    x = design(p)
    assert list(x["expected_inflation"]) == [0.0, 0.0, 1.5]
    assert list(x["debt_gdp"]) == [80.0, 90.0, 100.0]


def _synthetic(n=300, seed=0, debt_signal=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    p = pd.DataFrame({
        "expected_inflation": 2.0 + np.abs(rng.normal(0.8, 0.5, n)),
        "inflation_uncertainty": np.abs(rng.normal(1.0, 0.3, n)),
        "price_insensitive": rng.normal(25.0, 8.0, n),
        "debt_gdp": (np.linspace(60, 110, n) if debt_signal
                     else np.linspace(150, 240, n))}, index=idx)
    x = design(p)
    sign = 1.0 if debt_signal else -1.0
    p["tp"] = (0.5 + sum(COEF[k] * x[k] for k in DRIVERS if k != "debt_gdp")
               + sign * 0.02 * x["debt_gdp"] + rng.normal(0, 0.05, n))
    return p


def test_fit_recovers_known_coefficients_when_the_signal_is_there():
    m = fit_market("SYN", _synthetic())
    assert m.debt_fitted
    for k in DRIVERS:
        assert abs(m.coef[k] - COEF[k]) < 0.35 * max(abs(COEF[k]), 0.02), k


def test_a_perverse_debt_sample_falls_back_to_the_calibration():
    """Japan's problem in miniature: the premium falls as debt climbs. The fit must
    refuse to estimate a fiscal effect rather than pin it at zero."""
    m = fit_market("SYN", _synthetic(debt_signal=False))
    assert not m.debt_fitted
    assert m.coef["debt_gdp"] == DEBT_COEF
    assert m.debt_base == 240.0
