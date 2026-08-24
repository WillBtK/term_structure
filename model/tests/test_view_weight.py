"""One view, one speed: the premium and the expectations leg must price a stated
view at the same rate.

The defect these lock out: the expectations leg phased a stated rate view in at
``1 - lambda**h`` while the premium's drivers took the matching inflation path at face
value from year one. A single coherent forecast -- tighten, and inflation falls with it
-- therefore moved the two legs by different fractions of itself, and the net yield
moved on the gap between two dials rather than on the economics.
"""
from __future__ import annotations

import numpy as np
import pytest

from model.tp_drivers import DriverModel, RESIDUAL_PERSISTENCE
from model.tp_relative import RelativeModel


def _us_model() -> DriverModel:
    return DriverModel(
        drivers=("expected_inflation", "inflation_uncertainty", "price_insensitive"),
        coef={"expected_inflation": 1.0, "inflation_uncertainty": 0.8,
              "price_insensitive": -0.03},
        intercept=0.15, r2=0.9, rmse=0.5, adjust={}, base={})


def _drv(infl=2.5, unc=1.2, own=27.0) -> dict:
    return {"expected_inflation": infl, "inflation_uncertainty": unc,
            "price_insensitive": own}


def test_weight_one_reproduces_the_unblended_premium():
    """The generalisation must be exact at w=1, or it is a change of model."""
    m, today, scen = _us_model(), _drv(), _drv(infl=4.0, unc=1.8)
    for h in (1, 3, 10):
        old = m.premium(**scen) + (1.63 - m.premium(**today)) * RESIDUAL_PERSISTENCE ** h
        new = m.scenario(today, scen, 1.63, horizon=h, rho=RESIDUAL_PERSISTENCE,
                         view_weight=1.0)
        assert new == pytest.approx(old, abs=1e-12)


def test_weight_zero_leaves_only_the_residual_decaying():
    """None of the view priced yet: the premium is today's, less what the residual
    has shed. The residual is not a view and keeps decaying at rho."""
    m, today, scen = _us_model(), _drv(), _drv(infl=4.0)
    resid = 1.63 - m.premium(**today)
    for h in (1, 3, 10):
        got = m.scenario(today, scen, 1.63, horizon=h, rho=0.9, view_weight=0.0)
        assert got == pytest.approx(1.63 + resid * (0.9 ** h - 1), abs=1e-12)


def test_the_premium_moves_linearly_in_the_weight():
    m, today, scen = _us_model(), _drv(), _drv(infl=4.0, unc=1.8)
    full = m.scenario(today, scen, 1.63, horizon=3, rho=0.9, view_weight=1.0)
    none = m.scenario(today, scen, 1.63, horizon=3, rho=0.9, view_weight=0.0)
    half = m.scenario(today, scen, 1.63, horizon=3, rho=0.9, view_weight=0.5)
    assert half == pytest.approx((full + none) / 2.0, abs=1e-12)


def test_zero_horizon_now_returns_today_without_the_engine_special_casing_it():
    """At h=0 the engine's weight is 1 - lam**0 = 0, so the premium is today's for
    any rho -- the identity now falls out of the weight.

    It did not before. Unweighted, h=0 returned today's premium PLUS the whole move
    in fair value, because ``drivers_at(.., 0)`` reads the scenario's ten-year forward
    average, which is the user's entire path. The engine papered over that by writing
    ``tp[0] = tp_today`` at the call site rather than calling the model. That worked,
    but it meant the anchoring was a convention of one caller instead of a property of
    the model."""
    m, today, scen = _us_model(), _drv(), _drv(infl=6.0, unc=3.0)
    for rho in (0.0, 0.9, 1.0):
        assert m.scenario(today, scen, 1.63, horizon=0, rho=rho,
                          view_weight=0.0) == pytest.approx(1.63, abs=1e-12)
    # and the old behaviour is what it was: today plus the full move
    unweighted = m.scenario(today, scen, 1.63, horizon=0, rho=0.9, view_weight=1.0)
    assert unweighted == pytest.approx(
        1.63 + m.premium(**scen) - m.premium(**today), abs=1e-12)


def test_a_disinflationary_tightening_moves_both_legs_by_the_same_fraction():
    """The case that motivated this. The premium's share of its own full move must
    equal the expectations leg's share of the stated rate view -- otherwise the net
    yield is an artefact of the two dials disagreeing."""
    m, today = _us_model(), _drv(infl=3.5)
    scen = _drv(infl=2.0)                       # inflation brought back to target
    lam, h = 0.8, 3.0
    w = 1 - lam ** h
    full_move = m.premium(**scen) - m.premium(**today)
    got = m.scenario(today, scen, 1.63, horizon=h, rho=1.0, view_weight=w)
    priced = (got - 1.63) / full_move           # rho=1 isolates the view channel
    assert priced == pytest.approx(w, abs=1e-12)


def _rel_model() -> RelativeModel:
    return RelativeModel(
        country="DE", beta_us=1.30, intercept=-0.2,
        coef={"expected_inflation": 0.6, "inflation_uncertainty": 0.4,
              "price_insensitive": -0.02, "debt_gdp": 0.0},
        adjust={}, base={}, r2=0.8, rmse=0.4)


def test_relative_weight_one_reproduces_the_unblended_premium():
    m = _rel_model()
    us_t, us_s = _drv(), _drv(infl=4.0)
    own_t, own_s = _drv(infl=2.0), _drv(infl=3.0)
    us_t["debt_gdp"] = us_s["debt_gdp"] = own_t["debt_gdp"] = own_s["debt_gdp"] = 90.0
    for h in (1, 3, 10):
        resid = 0.9 - m.premium(us_t, own_t, 1.63)
        old = m.premium(us_s, own_s, 2.1) + resid * 0.9 ** h
        new = m.scenario(us_t, us_s, own_t, own_s, 1.63, 2.1, 0.9,
                         horizon=h, rho=0.9, view_weight=1.0)
        assert new == pytest.approx(old, abs=1e-12)


def test_relative_does_not_damp_the_us_channel_twice():
    """The US premium arriving here has already been blended, so a pure US move must
    reach this market at full beta whatever the weight -- damping it again would make
    every spread market converge more slowly than the US for no reason."""
    m = _rel_model()
    d = _drv(); d["debt_gdp"] = 90.0
    for w in (0.0, 0.4, 1.0):
        got = m.scenario(d, d, d, d, 1.63, 2.13, 0.9, horizon=3, rho=1.0, view_weight=w)
        assert got - 0.9 == pytest.approx(m.beta_us * 0.5, abs=1e-12)
