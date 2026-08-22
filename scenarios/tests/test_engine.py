"""The scenario engine's contracts: anchoring, Leibowitz, monotonicity, paths."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model.tp_drivers import DriverModel
from model.tp_relative import RelativeModel
from model.tp_drivers import RESIDUAL_PERSISTENCE
from scenarios.engine import (HORIZONS, MarketState, Scenario, drivers_at,
                              forward_average, leibowitz_return, load_scenario,
                              propagated_uncertainty, real_return, run, run_market,
                              step)

CORE = pd.Series(2.3, index=pd.date_range("2000-01-01", "2026-07-01", freq="MS"))
TODAY = {"expected_inflation": 2.3, "inflation_uncertainty": 0.6,
         "price_insensitive": 27.0, "debt_gdp": 99.0}
STATE = MarketState("US", y_today=4.74, tp_today=1.63,
                    drivers_today=dict(TODAY), core_history=CORE)
US_MODEL = DriverModel(intercept=0.5,
                       coef={"expected_inflation": 1.25, "inflation_uncertainty": 0.4,
                             "price_insensitive": -0.065, "debt_gdp": 0.014})

def flat_scenario(country="US", policy=3.0, infl=2.3, n=3, **kw):
    return Scenario(country, policy_rate={i: policy for i in range(1, n + 1)},
                    inflation={i: infl for i in range(1, n + 1)}, **kw)


# ------------------------------------------------------------------------- paths

def test_paths_hold_flat_beyond_the_last_projection():
    p = {1: 4.0, 2: 3.0, 3: 2.5}
    assert step(p, 3) == step(p, 10) == step(p, 20) == 2.5
    assert step(p, 0) == step(p, 1) == 4.0
    assert forward_average(p, 3) == pytest.approx(2.5)      # all terminal
    assert forward_average(p, 0) == pytest.approx((4.0 + 3.0 + 2.5 * 8) / 10)


def test_scenario_years_must_be_contiguous_from_one():
    with pytest.raises(ValueError):
        Scenario("US", policy_rate={2: 3.0}, inflation={2: 2.0})
    with pytest.raises(ValueError):
        Scenario("US", policy_rate={1: 3.0, 3: 2.0}, inflation={1: 2.0, 3: 2.0})


# --------------------------------------------------------------------- leibowitz

def test_return_is_the_yield_when_nothing_moves():
    assert leibowitz_return(4.0, 4.0, 3, 8.0) == pytest.approx(4.0)


def test_convergence_at_twice_duration_minus_one():
    """LBK: at H = 2D-1 the annualised return equals the starting yield whatever
    happens to yields -- the duration drag and the reinvestment gain cancel."""
    D = 8.0
    H = int(2 * D - 1)
    for yH in (2.0, 4.0, 7.5):
        assert leibowitz_return(4.0, yH, H, D) == pytest.approx(4.0)


def test_short_horizons_are_duration_dominated_long_ones_reinvestment():
    up = 1.0
    assert leibowitz_return(4.0, 5.0, 3, 8.0) < 4.0        # H < 2D-1: selloff hurts
    assert leibowitz_return(4.0, 5.0, 20, 8.0) > 4.0       # H > 2D-1: selloff helps


def test_real_return_deflates_by_average_projected_inflation():
    sc = flat_scenario(infl=2.0)
    assert real_return(5.0, sc, 3) == pytest.approx((1.05 / 1.02 - 1) * 100)
    assert real_return(5.0, sc, 3) < 5.0


# ------------------------------------------------------------------- uncertainty

def test_extending_the_status_quo_leaves_uncertainty_at_todays_value():
    """The anchoring contract for the volatility channel: a scenario that just
    continues today's inflation rate is not information, and must not move u."""
    sc = flat_scenario(infl=float(CORE.iloc[-1]))
    for h in (1, 3, 10):
        assert propagated_uncertainty(STATE, sc, h) == pytest.approx(
            TODAY["inflation_uncertainty"], abs=1e-9)


def test_the_baseline_makes_no_calm_assumption_after_turbulence():
    """After a turbulent stretch, freezing at the last reading is still the neutral
    path -- the model must not assume calm returns any more than it assumes calm
    ends. Only the scenario's own departures move the driver."""
    hist = CORE.copy()
    hist.iloc[-24:] = [2 + 3 * np.sin(i) for i in range(24)]
    st = MarketState("US", 4.7, 1.6, dict(TODAY), hist)
    freeze = flat_scenario(infl=float(hist.iloc[-1]))
    for h in (3, 10):
        assert propagated_uncertainty(st, freeze, h) == pytest.approx(
            TODAY["inflation_uncertainty"], abs=1e-9)
    swing = Scenario("US", policy_rate={1: 3.0, 2: 3.0, 3: 3.0},
                     inflation={1: 6.0, 2: 1.0, 3: 5.0})
    assert (propagated_uncertainty(st, swing, 3)
            > propagated_uncertainty(st, freeze, 3))


def test_uncertainty_never_goes_negative():
    st = MarketState("US", 4.7, 1.6, {**TODAY, "inflation_uncertainty": 0.05}, CORE)
    calm = flat_scenario(infl=float(CORE.iloc[-1]))
    assert propagated_uncertainty(st, calm, 10) >= 0.0


def test_a_swinging_path_raises_uncertainty_a_flat_one_does_not():
    calm = flat_scenario(infl=2.0)
    swing = Scenario("US", policy_rate={1: 3.0, 2: 3.0, 3: 3.0},
                     inflation={1: 6.0, 2: 1.0, 3: 5.0})
    assert (propagated_uncertainty(STATE, swing, 3)
            > propagated_uncertainty(STATE, calm, 3) + 0.5)


# ----------------------------------------------------------------- the full chain

CURVE = STATE.y_today - STATE.tp_today          # what today's curve prices


def _flat_everything(policy=CURVE):
    """The genuinely neutral scenario: a policy path whose ten-year average equals
    what the curve already prices, inflation at today's expectation, drivers pinned."""
    return flat_scenario(policy=policy, infl=TODAY["expected_inflation"],
                         price_insensitive={i: TODAY["price_insensitive"]
                                            for i in range(1, 4)},
                         debt_gdp={i: TODAY["debt_gdp"] for i in range(1, 4)})


def _rows(sc, state=STATE, model=US_MODEL, rho=1.0):
    tp = {0: state.tp_today}
    for h in HORIZONS:
        tp[h] = model.scenario(state.drivers_today, drivers_at(state, sc, h),
                               state.tp_today, horizon=h, rho=rho)
    return run_market(state, sc, tp)


def test_the_residual_decays_geometrically_and_is_whole_at_the_zero_horizon():
    """rho=1 keeps today's unexplained premium for ever; rho=0 discards it at once;
    in between it decays at rho per year. At h=0 it is always whole, whatever rho --
    the model never restates today."""
    sc = _flat_everything()
    d0 = STATE.drivers_today
    resid = STATE.tp_today - US_MODEL.premium(**d0)
    assert abs(resid) > 0.1                                   # a real gap to decay
    for rho in (1.0, 0.9, 0.5, 0.0):
        assert US_MODEL.scenario(d0, d0, STATE.tp_today, horizon=0, rho=rho) == \
            pytest.approx(STATE.tp_today, abs=1e-12)
        for h in HORIZONS:
            got = US_MODEL.scenario(d0, drivers_at(STATE, sc, h), STATE.tp_today,
                                    horizon=h, rho=rho)
            fair = US_MODEL.premium(**drivers_at(STATE, sc, h))
            assert got == pytest.approx(fair + resid * rho ** h, abs=1e-12)


def test_a_temporary_premium_shock_fades_rather_than_persisting():
    """The point of the dial: an unexplained spike today must not be projected forward
    as though permanent."""
    spiked = replace(STATE, tp_today=STATE.tp_today + 1.0)
    sc, rho = _flat_everything(), RESIDUAL_PERSISTENCE
    calm_kept, calm_faded = _rows(sc, rho=1.0), _rows(sc, rho=rho)
    kept, faded = _rows(sc, state=spiked, rho=1.0), _rows(sc, state=spiked, rho=rho)
    for ck, cf, k, f in zip(calm_kept, calm_faded, kept, faded):
        assert k.tp == pytest.approx(ck.tp + 1.0, abs=1e-9)      # rho=1: whole shock
        # like for like at the same rho: the shock survives in part, never in full
        assert f.tp == pytest.approx(cf.tp + rho ** f.horizon, abs=1e-9)
        assert cf.tp < f.tp < k.tp


def test_lower_persistence_never_changes_todays_anchor():
    for rho in (1.0, 0.9, 0.0):
        rows = _rows(_flat_everything(), rho=rho)
        for r in rows:
            assert r.yield_10y == pytest.approx(r.tp + r.policy_avg, abs=1e-12)


def test_the_yield_is_always_the_premium_plus_the_expected_policy_average():
    """The identity the results card asserts. It must hold exactly, for every
    scenario -- not just the neutral one."""
    for policy in (CURVE, CURVE + 1.5, 0.5):
        for r in _rows(_flat_everything(policy=policy)):
            assert r.yield_10y == pytest.approx(r.tp + r.policy_avg, abs=1e-12)


def test_a_policy_path_matching_the_curve_returns_todays_yield_and_carry():
    for r in _rows(_flat_everything()):
        assert r.policy_gap == pytest.approx(0.0, abs=1e-9)
        assert r.yield_10y == pytest.approx(STATE.y_today, abs=1e-9)
        assert r.nominal_return == pytest.approx(STATE.y_today, abs=1e-9)


def test_a_policy_view_above_the_curve_raises_the_yield_and_costs_return():
    """The whole point of taking the level: disagreeing with the market has to show
    up as a capital loss, not be quietly discarded."""
    base, hawk = _rows(_flat_everything()), _rows(_flat_everything(policy=CURVE + 0.5))
    for b, h in zip(base, hawk):
        assert h.policy_gap == pytest.approx(0.5, abs=1e-9)
        assert h.yield_10y == pytest.approx(b.yield_10y + 0.5, abs=1e-9)
        assert h.nominal_return < b.nominal_return


def test_the_policy_gap_is_measured_against_the_curve_not_the_scenario():
    r = _rows(_flat_everything(policy=CURVE + 0.4))[0]
    assert r.curve_expected == pytest.approx(CURVE, abs=1e-12)
    assert r.policy_gap == pytest.approx(0.4, abs=1e-9)


def test_higher_inflation_means_higher_yield_and_lower_three_year_return():
    base = _flat_everything()
    hot = flat_scenario(policy=CURVE, infl=4.5,
                        price_insensitive=base.price_insensitive,
                        debt_gdp=base.debt_gdp)
    b, h = _rows(base)[0], _rows(hot)[0]
    assert h.yield_10y > b.yield_10y + 1.0     # premium + expectations both rise
    assert h.nominal_return < b.nominal_return
    assert h.real_return < b.real_return - 1.0


def test_spread_market_inherits_the_us_move_at_its_beta():
    rel = RelativeModel(country="DE", intercept=0.1, beta_us=1.0,
                        coef={"expected_inflation": 1.0, "inflation_uncertainty": 0.5,
                              "price_insensitive": -0.05, "debt_gdp": 0.02})
    # anchored means anchored: the DE scenario extends DE's own status quo, so its
    # inflation sits at the core history's last reading and today's expectation.
    de_state = MarketState("DE", 3.32, 0.96,
                           {"expected_inflation": 2.3, "inflation_uncertainty": 0.4,
                            "price_insensitive": 16.0, "debt_gdp": 64.0}, CORE)
    de_curve = de_state.y_today - de_state.tp_today
    de_flat = flat_scenario("DE", policy=de_curve, infl=float(CORE.iloc[-1]),
                            price_insensitive={1: 16.0}, debt_gdp={1: 64.0})
    res = run(US_MODEL, {"DE": rel},
              {"US": STATE, "DE": de_state},
              {"US": _flat_everything(), "DE": de_flat}, rho=1.0)
    for r in res["DE"]:      # nothing moved, path matches the curve, residual retained
        assert r.yield_10y == pytest.approx(de_state.y_today, abs=1e-9)
        assert r.yield_10y == pytest.approx(r.tp + r.policy_avg, abs=1e-12)


def test_load_scenario_roundtrip_and_validation(tmp_path):
    f = tmp_path / "us.csv"
    f.write_text("year,policy_rate,inflation,real_growth\n1,3.9,2.7,1.8\n2,3.4,2.4,1.7\n")
    sc = load_scenario(f)
    assert sc.country == "US" and sc.policy_rate == {1: 3.9, 2: 3.4}
    bad = tmp_path / "uk.csv"
    bad.write_text("year,policy_rate\n1,3.5\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_scenario(bad)


def test_the_us_fiscal_driver_reads_the_path_five_years_ahead():
    """The driver is the five-year-ahead projected deficit, so horizon h looks at
    year h+5 of the path, flat beyond its last year."""
    st = MarketState("US", 4.74, 1.63, {**TODAY, "fiscal": 5.9}, CORE)
    sc = flat_scenario()
    path = {1: 5.0, 2: 5.5, 3: 6.0}
    d = drivers_at(st, sc, 3, deficit_default=path)
    assert d["fiscal"] == 6.0                      # year 8 -> flat extension of year 3
    d0 = drivers_at(st, sc, 0, deficit_default=path)
    assert d0["fiscal"] == 6.0                     # year 5 -> also the terminal
    assert "fiscal" not in drivers_at(
        MarketState("DE", 3.3, 0.96, dict(TODAY), CORE), sc, 3)


def test_without_any_path_the_us_fiscal_driver_holds_at_today():
    st = MarketState("US", 4.74, 1.63, {**TODAY, "fiscal": 5.87}, CORE)
    assert drivers_at(st, flat_scenario(), 3)["fiscal"] == 5.87


def test_a_wider_deficit_path_raises_the_us_premium_at_the_measured_sensitivity():
    m = DriverModel(intercept=0.5,
                    coef={"expected_inflation": 1.0, "inflation_uncertainty": 0.8,
                          "price_insensitive": -0.03},
                    drivers=("expected_inflation", "inflation_uncertainty",
                             "price_insensitive"),
                    adjust={"fiscal": 0.52}, base={"fiscal": 5.87})
    st = MarketState("US", 4.74, 1.63, {**TODAY, "fiscal": 5.87}, CORE)
    sc = flat_scenario()
    base = drivers_at(st, sc, 3, deficit_default={1: 5.87})
    wide = drivers_at(st, sc, 3, deficit_default={1: 7.87})
    got = (m.scenario(st.drivers_today, wide, 1.63, horizon=3, rho=1.0)
           - m.scenario(st.drivers_today, base, 1.63, horizon=3, rho=1.0))
    assert got == pytest.approx(2.0 * 0.52, abs=1e-9)
