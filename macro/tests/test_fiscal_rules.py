"""The endogenous deficit: sign, closure, calibration, and the off-switch."""
import pytest

from macro.fiscal_rules import GAP_CLOSURE, SEMI_ELASTICITY, adjusted_deficit

OFFICIAL = {k: 5.0 for k in range(0, 11)}
BASE = {k: 1.8 for k in range(1, 11)}


def test_no_growth_view_returns_the_official_path_unchanged():
    assert adjusted_deficit("US", {}, BASE, OFFICIAL) == OFFICIAL


def test_growth_on_the_baseline_changes_nothing():
    out = adjusted_deficit("US", dict(BASE), BASE, OFFICIAL)
    assert all(abs(out[k] - 5.0) < 1e-12 for k in out)


def test_slower_growth_widens_the_deficit_and_faster_narrows_it():
    slow = adjusted_deficit("US", {1: 0.8}, BASE, OFFICIAL)   # holds flat: -1pp/yr
    fast = adjusted_deficit("US", {1: 2.8}, BASE, OFFICIAL)
    assert slow[5] > 5.0 > fast[5]
    assert abs((slow[5] - 5.0) + (fast[5] - 5.0)) < 1e-12     # linear, symmetric


def test_the_us_calibration_reproduces_cbo_at_year_five():
    # CBO rules of thumb (Feb 2026): 1pp/yr slower growth widens the deficit
    # 1.0pp of GDP at year five. gap(5) = sum_{j=0..4} 0.7^j = 2.7731.
    out = adjusted_deficit("US", {1: 0.8}, BASE, OFFICIAL)
    gap5 = sum(0.7 ** j for j in range(5))
    assert out[5] - 5.0 == pytest.approx(SEMI_ELASTICITY["US"] * gap5, abs=1e-9)
    assert out[5] - 5.0 == pytest.approx(1.0, abs=0.005)


def test_a_one_year_miss_decays_back_toward_the_official_path():
    # 1pp slower in year 1 only, then back on baseline: the gap closes at 30%/yr.
    growth = {1: 0.8, 2: 1.8}
    out = adjusted_deficit("UK", growth, BASE, OFFICIAL)
    assert out[1] - 5.0 == pytest.approx(SEMI_ELASTICITY["UK"] * 1.0, abs=1e-9)
    assert out[5] - 5.0 == pytest.approx(
        SEMI_ELASTICITY["UK"] * (1 - GAP_CLOSURE) ** 4, abs=1e-9)
    assert out[10] < out[5]                                    # still decaying


def test_year_zero_is_never_adjusted():
    out = adjusted_deficit("DE", {1: 0.0}, BASE, OFFICIAL)
    assert out[0] == OFFICIAL[0]
    assert out[1] != OFFICIAL[1]


def test_each_market_uses_its_own_semi_elasticity():
    outs = {c: adjusted_deficit(c, {1: 0.8}, BASE, OFFICIAL)[1] - 5.0
            for c in SEMI_ELASTICITY}
    for c, eps in SEMI_ELASTICITY.items():
        assert outs[c] == pytest.approx(eps, abs=1e-9)
    assert outs["UK"] > outs["DE"] > outs["JP"] > outs["US"]


def test_fractional_scenario_keys_are_handled():
    # load_scenario converts calendar years to fractional horizons (e.g. 0.36).
    out = adjusted_deficit("US", {0.36: 0.8, 1.36: 0.8}, BASE, OFFICIAL)
    assert out[5] > 5.0
