"""The endogenous deficit: sign, closure, calibration, and the off-switch."""
import pytest

from macro.fiscal_rules import (GAP_CLOSURE, SEMI_ELASTICITY, STRUCTURAL_HORIZON,
                                adjusted_deficit, output_gap, structural_deficit)

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


# ---------------------------------------------------- the structural (priced) deficit

def _official(path, k):
    return path[max(x for x in path if x <= k)]


def test_taking_the_cycle_out_of_a_derived_deficit_returns_the_official_path():
    """The identity everything rests on: derived row -> exactly official, every market,
    every year a bond window can reach, including growth that keeps moving late."""
    growth = {1: 0.2, 2: -0.5, 3: 2.4, 6: 1.1, 20: 1.9}
    base = {1: 1.8, 10: 1.5, 30: 1.2, 50: 1.0}           # an OBR-length baseline
    for c in SEMI_ELASTICITY:
        d = adjusted_deficit(c, growth, base, OFFICIAL)
        s = structural_deficit(c, d, growth, base)
        for k in range(1, STRUCTURAL_HORIZON + 1):
            assert s[k] == pytest.approx(_official(OFFICIAL, k), abs=1e-12)


def test_a_stated_deficit_is_adjusted_by_the_cycle_its_growth_row_implies():
    growth = {1: 0.8}
    stated = {k: 6.0 for k in range(1, 11)}
    s = structural_deficit("DE", stated, growth, BASE)
    gap = output_gap(growth, BASE, STRUCTURAL_HORIZON)
    for k in (1, 5, 10, 30):
        assert s[k] == pytest.approx(6.0 + SEMI_ELASTICITY["DE"] * gap[k])
    assert s[10] < 6.0                     # a slump's headline deficit is partly cycle


def test_no_growth_view_means_no_adjustment():
    stated = {1: 4.0, 5: 3.5}
    assert structural_deficit("UK", stated, {}, BASE) == stated


def test_the_rule_and_the_adjustment_share_one_gap():
    """If the recursion ever forks between derivation and adjustment, the identity
    above breaks silently; pin the gap itself."""
    gap = output_gap({1: 0.8}, BASE, 3)
    d1 = BASE[min(BASE)] - 0.8
    assert gap[1] == pytest.approx(-d1)
    assert gap[2] == pytest.approx((1 - GAP_CLOSURE) * gap[1] - d1)


# ------------------------------------------------ a stated row past its last year

def test_a_copy_of_the_derived_row_prices_exactly_like_it():
    """THE regression. Taking a row manual without changing a number used to move the
    US premium -30bp: the stated six years were held flat past the grid, throwing away
    the official out-years. The deviation is what is held now."""
    from macro.fiscal_rules import extend_stated_deficit
    derived = {k: 5.8 + 0.08 * k for k in range(0, 16)}          # rising, like CBO
    stated = {k: derived[k] for k in range(0, 6)}                 # the six grid years
    assert extend_stated_deficit(stated, derived) == pytest.approx(derived)


def test_a_stated_deviation_is_held_for_the_rest_of_the_path():
    from macro.fiscal_rules import extend_stated_deficit
    derived = {k: 5.8 + 0.08 * k for k in range(0, 16)}
    stated = {k: derived[k] + 1.0 for k in range(0, 6)}
    out = extend_stated_deficit(stated, derived)
    for k in range(6, 16):
        assert out[k] == pytest.approx(derived[k] + 1.0)
