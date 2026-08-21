"""Offline tests for the assembled scenario -> term premium chain (no network)."""
import numpy as np
import pytest

from model.okun import OkunParams
from model.stock_bond import SBCParams
from model.tp_scenario import (FairValueParams, fit_error_correction,
                               fit_fair_value, scenario_term_premium)

OKUN = OkunParams(floor=0.61, slope=0.41, kink=1.25)
SBC = SBCParams(intercept=-0.289, slope=0.936)
FV = FairValueParams(intercept=1.80, b_sbc=4.99, b_gap=-0.04, lam=0.315)


def test_horizon_zero_is_today_and_infinite_horizon_is_fair_value():
    assert FV.project(1.0, 2.5, 0.0) == pytest.approx(1.0)
    assert FV.project(1.0, 2.5, 200.0) == pytest.approx(2.5, abs=1e-6)


def test_share_closed_is_monotone_and_matches_the_quoted_speeds():
    assert FV.share_closed(3) < FV.share_closed(10)
    # 10y tenor, lam=0.315: ~68% closed at 3y, ~98% at 10y (see FINDINGS)
    assert FV.share_closed(3) == pytest.approx(0.68, abs=0.02)
    assert FV.share_closed(10) == pytest.approx(0.98, abs=0.02)
    fast = FairValueParams(1.8, 4.99, lam=0.493)          # 3y tenor reverts faster
    assert fast.share_closed(3) > FV.share_closed(3)


def test_higher_terminal_inflation_raises_the_premium_through_the_whole_chain():
    """The economic content: more inflation -> more uncertainty -> bonds hedge worse
    -> higher premium. If any link flips sign this test fails."""
    lo = scenario_term_premium(2.0, 1.0, 10, OKUN, SBC, FV, sigma_g=4.43)
    hi = scenario_term_premium(5.0, 1.0, 10, OKUN, SBC, FV, sigma_g=4.43)
    assert hi["sigma_pi"] > lo["sigma_pi"]
    assert hi["stock_bond_corr"] > lo["stock_bond_corr"]
    assert hi["tp_fair_value"] > lo["tp_fair_value"]
    assert hi["tp_at_horizon"] > lo["tp_at_horizon"]


def test_below_the_okun_kink_inflation_does_not_move_the_premium():
    """The hockey stick must survive to the output: 0% and 1% are the same world."""
    a = scenario_term_premium(0.0, 1.0, 10, OKUN, SBC, FV, sigma_g=4.43)
    b = scenario_term_premium(1.0, 1.0, 10, OKUN, SBC, FV, sigma_g=4.43)
    assert a["tp_fair_value"] == pytest.approx(b["tp_fair_value"])


def test_every_intermediate_is_returned_for_inspection():
    out = scenario_term_premium(3.0, 1.2, 3, OKUN, SBC, FV, sigma_g=4.43)
    for k in ("sigma_pi", "ratio", "stock_bond_corr", "tp_fair_value",
              "share_closed", "tp_at_horizon"):
        assert k in out and np.isfinite(out[k])


def test_fair_value_and_ecm_recover_known_parameters():
    rng = np.random.default_rng(0)
    sbc = rng.uniform(-0.4, 0.3, 200)
    gap = rng.normal(0, 3.0, 200)
    tp = 1.80 + 4.99 * sbc - 0.04 * gap + rng.normal(0, 0.05, 200)
    f = fit_fair_value(tp, sbc, gap)
    assert f.b_sbc == pytest.approx(4.99, abs=0.1)
    assert f.b_gap == pytest.approx(-0.04, abs=0.02)
    gap_fv = rng.normal(0, 1.0, 200)
    d = 0.315 * gap_fv + rng.normal(0, 0.02, 200)
    assert fit_error_correction(d, gap_fv) == pytest.approx(0.315, abs=0.02)


def test_ecm_fit_refuses_a_sample_too_small():
    with pytest.raises(ValueError, match="at least 8"):
        fit_error_correction([0.1, 0.2], [1.0, 1.0])
