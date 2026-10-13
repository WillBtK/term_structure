"""US DV01: the per-security pricing and the flow logic that isolates supply/SOMA
purchases from rolldown and rate moves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.bond_math import accrued_interest, modified_duration, price_from_ytm
from supply.us_dv01 import (FRN_EFFECTIVE_DURATION, aggregate, flow_dv01, security_dv01,
                            security_values)

ASOF = pd.Timestamp("2020-01-15")


def flat(y):
    return lambda t: np.full_like(np.asarray(t, dtype=float), y)


def _ladder(rows):
    """rows: list of (maturity_date, coupon_pct, amount, asset_class)."""
    mats, cps, amts, acs = zip(*rows)
    return pd.DataFrame({"maturity_date": list(mats), "coupon_pct": list(cps),
                         "amount": list(amts), "asset_class": list(acs)})


# --------------------------------------------------------------------- security_dv01

def test_bill_dv01_matches_hand_computed_modified_duration_times_price():
    mat = ASOF + pd.DateOffset(months=6)
    lad = _ladder([(mat, 0.0, 1000.0, "bill")])
    curve = flat(0.03)
    dv01 = security_dv01(lad, ASOF, curve, real_curve=None)
    ytm = 2.0 * (np.exp(0.03 / 2.0) - 1.0)
    moddur = modified_duration(ASOF, mat, 0.0, ytm)
    price = price_from_ytm(ASOF, mat, 0.0, ytm)
    expected = 1000.0 * (price / 100.0) * moddur * 0.0001
    assert abs(dv01.iloc[0] - expected) < 1e-9


def test_long_bond_carries_far_more_dv01_than_a_bill_of_equal_par():
    curve = flat(0.03)
    lad = _ladder([
        (ASOF + pd.DateOffset(months=3), 0.0, 1000.0, "bill"),
        (ASOF + pd.DateOffset(years=30), 3.0, 1000.0, "nominal"),
    ])
    dv01 = security_dv01(lad, ASOF, curve, real_curve=None)
    assert dv01.iloc[1] > 15 * dv01.iloc[0]


def test_tips_uses_the_real_curve_not_the_nominal_one():
    mat = ASOF + pd.DateOffset(years=10)
    lad = _ladder([(mat, 1.0, 1000.0, "tips")])
    nominal = flat(0.04)
    real = flat(0.01)
    dv01_real = security_dv01(lad, ASOF, nominal, real_curve=real)
    dv01_nominal_curve = security_dv01(lad, ASOF, real, real_curve=nominal)
    # swapping which curve is passed as "real" changes the answer -- proves the TIPS
    # row is priced off real_curve, not nominal_curve
    assert abs(dv01_real.iloc[0] - dv01_nominal_curve.iloc[0]) > 0.01 * dv01_real.iloc[0]


def test_tips_is_nan_without_a_real_curve():
    lad = _ladder([(ASOF + pd.DateOffset(years=10), 1.0, 1000.0, "tips")])
    dv01 = security_dv01(lad, ASOF, flat(0.03), real_curve=None)
    assert np.isnan(dv01.iloc[0])


def test_frn_dv01_is_the_stipulated_short_duration_not_curve_priced():
    """An FRN priced as a fixed coupon bond to its stated maturity would carry a long
    duration; the whole point is that it must not."""
    mat = ASOF + pd.DateOffset(years=5)
    lad = _ladder([(mat, 4.0, 1000.0, "frn")])
    dv01 = security_dv01(lad, ASOF, flat(0.04), real_curve=None)
    expected = 1000.0 * FRN_EFFECTIVE_DURATION * 0.0001
    assert abs(dv01.iloc[0] - expected) < 1e-9
    # a fixed-coupon bond of the same size/maturity/coupon would be far larger
    fixed = _ladder([(mat, 4.0, 1000.0, "nominal")])
    dv01_fixed = security_dv01(fixed, ASOF, flat(0.04), real_curve=None)
    assert dv01.iloc[0] < 0.1 * dv01_fixed.iloc[0]


# ------------------------------------------------------------------------- flow_dv01

def _flow_ladder(recs):
    """recs: list of (date, cusip, amount, dv01_mn)."""
    d, c, a, v = zip(*recs)
    return pd.DataFrame({"date": list(d), "cusip": list(c), "amount": list(a),
                         "dv01_mn": list(v)})


def test_unchanged_position_contributes_zero_flow():
    t0, t3 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-04-30")
    lad = _flow_ladder([(t0, "A", 100.0, 8.0), (t3, "A", 100.0, 7.5)])
    f = flow_dv01(lad, months=3)
    assert abs(f[t3]) < 1e-9          # same par both ends -> zero, despite ratio drift


def test_new_issue_contributes_its_full_current_dv01():
    t0, t3 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-04-30")
    lad = _flow_ladder([(t0, "A", 100.0, 8.0), (t3, "A", 100.0, 8.0),
                        (t3, "B", 50.0, 4.0)])           # B is new at t3
    f = flow_dv01(lad, months=3)
    assert abs(f[t3] - 4.0) < 1e-9


def test_matured_bond_removes_its_dv01_at_its_pre_maturity_duration():
    t0, t3 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-04-30")
    lad = _flow_ladder([(t0, "A", 100.0, 8.0), (t0, "C", 200.0, 0.4),
                        (t3, "A", 100.0, 8.0)])          # C matured away by t3
    f = flow_dv01(lad, months=3)
    assert abs(f[t3] - (-0.4)) < 1e-9


def test_reopened_bond_counts_only_the_incremental_par_at_current_ratio():
    t0, t3 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-04-30")
    lad = _flow_ladder([(t0, "A", 100.0, 8.0), (t3, "A", 150.0, 12.6)])   # +50 face
    f = flow_dv01(lad, months=3)
    ratio_t3 = 12.6 / 150.0
    assert abs(f[t3] - 50.0 * ratio_t3) < 1e-9


def test_no_flow_reported_when_the_lag_month_is_missing():
    t0 = pd.Timestamp("2020-01-31")
    lad = _flow_ladder([(t0, "A", 100.0, 8.0)])
    f = flow_dv01(lad, months=3)
    assert len(f) == 0


def test_market_value_is_par_times_invoice_price():
    mat = ASOF + pd.DateOffset(years=7, months=2)
    lad = _ladder([(mat, 2.5, 1000.0, "nominal"), (mat, 1.0, 500.0, "frn"),
                   (mat, 0.5, 300.0, "tips")])
    v = security_values(lad, ASOF, flat(0.04), real_curve=None)
    ytm = 2.0 * (np.exp(0.04 / 2.0) - 1.0)
    dirty = price_from_ytm(ASOF, mat, 2.5, ytm) + accrued_interest(ASOF, mat, 2.5)
    assert abs(v["mv_mn"].iloc[0] - 1000.0 * dirty / 100.0) < 1e-9
    assert v["mv_mn"].iloc[1] == 500.0                    # FRN at par
    assert v["mv_mn"].iloc[2] == 300.0                    # TIPS at par without a real curve


def test_tips_inflation_accretion_is_not_issuance():
    """Every TIPS's principal grows by the same reference-CPI ratio; only the reopening
    of D (+40 par, on top of its accretion) is supply."""
    t0, t1 = pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")
    rows = [(t0, c, 100.0, 9.0) for c in "ABCD"] + [(t1, c, 100.3, 9.0) for c in "ABC"] \
        + [(t1, "D", 140.3, 12.6)]
    lad = _flow_ladder(rows).assign(asset_class="tips")
    f = flow_dv01(lad, months=1)
    assert abs(f[t1] - 40.0 * 12.6 / 140.3) < 1e-9


# --------------------------------------------------------------------------- aggregate

def test_aggregate_joins_on_calendar_month_not_exact_date():
    """MSPD snapshots at true month-end; SOMA snapshots on whichever Wednesday closed
    the reporting month -- different exact dates for "the same month". Joining by exact
    date would union them into two NaN-riddled rows; aggregate() must produce one."""
    out = _flow_ladder([
        (pd.Timestamp("2020-01-31"), "A", 100.0, 8.0),
        (pd.Timestamp("2020-02-29"), "A", 100.0, 8.0),
    ]).assign(asset_class="nominal")
    held = _flow_ladder([
        (pd.Timestamp("2020-01-29"), "A", 40.0, 3.2),      # same MONTH, different day
        (pd.Timestamp("2020-02-26"), "A", 40.0, 3.2),
    ]).assign(asset_class="nominal")
    df = aggregate(out, held)
    assert len(df) == 2                                     # not 4
    jan = df.loc["2020-01-31"]
    assert jan.outstanding_dv01_mn == 8.0
    assert jan.soma_dv01_mn == 3.2                          # not NaN
