"""Tests for the regime/shadow splice join + continuity check (synthetic legs)."""
import numpy as np
import pandas as pd
import pytest

from us import splice as sp

GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def _months(dates):
    # months since a fixed epoch, so a value is a function of the DATE (not the
    # positional index) -> two legs agree exactly on shared dates.
    return np.array([(d.year - 1960) * 12 + d.month for d in dates], dtype=float)


def _factor_frame(dates, offset=0.0):
    t = _months(dates)
    L = 0.05 + 0.001 * np.sin(t / 12) + offset
    S = -0.01 + 0.0005 * np.cos(t / 9) + offset
    C = 0.002 + 0.0003 * np.sin(t / 6) + offset
    return pd.DataFrame({"L": L, "S": S, "C": C}, index=dates)


def _tp_frame(dates, offset=0.0):
    t = _months(dates)[:, None]
    vals = 0.01 + 1e-5 * t + 0.0 * GRID[None, :] + offset
    return pd.DataFrame(vals, index=dates, columns=GRID)


def _legs(overlap_agrees=True):
    rdates = pd.date_range("1990-01-31", "2008-12-31", freq="M")
    sdates = pd.date_range("2000-01-31", "2020-12-31", freq="M")
    off = 0.0 if overlap_agrees else 0.02
    regime = {"dates": rdates, "grid": GRID,
              "factors": _factor_frame(rdates), "term_premium": _tp_frame(rdates)}
    shadow = {"dates": sdates, "grid": GRID,
              "factors": _factor_frame(sdates, offset=off),
              "term_premium": _tp_frame(sdates, offset=off),
              "shadow": pd.DataFrame({"date": sdates,
                                      "shadow_short_rate": _factor_frame(sdates, off)["L"].values
                                      + _factor_frame(sdates, off)["S"].values})}
    return regime, shadow


def test_splice_join_is_continuous_and_partitioned():
    regime, shadow = _legs()
    joined = sp.splice(regime, shadow, splice_date=sp.SPLICE_DATE)
    # index sorted, unique, spans both legs
    assert joined.index.is_monotonic_increasing
    assert joined.index.is_unique
    assert joined.index.min() == regime["factors"].index.min()
    assert joined.index.max() == shadow["factors"].index.max()
    # every date strictly before the seam comes from the regime leg; from the seam
    # on, from the shadow leg (partition with no gap/overlap)
    assert (joined.index[joined.index < sp.SPLICE_DATE]).isin(regime["factors"].index).all()
    assert (joined.index[joined.index >= sp.SPLICE_DATE]).isin(shadow["factors"].index).all()
    # schema
    for c in ["L", "S", "C", "shadow_short_rate"]:
        assert c in joined.columns
    assert "tp_10y" in joined.columns and "tp_0.25y" in joined.columns
    assert joined.notna().all().all()


def test_continuity_check_reports_agreement_when_legs_agree():
    regime, shadow = _legs(overlap_agrees=True)
    c = sp.continuity_check(regime, shadow)
    assert c["overlap_n"] > 50
    for f in ["L", "S", "C"]:
        assert c["per_factor"][f]["corr"] > 0.99
        assert c["per_factor"][f]["max_abs_gap_bp"] < 1e-6
    assert all(abs(g) < 1e-6 for g in c["seam_gap_LSC_bp"])


def test_continuity_check_flags_disagreement():
    regime, shadow = _legs(overlap_agrees=False)
    c = sp.continuity_check(regime, shadow)
    # a constant 2% factor offset -> ~200 bp seam gap on each factor
    assert max(abs(g) for g in c["seam_gap_LSC_bp"]) > 100
