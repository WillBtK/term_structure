"""The expectations ladder: the anchor carries history, the public rung carries today."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro.inflation_expectations import MIN_OVERLAP, _bridge

IDX = pd.date_range("2015-01-01", periods=60, freq="MS")
ANCHOR = pd.Series(np.linspace(2.0, 2.4, 60), index=IDX)


def _rung(offset, start="2018-01-01", periods=60):
    i = pd.date_range(start, periods=periods, freq="MS")
    return pd.Series(np.linspace(2.2, 2.9, periods) + offset, index=i)


def test_the_rung_extends_past_the_anchor_and_the_join_is_calibrated_away():
    """The calibrated handover has to be far smaller than the raw one -- that is the
    whole job. It is not exactly zero, because the two series have different slopes as
    well as different levels, and only the level is matched."""
    rung = _rung(-0.5)
    out = _bridge(ANCHOR, rung, "rung", "XX")
    assert out.index.max() > ANCHOR.index.max()
    join = out.index.get_loc(ANCHOR.index.max())
    step = abs(out.iloc[join + 1] - out.iloc[join])
    raw = abs(rung[rung.index > ANCHOR.index.max()].iloc[0] - ANCHOR.iloc[-1])
    assert step < 0.25 * raw


def test_the_anchor_is_never_overwritten_inside_its_own_span():
    out = _bridge(ANCHOR, _rung(-0.5), "rung", "XX")
    pd.testing.assert_series_equal(out.loc[ANCHOR.index], ANCHOR, check_names=False)


def test_the_level_offset_is_the_mean_gap_over_the_overlap():
    rung = _rung(-0.5)
    out = _bridge(ANCHOR, rung, "rung", "XX")
    ov = ANCHOR.index.intersection(rung.index)
    off = float((ANCHOR[ov] - rung[ov]).mean())
    after = out[out.index > ANCHOR.index.max()]
    assert abs(after.iloc[0] - (rung[after.index[0]] + off)) < 1e-9


def test_a_rung_that_barely_overlaps_is_refused():
    thin = _rung(0.0, start="2019-11-01", periods=40)          # 2 months of overlap
    assert len(ANCHOR.index.intersection(thin.index)) < MIN_OVERLAP
    pd.testing.assert_series_equal(_bridge(ANCHOR, thin, "thin", "XX"), ANCHOR)


def test_a_rung_wholly_inside_the_anchor_changes_nothing():
    """The live case: while the proprietary snapshot is current the public rung sits
    idle, and only takes over once the anchor stops moving."""
    inside = _rung(0.0, start="2016-01-01", periods=24)
    pd.testing.assert_series_equal(_bridge(ANCHOR, inside, "inside", "XX"), ANCHOR)


def test_a_rung_that_only_predates_the_anchor_back_extends_it():
    early = _rung(-0.5, start="2010-01-01", periods=80)
    out = _bridge(ANCHOR, early, "early", "XX")
    assert out.index.min() < ANCHOR.index.min()
    assert out.index.max() == ANCHOR.index.max()
