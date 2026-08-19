"""Tests for the BoC benchmark-yield nowcast + benchmark-roll guard (no network)."""
import numpy as np
import pandas as pd
import pytest

from canada import fetch_benchmark as fb


def _bm(dates, data):
    return pd.DataFrame(data, index=pd.DatetimeIndex(dates))


def test_deroll_neutralises_single_tenor_jump():
    """A benchmark roll moves ONE tenor while the curve sits still — it must be replaced
    by the curve-median change, not carried into the splice."""
    # curve up ~5bp across the board; the 3y "rolls" -13bp (the 2023-05-18 signature)
    chg = _bm(["2026-08-03"], {2.0: [0.0004], 3.0: [-0.0013], 5.0: [0.0007],
                               7.0: [0.0008], 10.0: [0.0009], 30.0: [0.0008]})
    out, n = fb._deroll(chg)
    assert n == 1
    assert out.loc["2026-08-03", 3.0] == pytest.approx(chg.loc["2026-08-03"].median())
    # untouched tenors keep their real moves
    for t in (2.0, 5.0, 7.0, 10.0, 30.0):
        assert out.loc["2026-08-03", t] == pytest.approx(chg.loc["2026-08-03", t])


def test_deroll_keeps_genuine_parallel_moves():
    """A real market move (whole curve shifts) must NOT be flagged — otherwise the
    nowcast would flatten actual information."""
    chg = _bm(["2026-08-03"], {2.0: [-0.0009], 3.0: [-0.0010], 5.0: [-0.0011],
                               7.0: [-0.0011], 10.0: [-0.0012], 30.0: [-0.0012]})
    out, n = fb._deroll(chg)
    assert n == 0
    pd.testing.assert_frame_equal(out, chg)


def test_deroll_ignores_small_wiggles():
    """A tenor that is a statistical outlier but tiny in bp is left alone (min_bp floor),
    so ordinary curve-shape noise is not scrubbed."""
    # tightly clustered curve (huge modified-z for the odd one) but only ~3bp away —
    # below the min_bp floor, so the guard must leave it alone
    chg = _bm(["2026-08-03"], {2.0: [0.005000], 3.0: [0.005010], 5.0: [0.005020],
                               7.0: [0.005010], 10.0: [0.004700], 30.0: [0.005000]})
    out, n = fb._deroll(chg)
    assert n == 0


def test_nowcast_is_increment_splice_anchored_to_zero_curve(monkeypatch):
    """nowcast(t) = ZC(t0) + cumulative benchmark change — levels come from the zero
    curve, only changes come from the benchmarks (their levels differ by construction)."""
    zc = _bm(pd.date_range("2026-07-27", periods=3, freq="B"),
             {2.0: [0.030] * 3, 10.0: [0.037] * 3})          # ZC ends 2026-07-29
    # benchmark levels sit ~50bp BELOW the zero curve (par vs zero) and rise 5bp/day
    bm = _bm(pd.date_range("2026-07-29", periods=3, freq="B"),
             {2.0: [0.025, 0.0255, 0.0260], 10.0: [0.032, 0.0325, 0.0330]})
    monkeypatch.setattr(fb, "load_benchmark_panel", lambda refresh=False: bm)
    ext, dates = fb.nowcast_extend(zc)
    assert dates == ["2026-07-30", "2026-07-31"]
    # +5bp then +10bp on the ZC level — NOT the benchmark's own level
    assert ext.loc["2026-07-30", 2.0] == pytest.approx(0.0305)
    assert ext.loc["2026-07-31", 2.0] == pytest.approx(0.0310)
    assert ext.loc["2026-07-31", 10.0] == pytest.approx(0.0380)
    # history untouched
    assert ext.loc["2026-07-29", 2.0] == pytest.approx(0.030)
    assert ext.index.is_monotonic_increasing and ext.index.is_unique


def test_roll_inside_window_does_not_corrupt_later_days(monkeypatch):
    """The splice is cumulative, so an unguarded roll would bias EVERY later nowcast day.
    With the guard, a roll on day 1 leaves day 2 correct."""
    zc = _bm(["2026-07-29"], {2.0: [0.030], 3.0: [0.031], 5.0: [0.032],
                              7.0: [0.033], 10.0: [0.034], 30.0: [0.035]})
    # day 1: whole curve +5bp but the 3y benchmark "rolls" -20bp; day 2: +5bp everywhere
    bm = _bm(pd.date_range("2026-07-29", periods=3, freq="B"), {
        2.0: [0.025, 0.0255, 0.0260], 3.0: [0.026, 0.0240, 0.0245],
        5.0: [0.027, 0.0275, 0.0280], 7.0: [0.028, 0.0285, 0.0290],
        10.0: [0.029, 0.0295, 0.0300], 30.0: [0.030, 0.0305, 0.0310]})
    monkeypatch.setattr(fb, "load_benchmark_panel", lambda refresh=False: bm)
    ext, dates = fb.nowcast_extend(zc)
    assert len(dates) == 2
    # the 3y tracks the curve (+5bp, +10bp) instead of inheriting the -20bp roll step
    assert ext.loc[dates[0], 3.0] == pytest.approx(0.0315, abs=1e-6)
    assert ext.loc[dates[1], 3.0] == pytest.approx(0.0320, abs=1e-6)


def test_missing_tenors_stay_ragged(monkeypatch):
    """Grid points with no benchmark counterpart (0.25-1y, 20y) must stay NaN, not be
    filled or dropped — the EKF masks them per date."""
    zc = _bm(["2026-07-29"], {0.25: [0.028], 2.0: [0.030], 20.0: [0.036], 10.0: [0.034]})
    bm = _bm(pd.date_range("2026-07-29", periods=2, freq="B"),
             {2.0: [0.025, 0.0255], 10.0: [0.029, 0.0295]})
    monkeypatch.setattr(fb, "load_benchmark_panel", lambda refresh=False: bm)
    ext, dates = fb.nowcast_extend(zc)
    assert len(dates) == 1
    assert np.isnan(ext.loc[dates[0], 0.25]) and np.isnan(ext.loc[dates[0], 20.0])
    assert ext.loc[dates[0], 2.0] == pytest.approx(0.0305)


def test_inert_on_stale_gap_and_on_failure(monkeypatch):
    zc = _bm(["2026-06-01"], {2.0: [0.030], 10.0: [0.034]})
    bm = _bm(pd.date_range("2026-06-01", periods=60, freq="B"), {2.0: 0.025, 10.0: 0.029})
    monkeypatch.setattr(fb, "load_benchmark_panel", lambda refresh=False: bm)
    ext, dates = fb.nowcast_extend(zc, max_gap_days=40)      # gap too long -> refuse
    assert dates == [] and ext.equals(zc)

    def boom(refresh=False):
        raise RuntimeError("valet down")
    monkeypatch.setattr(fb, "load_benchmark_panel", boom)
    ext2, dates2 = fb.nowcast_extend(zc)
    assert dates2 == [] and ext2.equals(zc)                  # zero-curve-only fallback
