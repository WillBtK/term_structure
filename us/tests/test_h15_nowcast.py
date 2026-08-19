"""Tests for the H.15 CMT nowcast extension (no network; synthetic panels)."""
import numpy as np
import pandas as pd
import pytest

from us import fetch_h15 as h15


def _panel(dates, tenors, base):
    """Constant-per-tenor panel (rows=dates) for building deterministic fixtures."""
    return pd.DataFrame({t: base[t] for t in tenors}, index=pd.DatetimeIndex(dates))


def test_increment_splice_anchors_to_gsw(monkeypatch):
    """nowcast(tau,t) = GSW(tau,t0) + [H15(tau,t) - H15(tau,t0)] — the GSW level is the
    anchor; only the H.15 change is transferred (so the par/zero/on-the-run wedge cancels)."""
    gsw = _panel(pd.date_range("2026-08-12", "2026-08-14"), [2.0, 10.0],
                 {2.0: 0.040, 10.0: 0.047})                       # GSW ends 08-14
    # H.15 sits 10bp BELOW GSW at 10y (on-the-run richness) and moves +5bp on 08-17
    hh = pd.DataFrame({2.0: [0.041, 0.041, 0.041, 0.041],
                       10.0: [0.046, 0.046, 0.046, 0.051]},
                      index=pd.to_datetime(["2026-08-12", "2026-08-13", "2026-08-14", "2026-08-17"]))
    monkeypatch.setattr(h15, "load_h15_panel", lambda inst="nominal", refresh=False: hh)
    ext, dates = h15.nowcast_extend(gsw, "nominal")
    assert dates == ["2026-08-17"]
    # 10y: GSW anchor 0.047 + H.15 change (+0.005) = 0.052 — NOT the raw H.15 level 0.051
    assert abs(ext.loc["2026-08-17", 10.0] - 0.052) < 1e-12
    # 2y: H.15 flat -> nowcast == GSW anchor
    assert abs(ext.loc["2026-08-17", 2.0] - 0.040) < 1e-12
    # history untouched, dates strictly increasing
    assert ext.loc["2026-08-14", 10.0] == 0.047
    assert ext.index.is_monotonic_increasing and ext.index.is_unique


def test_no_extension_when_h15_not_newer(monkeypatch):
    gsw = _panel(pd.date_range("2026-08-12", "2026-08-14"), [10.0], {10.0: 0.047})
    hh = _panel(pd.date_range("2026-08-12", "2026-08-14"), [10.0], {10.0: 0.046})
    monkeypatch.setattr(h15, "load_h15_panel", lambda inst="nominal", refresh=False: hh)
    ext, dates = h15.nowcast_extend(gsw, "nominal")
    assert dates == [] and ext.equals(gsw)


def test_ragged_when_tenor_absent(monkeypatch):
    """A grid tenor with no H.15 counterpart (e.g. 2y/3y real) stays NaN on nowcast rows."""
    gsw = _panel(pd.date_range("2026-08-06", "2026-08-07"), [3.0, 10.0],
                 {3.0: 0.010, 10.0: 0.024})
    hh = pd.DataFrame({10.0: [0.025, 0.025, 0.027]},                # only 10y in H.15
                      index=pd.to_datetime(["2026-08-06", "2026-08-07", "2026-08-10"]))
    monkeypatch.setattr(h15, "load_h15_panel", lambda inst="real", refresh=False: hh)
    ext, dates = h15.nowcast_extend(gsw, "real")
    assert dates == ["2026-08-10"]
    assert np.isnan(ext.loc["2026-08-10", 3.0])                    # 3y ragged
    assert abs(ext.loc["2026-08-10", 10.0] - (0.024 + 0.002)) < 1e-12


def test_stale_gsw_guard(monkeypatch):
    """If GSW is far behind the H.15 anchor, refuse to splice across the gap."""
    gsw = _panel(pd.date_range("2026-07-01", "2026-07-02"), [10.0], {10.0: 0.047})
    hh = _panel(pd.date_range("2026-07-01", "2026-08-17"), [10.0], {10.0: 0.05})
    monkeypatch.setattr(h15, "load_h15_panel", lambda inst="nominal", refresh=False: hh)
    ext, dates = h15.nowcast_extend(gsw, "nominal", max_gap_days=20)
    assert dates == [] and ext.equals(gsw)


def test_fetch_failure_is_inert(monkeypatch):
    def boom(inst="nominal", refresh=False):
        raise RuntimeError("network down")
    monkeypatch.setattr(h15, "load_h15_panel", boom)
    gsw = _panel(pd.date_range("2026-08-12", "2026-08-14"), [10.0], {10.0: 0.047})
    ext, dates = h15.nowcast_extend(gsw, "nominal")
    assert dates == [] and ext.equals(gsw)                         # GSW-only fallback
