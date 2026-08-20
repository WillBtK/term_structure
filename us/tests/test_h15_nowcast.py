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


# ---------------------------------------------------------------------------
# Treasury XML tail (the same CMT numbers, one business day earlier than FRED)
# ---------------------------------------------------------------------------

_XML = """<?xml version="1.0" encoding="utf-8" standalone="yes"?>
<feed xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns="http://www.w3.org/2005/Atom">
  <entry><content type="application/xml"><m:properties>
    <d:NEW_DATE m:type="Edm.DateTime">2026-08-18T00:00:00</d:NEW_DATE>
    <d:BC_2YEAR m:type="Edm.Double">4.19</d:BC_2YEAR>
    <d:BC_10YEAR m:type="Edm.Double">4.71</d:BC_10YEAR>
  </m:properties></content></entry>
  <entry><content type="application/xml"><m:properties>
    <d:NEW_DATE m:type="Edm.DateTime">2026-08-19T00:00:00</d:NEW_DATE>
    <d:BC_2YEAR m:type="Edm.Double">4.19</d:BC_2YEAR>
    <d:BC_10YEAR m:type="Edm.Double">4.65</d:BC_10YEAR>
  </m:properties></content></entry>
</feed>"""


class _Resp:
    def __init__(self, content):
        self.content = content.encode()

    def raise_for_status(self):
        pass


def test_treasury_month_parses_to_decimal(monkeypatch):
    """The <properties> wrapper is in the metadata namespace while the fields are in the
    dataservices one — getting that wrong parses to zero rows rather than raising."""
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_XML))
    got = h15._treasury_month("nominal", "202608")
    assert list(got.index) == [pd.Timestamp("2026-08-18"), pd.Timestamp("2026-08-19")]
    assert got.loc[pd.Timestamp("2026-08-19"), 10.0] == pytest.approx(0.0465)   # percent -> decimal
    assert 30.0 not in got.columns                                             # unquoted tenor omitted


def test_treasury_only_extends_never_revises():
    """Treasury may only ADD dates past FRED's last. An overlapping day keeps the FRED
    value even when the two differ, so a feed change shows up as a warning, not a silent
    rewrite of published history."""
    idx = pd.to_datetime(["2026-08-17", "2026-08-18"])
    fred = pd.DataFrame({2.0: [0.0419, 0.0419], 10.0: [0.0472, 0.0471]}, index=idx)
    tail = pd.DataFrame({2.0: [0.0500, 0.0419], 10.0: [0.0500, 0.0465]},       # 08-18 disagrees
                        index=pd.to_datetime(["2026-08-18", "2026-08-19"]))
    out = h15._splice_treasury(fred, tail, "nominal")
    assert out.index.max() == pd.Timestamp("2026-08-19")
    assert out.loc[pd.Timestamp("2026-08-18"), 10.0] == pytest.approx(0.0471)  # FRED kept
    assert out.loc[pd.Timestamp("2026-08-19"), 10.0] == pytest.approx(0.0465)  # Treasury added


def test_treasury_outage_leaves_panel_untouched():
    """Treasury is an accelerator, not a dependency: an empty tail degrades to FRED-only."""
    idx = pd.to_datetime(["2026-08-17", "2026-08-18"])
    fred = pd.DataFrame({10.0: [0.0472, 0.0471]}, index=idx)
    assert h15._splice_treasury(fred, pd.DataFrame(), "nominal").equals(fred)


def test_treasury_tail_survives_a_failing_month(monkeypatch):
    """One bad month must not take out the other — the feed is fetched a month at a time."""
    def flaky(*a, **k):
        if k.get("params", {}).get("field_tdr_date_value_month") == "202607":
            raise RuntimeError("boom")
        return _Resp(_XML)
    import requests
    monkeypatch.setattr(requests, "get", flaky)
    got = h15._treasury_tail("nominal", months=["202607", "202608"])
    assert list(got.index) == [pd.Timestamp("2026-08-18"), pd.Timestamp("2026-08-19")]
