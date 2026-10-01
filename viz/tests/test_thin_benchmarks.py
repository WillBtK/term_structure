"""The Atlas's external-benchmark panel ships each model at its native frequency.

The regression: all four sources publish daily, but the payload kept only the last
observation of each month, so a daily line of ours sat beside four monthly ones.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _daily(start, end, maturities=(2.0, 10.0), seed=0):
    days = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"date": d, "maturity": m, "term_premium": 0.01 + 1e-4 * rng.standard_normal()}
                         for d in days for m in maturities])


@pytest.fixture
def countries(tmp_path, monkeypatch):
    from validation import benchmarks as bm
    from validation import config as vcfg
    from validation import sf_fed
    frames = {"acm": _daily("2026-06-01", "2026-09-30", seed=1),
              "kim_wright": _daily("2026-06-01", "2026-09-30", seed=2),
              "dkw": _daily("2026-06-01", "2026-09-30", maturities=(2.0, 10.0), seed=3),
              "cr": _daily("2026-06-01", "2026-09-30", seed=4)}
    # a same-day duplicate: the later row must win, and must not add a date
    dup = frames["acm"].iloc[[-1]].assign(term_premium=0.0321)
    frames["acm"] = pd.concat([frames["acm"], dup], ignore_index=True)
    for attr, key in (("ACM_PARQUET_PATH", "acm"), ("KIM_WRIGHT_PARQUET_PATH", "kim_wright"),
                      ("DKW_PARQUET_PATH", "dkw"), ("SF_FED_PARQUET_PATH", "cr")):
        f = tmp_path / f"{key}.parquet"; f.write_bytes(b"x")
        monkeypatch.setattr(vcfg, attr, f)
    monkeypatch.setattr(bm, "load_acm", lambda refresh=False: frames["acm"])
    monkeypatch.setattr(bm, "load_kim_wright", lambda refresh=False: frames["kim_wright"])
    monkeypatch.setattr(bm, "load_dkw", lambda refresh=False: frames["dkw"])
    monkeypatch.setattr(sf_fed, "load_sf_fed_series", lambda refresh=False: frames["cr"])
    c = {"US": {"instruments": {"nominal": {}}}}
    from viz.thin_daily import _attach_us_benchmarks
    _attach_us_benchmarks(c)
    return c["US"]["instruments"]["nominal"]["benchmarks"]


def test_every_source_keeps_every_trading_day(countries):
    n_days = len(pd.bdate_range("2026-06-01", "2026-09-30"))
    for key, block in countries.items():
        assert len(block["dates"]) == n_days, key            # not 4 month-ends
        assert len(block["x"]) == n_days
        assert all(len(v) == n_days for v in block["tp"].values())


def test_a_same_day_revision_keeps_the_last_value_without_adding_a_date(countries):
    acm = countries["acm"]
    assert acm["dates"][-1] == "2026-09-30"
    assert acm["tp"]["10"][-1] == pytest.approx(3.21)


def test_values_arrive_in_percent(countries):
    assert all(0.5 < v < 1.5 for v in countries["cr"]["tp"]["10"] if v is not None)
