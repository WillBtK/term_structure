"""Tests for the Bundesbank BBSIS CSV parser + reconstruction wiring (no network)."""
import numpy as np
import pandas as pd
import pytest

from germany import config as cfg
from germany.fetch_bundesbank import parse_series

# Faithful-but-compact fixture mirroring the real BBSIS CSV: metadata preamble
# then YYYY-MM,value,flag rows (monthly), including an empty value.
FIXTURE = (
    '"",BBSIS.M.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A,..._FLAGS\n'
    '"",Term structure ... / Parameter Beta0 / monthly data,\n'
    "BBK_UNIT_ENG,,\n"
    "Comment (in english),,\n"
    "Decimals,5,\n"
    "1972-09,8.58387,\n"
    "1972-10,8.52889,\n"
    "1972-11,,\n"          # missing value -> NaN
    "2026-07,4.05875,\n"
)


def test_parse_series_skips_preamble_and_coerces_missing():
    s = parse_series(FIXTURE, freq="M")
    assert s.loc["1972-09-30"] == pytest.approx(8.58387)
    assert s.loc["1972-10-31"] == pytest.approx(8.52889)
    assert np.isnan(s.loc["1972-11-30"])
    assert s.loc["2026-07-31"] == pytest.approx(4.05875)
    # month-end timestamps, sorted
    assert s.index.is_monotonic_increasing
    assert str(s.index[0].date()) == "1972-09-30"


def test_series_keys_wellformed():
    # A representative parameter and spot key match the discovered BBSIS format.
    assert cfg.param_series_key("BETA0", "M") == \
        "M.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A"
    assert cfg.spot_series_key(10.0, "M") == \
        "M.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A"


def test_reconstruct_from_bundesbank_params_matches_manual():
    # Reconstruction uses the shared Svensson math; a single-date sanity check.
    from us.reconstruct import svensson_yield, reconstruct_grid
    params = pd.DataFrame(
        {"BETA0": [4.05], "BETA1": [-1.80], "BETA2": [28.2], "BETA3": [-29.7],
         "TAU1": [1.63], "TAU2": [1.75]},
        index=[pd.Timestamp("2026-07-31")])
    panel = reconstruct_grid(params, np.array([2.0, 10.0]),
                             mask_by_availability=False)
    expect2 = svensson_yield(2.0, 4.05, -1.80, 28.2, -29.7, 1.63, 1.75) / 100.0
    assert panel.iloc[0][2.0] == pytest.approx(expect2, abs=1e-12)


# ---------------------------------------------------------------------------
# The SDMX API host is fronted by a bot challenge that answers automated clients
# with a redirect to an HTML verification page — a 200 carrying no observations.
# _download_csv must recognise that as a failure and fall through to the
# time-series-database mirror rather than handing an empty frame downstream.
# ---------------------------------------------------------------------------
CHALLENGE_HTML = (
    '<!doctype html><html><head><title>Enodia Verification</title>'
    '<script src="/.enodia/static/sha256.js"></script></head><body></body></html>'
)


def test_download_falls_back_when_the_api_returns_no_observations(monkeypatch):
    from germany import fetch_bundesbank as fb

    class _Resp:
        content = CHALLENGE_HTML.encode()
        def raise_for_status(self): pass

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fb, "_download_tsdb", lambda key, timeout=60: FIXTURE)
    out = fb._download_csv("M.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A")
    assert out is FIXTURE
    assert not fb._has_observations(CHALLENGE_HTML)
    assert fb._has_observations(FIXTURE)


def test_download_keeps_the_api_response_when_it_carries_data(monkeypatch):
    from germany import fetch_bundesbank as fb

    class _Resp:
        content = FIXTURE.encode()
        def raise_for_status(self): pass

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(fb, "_download_tsdb",
                        lambda key, timeout=60: pytest.fail("mirror must not be used"))
    assert fb._download_csv("M.I.ZST.B0.EUR.S1311.B.A604._Z.R.A.A._Z._Z.A") == FIXTURE


# ------------------------------------------ one stale parameter must not truncate the curve

def _csv(last: str) -> str:
    days = pd.bdate_range("2026-07-01", last)
    return "series,meta\n" + "\n".join(f"{d.date()},1.25,"for d in days) + "\n"


def _patch(monkeypatch, tmp_path, responses):
    from germany import config as cfg
    import germany.fetch_bundesbank as fb
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "PARAMS_PARQUET_PATH", tmp_path / "params.parquet")
    calls = []
    def fake(key, timeout=60, attempts=3):
        name = next(n for n in cfg.PARAM_COLUMNS if cfg.param_series_key(n, "D") == key)
        calls.append(name)
        seq = responses[name]
        return _csv(seq.pop(0) if len(seq) > 1 else seq[0])
    monkeypatch.setattr(fb, "_download_csv", fake)
    return calls


def test_a_parameter_from_the_lagging_mirror_is_refetched(monkeypatch, tmp_path):
    """2026-10-01: BETA0 fell back to the mirror, which ended 2026-08-03, and cut the
    whole curve back two months. A re-fetch that comes back current must be used."""
    from germany import config as cfg
    from germany.fetch_bundesbank import load_params
    resp = {n: ["2026-09-30"] for n in cfg.PARAM_COLUMNS}
    resp["BETA0"] = ["2026-08-03", "2026-09-30"]
    calls = _patch(monkeypatch, tmp_path, resp)
    p = load_params(freq="D", refresh=True)
    assert all(p[c].last_valid_index() == pd.Timestamp("2026-09-30") for c in p.columns)
    assert calls.count("BETA0") == 2


def test_a_parameter_that_stays_stale_fails_loudly(monkeypatch, tmp_path):
    from germany import config as cfg
    from germany.fetch_bundesbank import load_params
    resp = {n: ["2026-09-30"] for n in cfg.PARAM_COLUMNS}
    resp["BETA0"] = ["2026-08-03"]
    _patch(monkeypatch, tmp_path, resp)
    with pytest.raises(RuntimeError, match="BETA0 ends 2026-08-03"):
        load_params(freq="D", refresh=True)


def test_a_one_day_holiday_difference_is_not_lag(monkeypatch, tmp_path):
    from germany.fetch_bundesbank import _lagging
    idx = pd.bdate_range("2026-09-01", "2026-09-30")
    df = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
    df.loc[idx[-1], "B"] = float("nan")
    assert _lagging(df) == {}
