"""The browser payload: the EWMA sufficient statistic, and schema completeness."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro.fetch_core_cpi import UNCERTAINTY_HALFLIFE as HL
from scenarios.build_sheet import OUT, annual

MARKETS = ("US", "UK", "DE", "JP")


def test_the_ewma_state_is_a_sufficient_statistic_for_the_scenario_path():
    """The sheet ships two numbers instead of seventy years of history. Continuing
    the recursion from them must reproduce a full recomputation -- otherwise the
    browser's uncertainty driver silently diverges from the Python engine's."""
    rng = np.random.default_rng(0)
    hist = pd.Series(2 + rng.normal(0, 1.5, 600),
                     index=pd.date_range("1976-01-01", periods=600, freq="MS"))
    m = hist.ewm(halflife=HL).mean()
    v = ((hist - m) ** 2).ewm(halflife=HL).mean()
    m0, v0, lam = float(m.iloc[-1]), float(v.iloc[-1]), 0.5 ** (1.0 / HL)

    for path in ([2.3] * 36, [6.0] * 12 + [1.0] * 12 + [5.0] * 12, [2.0] * 120):
        full = pd.concat([hist, pd.Series(path, index=pd.date_range(
            hist.index[-1] + pd.DateOffset(months=1), periods=len(path), freq="MS"))])
        mm = full.ewm(halflife=HL).mean()
        exact = float(np.sqrt(((full - mm) ** 2).ewm(halflife=HL).mean()).iloc[-1])
        mj, vj = m0, v0
        for x in path:
            mj = lam * mj + (1 - lam) * x
            vj = lam * vj + (1 - lam) * (x - mj) ** 2
        assert np.sqrt(vj) == pytest.approx(exact, abs=1e-6)


def test_annual_collapses_to_calendar_year_means():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    s = pd.Series(np.arange(24, dtype=float), index=idx)
    a = annual(s, start="2020")
    assert a["years"] == [2020, 2021]
    assert a["v"] == [pytest.approx(5.5), pytest.approx(17.5)]


@pytest.mark.skipif(not Path(OUT).is_file(), reason="payload not built")
def test_the_committed_payload_carries_everything_the_page_reads():
    p = json.loads(Path(OUT).read_text())
    assert set(p) >= {"halflife", "tenor", "lam", "asof", "markets"}
    assert p["lam"] == pytest.approx(0.5 ** (1.0 / p["halflife"]))
    for c in MARKETS:
        m = p["markets"][c]
        assert set(m) >= {"y_today", "tp_today", "drivers", "duration", "ewma",
                          "debt_proj", "deficit_proj", "hist", "model"}
        # every market carries its official 5y-ahead projected deficit
        want = {"expected_inflation", "inflation_uncertainty",
                "price_insensitive", "debt_gdp", "fiscal"}
        assert set(m["drivers"]) == want
        assert m["deficit_proj"]            # CBO / OBR / AMECO / CAO default path
        assert 4.0 < m["duration"] < 10.0                # a 10y par bond
        assert m["ewma"]["v"] >= 0.0
        for k in ("core", "short", "ten"):
            h = m["hist"][k]
            assert len(h["years"]) == len(h["v"]) > 10
        mod = m["model"]
        if c == "US":
            assert mod["kind"] == "us"
            assert set(mod["coef"]) == {"expected_inflation", "inflation_uncertainty",
                                        "price_insensitive"}
            assert set(mod["adjust"]) == set(mod["base"]) == {"fiscal"}
            assert 0.35 <= mod["adjust"]["fiscal"] <= 0.66   # the measured window
        else:
            assert mod["kind"] == "spread" and 0.0 <= mod["beta_us"] <= 1.5
            assert not set(mod["coef"]) & set(mod["adjust"])   # fitted xor adjusted
            # the three core channels are fitted or borrowed; fiscal is always the
            # borrowed US sensitivity on the relative projected deficit, and the
            # debt stock enters nowhere
            core = {"expected_inflation", "inflation_uncertainty", "price_insensitive"}
            assert set(mod["coef"]) | set(mod["adjust"]) == core | {"fiscal"}
            assert "fiscal" in mod["adjust"] and "fiscal" in mod["base"]
            assert mod["adjust"]["fiscal"] == p["markets"]["US"]["model"]["adjust"]["fiscal"]
            assert "debt_gdp" not in mod["coef"] and "debt_gdp" not in mod["adjust"]


@pytest.mark.skipif(not Path(OUT).is_file(), reason="payload not built")
def test_payload_stays_small_enough_to_inline():
    assert len(Path(OUT).read_text()) < 200_000
