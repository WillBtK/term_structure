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
