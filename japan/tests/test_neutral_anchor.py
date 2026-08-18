"""Tests for the Japan neutral-anchor degradation ladder (no network; synthetic CSVs)."""
import numpy as np
import pandas as pd
import pytest

from japan import neutral_anchor as na


def _write(d, name, rows):
    p = d / na.FILES[name]
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(p, index=False)


def test_no_files_is_inert(tmp_path):
    vals, meta = na.build_nominal_neutral(pd.date_range("2020-01-31", periods=12, freq="M"),
                                          dir=tmp_path)
    assert meta["source"] == "none"
    assert np.all(np.isnan(vals))
    assert not na.has_anchor(tmp_path)


def test_bridge_from_ferreira_plus_cpi(tmp_path):
    # r* = 0.5%, CPI = 1.5% -> nominal bridge 2.0% (decimal 0.02), carried forward.
    _write(tmp_path, "rstar", [["2019-06-30", 0.5], ["2024-06-30", 0.5]])
    _write(tmp_path, "cpi", [["2019-04-30", 1.5], ["2024-04-30", 1.5]])
    idx = pd.date_range("2020-01-31", "2024-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "bridge" and meta["pi_e"] == "consensus_cpi"
    assert np.allclose(vals, 0.02, atol=1e-9)      # 0.5 + 1.5 = 2.0% -> 0.02


def test_direct_shortrate_overrides_and_calibrates(tmp_path):
    # bridge would be 2.0%; the direct short-rate survey says 1.6% over the overlap ->
    # the survey is used directly where present (and the bridge is level-shifted to match).
    _write(tmp_path, "rstar", [["2018-06-30", 0.5]])
    _write(tmp_path, "cpi", [["2018-04-30", 1.5]])
    _write(tmp_path, "shortrate", [["2018-10-31", 1.6], ["2022-10-31", 1.6]])
    idx = pd.date_range("2018-01-31", "2022-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "consensus_shortrate+bridge"
    s = pd.Series(vals, index=idx)
    assert abs(s.loc["2020-06-30"] - 0.016) < 1e-6     # inside the survey span -> 1.6%
    # after r*/CPI start but before the short-rate survey, the bridge (calibrated to the
    # survey's 1.6% over the overlap) applies, not the raw 2.0%
    assert abs(s.loc["2018-07-31"] - 0.016) < 2e-3


def test_implied_real_when_no_ferreira(tmp_path):
    """With no Ferreira file but both a short-rate and a CPI survey, the real leg is
    backed out from the overlap (r ~= shortrate - CPI) and held through history."""
    _write(tmp_path, "cpi", [["2016-04-30", 1.0], ["2024-04-30", 1.0]])
    _write(tmp_path, "shortrate", [["2018-10-31", 0.6], ["2024-10-31", 0.6]])
    idx = pd.date_range("2016-01-31", "2024-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["r_star"] == "implied_from_overlap"
    s = pd.Series(vals, index=idx)
    # implied real = mean(0.6 - 1.0) = -0.4; pre-survey bridge = CPI + real = 1.0 - 0.4 = 0.6%
    assert abs(s.loc["2016-06-30"] - 0.006) < 5e-4
    assert abs(s.loc["2020-06-30"] - 0.006) < 5e-4   # inside survey span -> direct 0.6%


def test_tankan_tail_extends_cpi(tmp_path):
    # CPI ends 2023; Tankan continues to 2025 -> the tail (level-bridged) maintains pi^e.
    _write(tmp_path, "rstar", [["2018-06-30", 0.0]])
    _write(tmp_path, "cpi", [["2018-04-30", 1.0], ["2023-04-30", 1.0]])
    _write(tmp_path, "tankan", [["2018-06-30", 1.2], ["2023-06-30", 1.2], ["2025-06-30", 1.8]])
    idx = pd.date_range("2018-01-31", "2025-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert "tankan_tail" in meta["pi_e"]
    s = pd.Series(vals, index=idx)
    # 2022 uses CPI (1.0%) + r*(0) = 1.0%; 2025 uses Tankan tail bridged to CPI:
    # offset = mean(CPI - Tankan) over overlap = (1.0 - 1.2) = -0.2, so 2025 pi^e = 1.8-0.2 = 1.6%
    assert abs(s.loc["2022-06-30"] - 0.010) < 1e-6
    assert abs(s.loc["2025-06-30"] - 0.016) < 1e-6
