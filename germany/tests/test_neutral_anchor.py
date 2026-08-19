"""Germany neutral-anchor ladder (no network; synthetic CSVs).

Mirrors the Japan suite — the ladder itself is shared (`model.neutral_anchor`), so what
is tested here is Germany's wiring: its filenames, its rung labels, and the activation
gate that keeps the anchor off until the Consensus inflation leg extends coverage back
to 1990.
"""
import numpy as np
import pandas as pd
import pytest

from germany import neutral_anchor as na


@pytest.fixture(autouse=True)
def _no_smoothing(monkeypatch):
    """These tests assert exact levels on minimal constant series, so isolate the ladder
    from the RW endpoint smoother (exercised in the Japan suite, same code)."""
    monkeypatch.setattr(na, "SMOOTH", False)


def _write(d, name, rows):
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(d / na.FILES[name], index=False)


def test_no_files_is_inert(tmp_path):
    vals, meta = na.build_nominal_neutral(pd.date_range("2020-01-31", periods=12, freq="M"),
                                          dir=tmp_path)
    assert meta["source"] == "none"
    assert np.all(np.isnan(vals))
    assert not na.has_anchor(tmp_path) and not na.is_enabled(tmp_path)


def test_bridge_is_rstar_plus_expected_inflation(tmp_path):
    """The public rung: n = Ferreira EA r* + pi^e, in percent, returned as a decimal."""
    _write(tmp_path, "rstar", [("2019-12-31", 0.5), ("2020-12-31", 0.5)])
    _write(tmp_path, "cpi", [("2019-12-31", 1.8), ("2020-12-31", 1.8)])
    idx = pd.date_range("2020-01-31", periods=6, freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "bridge" and meta["pi_e"] == "consensus_cpi"
    assert meta["r_star"] == "ferreira"
    assert np.allclose(vals, 0.023)                       # (0.5 + 1.8)% -> decimal


def test_direct_shortrate_overrides_and_calibrates_the_bridge(tmp_path):
    """Rung 1 wins inside its staleness window, and the bridge is level-matched to it so
    the two rungs join without a step when the survey lapses."""
    _write(tmp_path, "rstar", [("2019-12-31", 0.5), ("2021-12-31", 0.5)])
    _write(tmp_path, "cpi", [("2019-12-31", 1.8), ("2021-12-31", 1.8)])
    _write(tmp_path, "shortrate", [("2020-01-31", 2.8), ("2020-06-30", 2.8)])
    idx = pd.date_range("2020-01-31", periods=6, freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "consensus_shortrate+bridge"
    assert meta["calib_offset_pct"] == pytest.approx(0.5)   # 2.8 - (0.5+1.8)
    assert np.allclose(vals, 0.028)


def test_ecb_spf_extends_the_consensus_inflation_leg(tmp_path):
    """Past the last Consensus release the inflation leg continues on the ECB SPF,
    level-bridged over the overlap — the mechanism that keeps a frozen snapshot live."""
    _write(tmp_path, "rstar", [("2020-12-31", 0.5), ("2024-12-31", 0.5)])
    _write(tmp_path, "cpi", [("2020-12-31", 1.8), ("2021-12-31", 1.8)])
    _write(tmp_path, "extender", [("2021-12-31", 1.5), ("2023-12-31", 1.5)])
    idx = pd.date_range("2021-01-31", "2023-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["pi_e"] == "consensus_cpi+ecb_spf_tail"
    # the SPF tail is shifted by the +0.3 overlap offset, so the level does not step
    assert np.allclose(vals[-1], 0.023, atol=1e-9)


def test_target_is_the_last_resort_inflation_anchor(tmp_path):
    _write(tmp_path, "rstar", [("2020-12-31", 0.5), ("2021-12-31", 0.5)])
    idx = pd.date_range("2021-01-31", periods=4, freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["pi_e"] == "ecb_target"
    assert np.allclose(vals, (0.5 + na.ECB_TARGET_PCT) / 100.0)


def test_activation_waits_for_the_consensus_inflation_leg(tmp_path):
    """The public rungs alone only reach 1999, and the production sample starts 1984 —
    enabling on that coverage would seam the sample at 1999. So a buildable anchor is
    not yet an ACTIVE one."""
    _write(tmp_path, "rstar", [("2020-12-31", 0.5), ("2021-12-31", 0.5)])
    _write(tmp_path, "extender", [("2020-12-31", 1.8), ("2021-12-31", 1.8)])
    assert na.has_anchor(tmp_path)          # buildable
    assert not na.is_enabled(tmp_path)      # but not applied
    _write(tmp_path, "cpi", [("1990-12-31", 2.5), ("2021-12-31", 1.8)])
    assert na.is_enabled(tmp_path)          # Consensus CPI extends coverage -> on
