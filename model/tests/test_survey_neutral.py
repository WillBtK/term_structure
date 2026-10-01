"""The survey-only endpoint builder (`build_survey_neutral`) — the production
construction for every anchored market since the Ferreira review.

The contract these tests pin: surveys are the ONLY source of endpoint variation;
history before the first survey release stays NaN (constant-theta); the public
extension is level-bridged and refused without a real overlap; and the two
carry-forward modes past staleness — freeze, or glide to a communicated terminal —
join the survey era without a step.
"""
import numpy as np
import pandas as pd
import pytest

from model.neutral_anchor import AnchorSpec, build_survey_neutral


def _spec(d, smooth=False, files=None):
    return AnchorSpec(market="XX", dir=d, files=files or {"shortrate": "sr.csv",
                                                          "shortrate_ext": "ext.csv"},
                      target_pct=2.0, smooth=smooth, survey_label="consensus",
                      shortrate_ext_label="mps")


def _write(d, name, rows):
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(d / name, index=False)


def test_nan_before_the_survey_era_and_direct_inside_it(tmp_path):
    """Pre-survey history is NOT rewritten — that is the whole point of the re-design."""
    _write(tmp_path, "sr.csv", [("2017-01-31", 2.5), ("2020-01-31", 2.5)])
    idx = pd.date_range("2015-01-31", "2019-12-31", freq="M")
    v, meta = build_survey_neutral(idx, _spec(tmp_path), dir=tmp_path)
    s = pd.Series(v, index=idx)
    assert s[idx < "2017-01-31"].isna().all()
    assert np.allclose(s["2017-02-28":].dropna(), 0.025)
    assert meta["source"] == "consensus_shortrate"


def test_no_model_rstar_enters_even_when_a_file_exists(tmp_path):
    """An rstar file on disk is inert: the builder never reads it."""
    _write(tmp_path, "sr.csv", [("2017-01-31", 2.5), ("2020-01-31", 2.5)])
    _write(tmp_path, "rstar.csv", [("2000-01-31", -0.9), ("2020-01-31", -0.9)])
    spec = _spec(tmp_path, files={"shortrate": "sr.csv", "rstar": "rstar.csv"})
    idx = pd.date_range("2001-01-31", "2019-12-31", freq="M")
    v, meta = build_survey_neutral(idx, spec, dir=tmp_path)
    s = pd.Series(v, index=idx)
    assert s[idx < "2017-01-31"].isna().all()      # no bridge built from it
    assert "rstar" not in str(meta["source"])


def test_public_extension_is_level_bridged_over_the_overlap(tmp_path):
    """The living survey continues the frozen one at the FROZEN survey's level."""
    _write(tmp_path, "sr.csv", [(f"20{y}-06-30", 3.0) for y in range(17, 27)])
    # public survey reads 0.5pp lower throughout, and extends one year further
    _write(tmp_path, "ext.csv", [(f"20{y}-09-30", 2.5) for y in range(22, 28)])
    idx = pd.date_range("2017-06-30", "2027-09-30", freq="M")
    v, meta = build_survey_neutral(idx, _spec(tmp_path), dir=tmp_path)
    s = pd.Series(v * 100, index=idx)
    assert meta["source"] == "consensus_shortrate+mps"
    assert s.loc["2027-09-30"] == pytest.approx(3.0, abs=1e-6)   # 2.5 + 0.5 offset


def test_extension_refused_without_overlap(tmp_path):
    _write(tmp_path, "sr.csv", [("2017-06-30", 3.0), ("2018-06-30", 3.0)])
    _write(tmp_path, "ext.csv", [("2025-06-30", 2.5), ("2026-06-30", 2.5)])
    idx = pd.date_range("2017-06-30", "2026-06-30", freq="M")
    v, meta = build_survey_neutral(idx, _spec(tmp_path), dir=tmp_path)
    assert meta["source"] == "consensus_shortrate"               # ext not spliced


def test_terminal_glide_reaches_the_corridor_without_a_step(tmp_path):
    """Japan's carry-forward: the endpoint glides from the last survey read to the
    communicated terminal across the staleness window — no step on lapse day."""
    _write(tmp_path, "sr.csv", [("2024-06-30", 1.2), ("2025-06-30", 1.5), ("2026-06-30", 1.55)])
    spec = _spec(tmp_path, smooth=True, files={"shortrate": "sr.csv"})
    idx = pd.date_range("2024-06-30", "2029-12-31", freq="M")
    v, meta = build_survey_neutral(idx, spec, dir=tmp_path,
                                   terminal_pct=2.0, terminal_band=(1.75, 2.25))
    s = pd.Series(v * 100, index=idx)
    assert meta["terminal_pct"] == 2.0 and meta["terminal_band"] == (1.75, 2.25)
    assert s.loc["2029-12-31"] == pytest.approx(2.0, abs=1e-9)   # holds at terminal
    assert float(s.diff().abs().max()) < 0.25                    # glide, never a step
    assert s.is_monotonic_increasing or (s.diff().dropna() > -0.05).all()


def test_frozen_hold_when_no_terminal(tmp_path):
    """Switzerland's carry-forward: past staleness the endpoint freezes at the last
    smoothed value rather than reverting to the (wrong) sample mean."""
    _write(tmp_path, "sr.csv", [("2025-06-30", 0.7), ("2026-06-30", 0.7)])
    spec = _spec(tmp_path, files={"shortrate": "sr.csv"})
    idx = pd.date_range("2025-06-30", "2030-12-31", freq="M")
    v, meta = build_survey_neutral(idx, spec, dir=tmp_path)
    s = pd.Series(v * 100, index=idx)
    assert s.loc["2030-12-31"] == pytest.approx(0.7, abs=1e-9)
    assert meta["stale"] is True


def test_no_files_is_inert(tmp_path):
    v, meta = build_survey_neutral(pd.date_range("2020-01-31", periods=6, freq="M"),
                                   _spec(tmp_path), dir=tmp_path)
    assert meta["source"] == "none" and np.all(np.isnan(v))


# --- survey pass-through (driver) --------------------------------------------------

def test_passthrough_zero_beta_is_plain_random_walk():
    from model.neutral_anchor import rw_smooth
    reads = pd.Series([2.0, 2.5], pd.to_datetime(["2024-01-15", "2024-07-15"]))
    idx = pd.date_range("2024-01-31", "2025-06-30", freq="M")
    drv = pd.Series(np.linspace(3, 5, len(idx)), idx)
    a = rw_smooth(reads, idx)
    b = rw_smooth(reads, idx, driver=drv, beta=0.0)
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_passthrough_moves_the_edge_by_beta_times_forward():
    from model.neutral_anchor import rw_smooth
    reads = pd.Series([2.0, 2.0, 2.0],
                      pd.to_datetime(["2024-01-15", "2024-04-15", "2024-07-15"]))
    idx = pd.bdate_range("2024-01-15", "2024-12-31")
    drv = pd.Series(4.0, idx)
    drv[drv.index > "2024-07-15"] = 5.0                  # forward +1pp after the last read
    out = rw_smooth(reads, idx, driver=drv, beta=0.16)
    flat = rw_smooth(reads, idx)
    assert out.iloc[-1] - flat.iloc[-1] == pytest.approx(0.16, abs=1e-6)
    # causal: nothing before the forward moved is touched
    assert np.allclose(out[:"2024-07-15"], flat[:"2024-07-15"])


def test_passthrough_reads_still_pin_the_level():
    """A forward move between reads that the next read does NOT confirm is undone there."""
    from model.neutral_anchor import rw_smooth
    reads = pd.Series([2.0, 2.0], pd.to_datetime(["2024-01-15", "2024-07-15"]))
    idx = pd.bdate_range("2024-01-15", "2024-07-15")
    drv = pd.Series(np.where(idx > "2024-03-01", 6.0, 4.0), idx)
    out = rw_smooth(reads, idx, obs_std=0.03, driver=drv, beta=0.16)
    assert out.iloc[-1] == pytest.approx(2.0, abs=0.02)


def test_survey_neutral_with_driver_tracks_past_staleness(tmp_path):
    spec = _spec(tmp_path, smooth=True)
    _write(tmp_path, spec.files["shortrate"], [("2020-01-15", 2.0), ("2020-07-15", 2.0)])
    idx = pd.date_range("2020-01-31", "2024-12-31", freq="M")
    drv = pd.Series(np.where(idx > "2023-01-01", 5.0, 4.0), idx)
    v, meta = build_survey_neutral(idx, spec, dir=tmp_path, driver=drv, beta=0.16)
    s = pd.Series(v * 100, idx)
    assert meta["stale"] and meta["pass_through"] == 0.16
    assert s.iloc[-1] == pytest.approx(2.16, abs=0.01)   # not frozen at 2.0


def test_basis_moves_the_survey_and_the_bridged_extension_together(tmp_path):
    from dataclasses import replace
    spec = _spec(tmp_path)
    _write(tmp_path, "sr.csv", [("2017-01-31", 2.5), ("2020-01-31", 2.5), ("2022-01-31", 2.5)])
    _write(tmp_path, "ext.csv", [("2019-01-31", 3.0), ("2021-01-31", 3.0), ("2023-01-31", 3.5)])
    idx = pd.date_range("2017-01-31", "2023-06-30", freq="M")
    a, _ = build_survey_neutral(idx, spec, dir=tmp_path)
    b, meta = build_survey_neutral(idx, replace(spec, basis_pct=-0.24), dir=tmp_path)
    assert meta["basis_pct"] == -0.24
    assert np.allclose(b - a, -0.0024, equal_nan=True)   # decimals; a pure level shift
