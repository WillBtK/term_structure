"""Germany neutral-anchor wiring (no network; synthetic CSVs).

Survey-only since the Ferreira review (2026-08-20): the endpoint is the Consensus EA
short-rate survey, carried forward by the ECB SPF longer-term rate assumption, NaN
before Oct-2016. The shared construction is tested in model/tests/test_survey_neutral;
here: Germany's filenames, its activation contract, and the bridge_fidelity report
that motivated the design.
"""
import numpy as np
import pandas as pd
import pytest

from germany import neutral_anchor as na


def _write(d, key, rows):
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(d / na.FILES[key], index=False)


def test_no_files_is_inert(tmp_path):
    vals, meta = na.build_nominal_neutral(
        pd.date_range("2020-01-31", periods=6, freq="M"), dir=tmp_path)
    assert meta["source"] == "none" and np.all(np.isnan(vals))
    assert not na.has_anchor(tmp_path) and not na.is_enabled(tmp_path)


def test_survey_era_only(tmp_path):
    """NaN before the first Consensus release — 1990-2016 history is not rewritten,
    which is what the Ferreira review demanded."""
    _write(tmp_path, "shortrate", [("2016-10-10", 2.08), ("2020-10-12", 0.69)])
    idx = pd.date_range("2010-01-31", "2020-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    s = pd.Series(vals, index=idx)
    assert s[idx < "2016-10-10"].isna().all()
    assert s["2016-11-30":].notna().all()


def test_spf_rate_extends_the_survey(tmp_path):
    """Past the frozen snapshot the ECB SPF longer-term rate assumption carries the
    endpoint, level-bridged over >=6 months of overlap."""
    _write(tmp_path, "shortrate", [(f"202{y}-06-30", 2.2) for y in range(0, 7)])
    _write(tmp_path, "shortrate_ext", [("2025-03-31", 2.2), ("2025-09-30", 2.2),
                                       ("2026-03-31", 2.2), ("2026-09-30", 2.2),
                                       ("2027-03-31", 2.4)])
    idx = pd.date_range("2020-06-30", "2027-03-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "consensus_shortrate+ecb_spf_rate"
    s = pd.Series(vals * 100, index=idx)
    # smoothed toward the SPF, on the Bund curve's basis (the survey is a Euribor forecast)
    assert s.loc["2027-03-31"] == pytest.approx(2.4 + na.BASIS_PCT, abs=0.15)


def test_activation_is_a_decision_not_a_side_effect(tmp_path, monkeypatch):
    _write(tmp_path, "shortrate", [("2016-10-10", 2.08), ("2020-10-12", 0.69)])
    assert na.is_enabled(tmp_path)                     # ACTIVATED=True since 2026-08-20
    monkeypatch.setattr(na, "ACTIVATED", False)
    assert not na.is_enabled(tmp_path)                 # the flag is the switch
