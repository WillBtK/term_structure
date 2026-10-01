"""Switzerland neutral-anchor wiring (no network; synthetic CSVs).

The construction is shared (`model.neutral_anchor.build_survey_neutral`); what is
pinned here is the Swiss design decision of 2026-08-20: the carry-forward is the KOF
CPI 5y median plus a CONSTANT real component, delivered through the standard
``shortrate_ext`` level bridge — the offset over the Consensus/KOF overlap IS the
real-component constant. No model r* anywhere.
"""
import numpy as np
import pandas as pd
import pytest

from switzerland import neutral_anchor as na


def _write(d, key, rows):
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(d / na.FILES[key], index=False)


def test_survey_era_only(tmp_path):
    _write(tmp_path, "shortrate", [("2017-01-09", 1.36), ("2020-07-13", -0.01)])
    idx = pd.date_range("2010-01-31", "2020-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    s = pd.Series(vals, index=idx)
    assert s[idx < "2017-01-09"].isna().all()      # pre-survey history untouched
    assert s["2017-02-28":].notna().all()


def test_carry_forward_is_kof_inflation_plus_constant_real(tmp_path):
    """Consensus rate sits a constant 0.3pp BELOW KOF pi_5y over the overlap (a
    negative implied real component, as measured on the real data: -0.14pp). Past the
    snapshot the endpoint must follow KOF + that constant — inflation supplies ALL the
    time variation, the real component none."""
    _write(tmp_path, "shortrate", [(f"20{y}-06-30", 0.7) for y in range(17, 27)])
    # two confirming KOF reads at the new level: the RW smoother deliberately damps a
    # single release (gain ~0.78), so convergence takes a second read — as it does on
    # the real quarterly feed.
    _write(tmp_path, "shortrate_ext",
           [(f"20{y}-09-30", 1.0) for y in range(17, 27)]
           + [("2027-06-30", 1.4), ("2027-09-30", 1.4)])
    idx = pd.date_range("2017-06-30", "2027-09-30", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "consensus_shortrate+kof_cpi5y+const_real"
    s = pd.Series(vals * 100, index=idx)
    # KOF moved +0.4 (1.0 -> 1.4); the endpoint converges onto 1.4 - 0.3 = 1.1
    assert s.loc["2027-09-30"] == pytest.approx(1.1, abs=0.05)


def test_no_kof_file_still_works_frozen(tmp_path):
    """Without the KOF file the anchor degrades to the plain frozen hold."""
    _write(tmp_path, "shortrate", [("2025-06-30", 0.7), ("2026-06-30", 0.7)])
    idx = pd.date_range("2025-06-30", "2030-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    assert meta["source"] == "consensus_shortrate" and meta["stale"] is True
    assert pd.Series(vals * 100, index=idx).loc["2030-12-31"] == pytest.approx(
        0.7 + na.BASIS_PCT, abs=1e-9)
