"""Japan neutral-anchor wiring (no network; synthetic CSVs).

The construction itself is shared (`model.neutral_anchor.build_survey_neutral`, tested
in `model/tests/test_survey_neutral.py`); what is tested here is Japan's wiring — the
survey-only contract after the Ferreira review, and the BoJ-corridor terminal — plus
the RW endpoint smoother, which lives in the shared module but was born here.
"""
import numpy as np
import pandas as pd
import pytest

from japan import neutral_anchor as na
from model.neutral_anchor import rw_smooth


def _write(d, rows):
    pd.DataFrame(rows, columns=["date", "value"]).to_csv(d / na.FILES["shortrate"],
                                                         index=False)


def test_no_files_is_inert(tmp_path):
    vals, meta = na.build_nominal_neutral(
        pd.date_range("2020-01-31", periods=12, freq="M"), dir=tmp_path)
    assert meta["source"] == "none" and np.all(np.isnan(vals))
    assert not na.has_anchor(tmp_path)


def test_survey_only_no_prehistory_rewrite(tmp_path):
    """Ferreira and the CPI/Tankan bridge are OUT of the construction: before the first
    Consensus short-rate release the endpoint is NaN (constant-theta), even though the
    retired rung files still sit in the real anchor directory."""
    _write(tmp_path, [("2017-01-09", 0.9), ("2020-01-13", 0.6)])
    idx = pd.date_range("2000-01-31", "2019-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    s = pd.Series(vals, index=idx)
    assert s[idx < "2017-01-09"].isna().all()
    assert s["2017-02-28":].notna().all()
    assert meta["source"].startswith("consensus_shortrate")
    assert "ferreira" not in str(meta)


def test_terminal_is_the_boj_corridor(tmp_path):
    """Past the staleness window the endpoint glides to 2.0% — the midpoint of the
    BoJ-communicated 1.75-2.25% corridor — and the band rides in the metadata."""
    _write(tmp_path, [("2025-07-14", 1.35), ("2026-07-13", 1.55)])
    idx = pd.date_range("2025-07-31", "2029-12-31", freq="M")
    vals, meta = na.build_nominal_neutral(idx, dir=tmp_path)
    s = pd.Series(vals * 100, index=idx)
    assert meta["terminal_pct"] == 2.0
    assert meta["terminal_band"] == (1.75, 2.25)
    assert s.loc["2029-12-31"] == pytest.approx(2.0, abs=1e-9)
    assert float(s.diff().abs().max()) < 0.25          # glide, not a step


# ---------------------------------------------------------------------------
# RW endpoint smoother (shared machinery; historically exercised from here)
# ---------------------------------------------------------------------------

def test_rw_smoother_removes_step_and_keeps_edge():
    dts = pd.date_range("2023-01-15", periods=9, freq="91D")
    reads = pd.Series([0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], index=dts)
    idx = pd.date_range("2023-01-31", "2025-01-31", freq="M")
    out = rw_smooth(reads, idx)
    assert out.diff().abs().max() < 0.60
    win = out.loc["2023-07-31":"2024-04-30"]
    assert (win.diff().dropna() >= -1e-9).all()
    assert 0.5 < win.iloc[len(win) // 2] < 1.5
    assert abs(out.iloc[-1] - 1.5) < 0.03


def test_rw_smoother_no_lookahead_at_edge():
    reads = pd.Series([0.5, 0.6, 0.7],
                      index=pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15"]))
    short = rw_smooth(reads, pd.date_range("2023-01-31", "2023-07-31", freq="M"))
    long = rw_smooth(reads, pd.date_range("2023-01-31", "2024-06-30", freq="M"))
    assert abs(short.loc["2023-07-31"] - long.loc["2023-07-31"]) < 1e-9
    assert long.loc["2023-10-31":].std() < 1e-6


def test_rw_smoother_degenerate_falls_back():
    one = pd.Series([1.0], index=pd.to_datetime(["2023-01-15"]))
    idx = pd.date_range("2023-01-31", "2023-06-30", freq="M")
    assert np.allclose(rw_smooth(one, idx).to_numpy(), 1.0)
