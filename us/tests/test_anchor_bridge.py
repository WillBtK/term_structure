"""The SPF/SME bridge for the US nominal anchor (us.anchor.bridged_forward_reads)."""
import numpy as np
import pandas as pd
import pytest

from us import anchor as A
from us import fetch_sme as F


def _write(d, name, s):
    pd.DataFrame({"date": s.index, "value": s.values}).to_csv(d / name, index=False)


@pytest.fixture
def anchors(tmp_path, monkeypatch):
    q1 = pd.to_datetime(["2024-02-15", "2025-02-15"])
    _write(tmp_path, "spf_bill10.csv", pd.Series([3.0, 3.2], q1))
    _write(tmp_path, "spf_tbill_y1.csv", pd.Series([4.0, 4.0], q1))
    _write(tmp_path, "spf_tbill_y2.csv", pd.Series([3.0, 3.0], q1))
    sme = pd.DataFrame({"date": pd.to_datetime(["2023-12-01", "2024-01-22", "2024-06-03",
                                                 "2025-01-21", "2025-06-09"]),
                        "fwd_2_10": [2.70, 2.80, 3.00, 3.10, 3.30]})
    sme.to_csv(tmp_path / "sme_ffr.csv", index=False)
    monkeypatch.setattr(A, "ANCHOR_DIR", tmp_path)
    return tmp_path


def test_spf_reads_are_window_forwards(anchors):
    r = A.spf_forward_reads()
    assert r.round(4).tolist() == [(30 - 7) / 8, (32 - 7) / 8]


def test_bridge_pins_spf_level_and_carries_sme_moves(anchors):
    reads, info = A.bridged_forward_reads()
    spf = A.spf_forward_reads()
    # SPF reads are kept exactly
    assert np.allclose(reads.reindex(spf.index), spf)
    # basis = SPF - nearest SME round at each Q1
    assert info["basis_by_q1"] == {"2024-02-15": round(2.875 - 2.80, 4),
                                   "2025-02-15": round(3.125 - 3.10, 4)}
    # an SME read after a Q1 uses THAT Q1's basis (causal): Jun-2025 = 3.30 + 0.025
    assert reads[pd.Timestamp("2025-06-09")] == pytest.approx(3.325)
    # ...and Jan-2025 still uses the 2024 basis, not the one learned a month later
    assert reads[pd.Timestamp("2025-01-21")] == pytest.approx(3.10 + 0.075)
    # SME rounds before the first overlap take the first basis
    assert reads[pd.Timestamp("2023-12-01")] == pytest.approx(2.70 + 0.075)
    assert info["last_read"] == "2025-06-09"


def test_without_sme_falls_back_to_spf(anchors):
    (anchors / "sme_ffr.csv").unlink()
    reads, info = A.bridged_forward_reads()
    assert reads.equals(A.spf_forward_reads()) and info["sme_rounds"] == 0


def test_survey_forward_moves_after_february(anchors):
    idx = pd.bdate_range("2025-02-17", "2025-07-31")
    s = A.survey_forward("nominal", idx)
    assert s.iloc[-1] > s.iloc[0] + 0.05         # SME rise after the Q1 round shows up


def test_horizon_dates():
    rel = pd.Timestamp("2026-07-20")
    assert F._horizon_date("fftr_pathofmodes_20260916", rel) == pd.Timestamp("2026-09-16")
    assert F._horizon_date("fftr_pathofmodes_2027q3", rel) == pd.Timestamp("2027-08-15")
    assert F._horizon_date("fftr_pathofmodes_2029", rel) == pd.Timestamp("2029-07-01")
    assert F._horizon_date("fftr_modalpe_10yravg", rel) is None


def test_parse_round_forward():
    rows = [("fftr_modalpe_10yravg", 0.032), ("fftr_modalpe_longerrun", 0.031),
            ("fftr_pathofmodes_20260101", 0.04), ("fftr_pathofmodes_20270101", 0.03),
            ("fftr_pathofmodes_2028", 0.03)]
    df = pd.DataFrame({"panel_type": "Combined", "subject": "fed_funds_target_range",
                       "aggregation": "pctl50", "value_tag": [r[0] for r in rows],
                       "aggregation_value": [r[1] for r in rows],
                       "survey_due_date": "2026-01-01"})
    import io
    buf = io.BytesIO(); df.to_excel(buf, index=False)
    r = F.parse_round(buf.getvalue())
    assert r["avg10"] == pytest.approx(3.2) and r["longer_run"] == pytest.approx(3.1)
    assert r["near2"] == pytest.approx(3.5, abs=0.01)    # one year at 4, one at 3
    assert r["fwd_2_10"] == pytest.approx((32 - 7) / 8, abs=0.01)


def test_central_is_interquartile_mean_not_median():
    df = pd.DataFrame({"value_tag": ["fftr_modalpe_10yravg"] * 3 + ["fftr_pathofmodes_2027"],
                       "aggregation": ["pctl25", "pctl50", "pctl75", "pctl50"],
                       "aggregation_value": [0.030, 0.0313, 0.035, 0.03]})
    c = F._central(df)
    assert c["fftr_modalpe_10yravg"] == pytest.approx((0.030 + 0.0313 + 0.035) / 3)
    assert c["fftr_pathofmodes_2027"] == pytest.approx(0.03)     # median fallback


def test_precise_reads_pull_the_smoother_harder(anchors):
    from model.neutral_anchor import rw_smooth
    s = pd.Series([3.0, 3.0, 3.3], pd.to_datetime(["2025-01-01", "2025-06-01", "2025-07-01"]))
    idx = pd.bdate_range("2025-07-01", "2025-07-02")
    loose = rw_smooth(s, idx, 0.15, 0.28).iloc[-1]
    tight = rw_smooth(s, idx, pd.Series([0.15, 0.15, 0.03], s.index), 0.28).iloc[-1]
    assert tight > loose and tight == pytest.approx(3.3, abs=0.03)
    reads, _ = A.bridged_forward_reads()
    sd = reads.attrs["obs_std"]
    assert set(sd[A.spf_forward_reads().index]) == {A.SMOOTH_OBS_STD}
    assert (sd.drop(A.spf_forward_reads().index) == A.SME_OBS_STD).all()
