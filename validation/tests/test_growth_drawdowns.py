"""Offline tests for the drawdown event study.

No network and no parquet: ``episodes`` and ``_aggregate`` are exercised on synthetic
inputs whose answer is known by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation import growth_drawdowns as g


def _u(vals, start="2000-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="MS"))


def test_episode_dates_the_trough_not_the_peak(monkeypatch):
    """The episode starts at the LAST GOOD month -- the minimum before the rise."""
    u = _u([5.0] * 12 + [4.0] + [4.5, 5.2, 6.0, 6.5] + [6.5] * 30)
    monkeypatch.setattr(g, "unemployment", lambda c: u)
    eps = g.episodes("US")
    assert len(eps) == 1
    assert eps[0] == u.index[12]                 # the 4.0 month, not the 6.5 peak


def test_a_rise_smaller_than_the_threshold_is_not_an_episode(monkeypatch):
    u = _u([4.0] + [4.3] * 40)                   # +0.3pp, below MIN_RISE
    monkeypatch.setattr(g, "unemployment", lambda c: u)
    assert g.episodes("US") == []


def test_a_double_dip_inside_the_gap_counts_once(monkeypatch):
    rise = [4.0, 5.0, 5.5, 5.2, 4.9, 5.0, 6.0, 6.5]
    u = _u(rise + [6.5] * 40)
    monkeypatch.setattr(g, "unemployment", lambda c: u)
    assert len(g.episodes("US")) == 1


def test_the_decomposition_adds_up_exactly():
    """A split whose legs do not sum to the total is not a decomposition."""
    rng = np.random.default_rng(0)
    n = 40
    e = pd.DataFrame({
        "country": ["US"] * n, "episode": ["1990-01"] * n,
        "h": list(range(n)),
        "e10": rng.normal(0, 20, n), "tp10": rng.normal(0, 20, n)})
    e["y10"] = e["e10"] + e["tp10"]
    e.loc[3, ["y10", "e10", "tp10"]] = np.nan          # a ragged row must not break it
    out = g._aggregate(e, "mean")
    assert np.allclose(out["y10"], out["e10"] + out["tp10"])


def test_a_column_missing_its_whole_tenor_is_skipped():
    e = pd.DataFrame({"country": ["US"], "episode": ["1990-01"], "h": [0],
                      "y10": [1.0], "e10": [0.5], "tp10": [0.5]})
    out = g._aggregate(e, "mean")
    assert "y10" in out and not any(c.startswith("y2") for c in out.columns)


def test_mean_and_median_both_available():
    e = pd.DataFrame({"country": ["US"] * 3, "episode": ["a", "b", "c"], "h": [0, 0, 0],
                      "y10": [1.0, 2.0, 30.0], "e10": [1.0, 1.0, 1.0],
                      "tp10": [0.0, 1.0, 29.0]})
    assert g._aggregate(e, "mean")["y10"].iloc[0] == 11.0
    assert g._aggregate(e, "median")["y10"].iloc[0] == 2.0


def test_the_regime_split_partitions_the_episodes(monkeypatch):
    """Every episode lands in exactly one bucket, on the right side of the threshold."""
    u = _u([5.0] * 6 + [4.0] + [5.5] * 60 + [4.0] + [5.5] * 60)
    monkeypatch.setattr(g, "unemployment", lambda c: u)
    eps = g.episodes("US")
    assert len(eps) == 2
    pi = {eps[0]: 2.0, eps[1]: 6.0}
    monkeypatch.setattr(g, "inflation_at", lambda c, d: pi)
    ev = pd.DataFrame({"country": ["US"] * 2,
                       "episode": [d.strftime("%Y-%m") for d in eps],
                       "h": [0, 0], "y10": [1.0, 2.0],
                       "e10": [0.5, 1.0], "tp10": [0.5, 1.0]})
    monkeypatch.setattr(g, "event_study", lambda c, **k: ev)
    out = g.by_inflation_regime(("US",))
    assert [c for c, _ in out["low inflation"][1]] == ["US"]
    assert [c for c, _ in out["high inflation"][1]] == ["US"]
    assert out["low inflation"][1][0][1] == eps[0].strftime("%Y-%m")
    assert out["high inflation"][1][0][1] == eps[1].strftime("%Y-%m")


def test_an_empty_regime_bucket_is_handled():
    """A market with no high-inflation episode must not crash the split."""
    out = g.by_inflation_regime(("US",), split=-99.0)   # everything is "high"
    assert out["low inflation"][0].empty
