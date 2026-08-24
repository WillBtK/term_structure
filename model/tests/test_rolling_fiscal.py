"""Offline tests for the rolling fiscal sensitivity.

``rolling_fiscal_sensitivity`` is a pure function of the panel it is handed, so it is
exercised on synthetic panels where the true coefficient is known and changes on a
date we choose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.tp_drivers import CORE, ROLL_WINDOW, rolling_fiscal_sensitivity


def _panel(n: int = 140, coef_before: float = 0.0, coef_after: float = 0.50,
           switch: int = 70, seed: int = 0) -> pd.DataFrame:
    """A vintage panel whose fiscal coefficient steps at ``switch``.

    ``fiscal`` changes every row so the vintage filter keeps all of them; the core
    drivers are noise so they cannot carry the signal.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1984-02-01", periods=n, freq="QS")
    fiscal = 3.0 + np.sin(np.arange(n) / 4.0) + rng.normal(0, 0.05, n)
    cols = {k: rng.normal(2.0, 0.4, n) for k in CORE}
    beta = np.where(np.arange(n) < switch, coef_before, coef_after)
    tp = 1.0 + beta * fiscal + rng.normal(0, 0.02, n)
    return pd.DataFrame({"tp": tp, "fiscal": fiscal, **cols}, index=idx)


def test_recovers_a_flat_coefficient():
    out = rolling_fiscal_sensitivity(_panel(coef_before=0.30, coef_after=0.30))
    vals = [v for _, v in out["trend"]]
    assert len(vals) > 10
    assert all(abs(v - 30.0) < 4.0 for v in vals)      # 0.30 -> 30bp


def test_the_step_shows_up_in_the_rolling_series():
    out = rolling_fiscal_sensitivity(_panel(coef_before=0.0, coef_after=0.50))
    vals = [v for _, v in out["trend"]]
    assert abs(vals[0]) < 6.0                          # window entirely pre-step
    assert vals[-1] > 40.0                             # window entirely post-step
    assert vals[-1] > vals[0] + 30.0                   # and it rose, not fell


def test_the_rise_is_monotone_enough_to_read_as_a_trend():
    out = rolling_fiscal_sensitivity(_panel(coef_before=0.0, coef_after=0.50))
    vals = np.array([v for _, v in out["trend"]])
    mid = len(vals) // 2
    assert vals[mid:].mean() > vals[:mid].mean()


def test_both_specifications_are_returned_and_aligned():
    out = rolling_fiscal_sensitivity(_panel())
    assert len(out["trend"]) == len(out["notrend"])
    assert [d for d, _ in out["trend"]] == [d for d, _ in out["notrend"]]
    assert out["window"] == ROLL_WINDOW


def test_window_length_sets_the_number_of_points():
    n = 140
    out = rolling_fiscal_sensitivity(_panel(n=n), window=50)
    assert out["window"] == 50
    assert len(out["trend"]) == out["n"] - 50 + 1


def test_short_panel_returns_empty_rather_than_a_spurious_fit():
    out = rolling_fiscal_sensitivity(_panel(n=20))
    assert out["trend"] == [] and out["notrend"] == []
