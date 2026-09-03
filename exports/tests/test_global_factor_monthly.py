"""Sanity checks for the global-factor export -- mainly a unit-inversion identity
check, since a units mistake there (percent vs decimal, standardized vs raw) would
silently hand the other repo a factor decomposition that looks plausible but is wrong.
"""
import numpy as np
import pandas as pd

from exports.global_factor_monthly import build
from model import global_factor as gf


def test_switzerland_excluded():
    m, _ = build()
    assert "CH" not in set(m["country"].unique())


def test_five_markets_present():
    m, _ = build()
    assert set(m["country"].unique()) == {"US", "DE", "JP", "UK", "CA"}


def test_loading_mean_sd_constant_per_market():
    m, _ = build()
    for c, sub in m.groupby("country", observed=True):
        assert sub["loading"].nunique() == 1
        assert sub["mean_pct"].nunique() == 1
        assert sub["sd_pct"].nunique() == 1


def test_inversion_reproduces_raw_term_premium():
    """TP = mean_pct + sd_pct * (loading * G + residual) should match the raw 10y
    term premium in the canonical panel, to within rounding (values are stored to 5dp
    standardized / 4dp percent)."""
    m, _ = build()
    panel = gf.load_measure_panel("term_premium", tenor=10.0, freq="M") * 100.0  # -> percent
    # the CSV labels months first-of-month; the panel itself is month-END-indexed
    panel.index = panel.index.values.astype("datetime64[M]")
    for c in m["country"].cat.categories:
        sub = m[m["country"] == c].set_index("month")
        implied = sub["mean_pct"] + sub["sd_pct"] * (sub["loading"] * sub["G"] + sub["residual"])
        raw = panel[c].reindex(implied.index.astype("datetime64[ns]"))
        diff = (implied.to_numpy() - raw.to_numpy())
        assert np.nanmax(np.abs(diff)) < 0.01, c   # 1bp: rounding in the stored columns


def test_no_month_before_its_own_market_starts():
    """A market's rows never predate the first month it actually reports in the
    canonical panel (the export must not fabricate a value from the Kalman prior)."""
    m, _ = build()
    panel = gf.load_measure_panel("term_premium", tenor=10.0, freq="M")
    for c in m["country"].cat.categories:
        first_row = m.loc[m["country"] == c, "month"].min()
        first_panel = panel[c].dropna().index.min().to_period("M").to_timestamp()
        assert pd.Timestamp(first_row) == first_panel, c
