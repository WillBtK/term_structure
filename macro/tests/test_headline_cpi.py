"""Calendar-year-average inflation: pricing the unpublished months of the current year."""
import pandas as pd
import pytest

from macro.fetch_headline_cpi import remaining_rate, year_to_date


def test_remaining_months_deliver_the_stated_average():
    ytd = {"year": 2026, "n": 8, "mean": 3.0}
    r = remaining_rate(3.6, ytd)
    assert r == pytest.approx((12 * 3.6 - 8 * 3.0) / 4)
    assert (8 * 3.0 + 4 * r) / 12 == pytest.approx(3.6)


def test_nothing_published_or_a_complete_year_uses_the_input():
    assert remaining_rate(2.5, {"year": 2026, "n": 0, "mean": None}) == 2.5
    assert remaining_rate(2.5, {"year": 2026, "n": 12, "mean": 2.0}) == 2.5


def test_year_to_date_counts_only_the_year_asked():
    idx = pd.date_range("2025-10-01", "2026-03-01", freq="MS")
    df = pd.DataFrame({"US": [3.0, 3.0, 3.0, 2.0, 4.0, 3.0]}, index=idx)
    y = year_to_date("US", 2026, df)
    assert y == {"year": 2026, "n": 3, "mean": 3.0, "last": "2026-03-01"}
    assert year_to_date("JP", 2026, df)["n"] == 0


def test_seasonal_adjustment_removes_a_fixed_seasonal_and_keeps_the_trend():
    import numpy as np
    from macro.fetch_headline_cpi import seasonally_adjust
    idx = pd.date_range("2010-01-01", "2026-08-01", freq="MS")
    trend = 100 * np.exp(0.02 / 12 * np.arange(len(idx)))         # 2% a year
    season = np.exp(0.006 * np.sin(2 * np.pi * idx.month / 12))
    sa = seasonally_adjust(pd.Series(trend * season, idx))
    yoy = (sa / sa.shift(12) - 1) * 100
    assert yoy.dropna().between(1.9, 2.15).all()
    mom = sa.pct_change().dropna() * 1200                         # no seasonal swing left
    assert mom.std() < 0.2
