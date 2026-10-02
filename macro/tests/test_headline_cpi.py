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
