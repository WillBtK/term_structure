import pandas as pd

from macro.fetch_quarterly_gdp import annual_from_remaining, year_to_date


def _df():
    idx = pd.to_datetime(["2025-10-01", "2026-01-01", "2026-04-01"])
    return pd.DataFrame({"US": [2.05, 2.64, 2.19]}, index=idx)


def test_year_to_date_lists_this_years_quarters():
    y = year_to_date("US", 2026, _df())
    assert y["n"] == 2 and y["yoy"] == [2.64, 2.19] and y["last"] == "2026Q2"


def test_remaining_rate_applies_to_unpublished_quarters_only():
    y = year_to_date("US", 2026, _df())
    assert abs(annual_from_remaining(-1.0, y) - (2.64 + 2.19 - 2.0) / 4) < 1e-9
    assert annual_from_remaining(-1.0, {"n": 0, "yoy": []}) == -1.0
