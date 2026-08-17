"""Tests for the Bank of Canada zero-coupon CSV parser (no network)."""
import numpy as np
import pandas as pd
import pytest

from canada.fetch_boc import parse_boc_csv

# A trimmed BoC yield-curve CSV: header + two dates, a few ZC columns, values already
# DECIMAL, trailing comma (empty column) and a blank 30y cell early (ragged long end).
FIXTURE = (
    "Date, ZC025YR, ZC100YR, ZC1000YR, ZC3000YR,\n"
    "1986-01-02, 0.1050000, 0.1010000, 0.0980000, ,\n"
    "2021-12-09, 0.0004900, 0.0075000, 0.0150000, 0.0190000,\n"
)


def test_parse_units_columns_ragged():
    df = parse_boc_csv(FIXTURE)
    # ZC codes -> years
    assert list(df.columns) == [0.25, 1.0, 10.0, 30.0]
    assert list(df.index) == [pd.Timestamp("1986-01-02"), pd.Timestamp("2021-12-09")]
    # values already decimal (no /100)
    assert df.loc[pd.Timestamp("1986-01-02"), 0.25] == pytest.approx(0.1050)
    assert df.loc[pd.Timestamp("1986-01-02"), 10.0] == pytest.approx(0.0980)
    # 30y ragged: NaN in 1986, present in 2021
    assert np.isnan(df.loc[pd.Timestamp("1986-01-02"), 30.0])
    assert df.loc[pd.Timestamp("2021-12-09"), 30.0] == pytest.approx(0.0190)
    # near-ELB short rate parsed (Canada's ~0.05% floor)
    assert df.loc[pd.Timestamp("2021-12-09"), 0.25] == pytest.approx(0.00049)


def test_trailing_empty_column_dropped():
    df = parse_boc_csv(FIXTURE)
    assert all(isinstance(c, float) for c in df.columns)     # no stray "Unnamed"/empty col
