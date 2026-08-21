"""Parsing rules for the price-insensitive ownership sources."""
from __future__ import annotations

import pandas as pd

from supply.price_insensitive import _ecb_dates, csv_rows

PSPP_HEAD = "31/03/2015,30/04/2015,31/05/2015"
PEPP_HEAD = "Mar-20,Apr-20,May-20"


def test_both_ecb_date_conventions_parse():
    a = _ecb_dates(PSPP_HEAD.split(","))
    b = _ecb_dates(PEPP_HEAD.split(","))
    assert list(a) == [pd.Timestamp(2015, 3, 31), pd.Timestamp(2015, 4, 30),
                       pd.Timestamp(2015, 5, 31)]
    assert list(b) == [pd.Timestamp(2020, 3, 1), pd.Timestamp(2020, 4, 1),
                       pd.Timestamp(2020, 5, 1)]
    assert not a.isna().any() and not b.isna().any()


def test_a_trailing_cumulative_column_is_not_read_as_a_date():
    d = _ecb_dates(["Mar-20", "Apr-20", "Cumulative"])
    assert d.isna().tolist() == [False, False, True]


def test_csv_rows_keeps_embedded_commas_inside_quotes():
    rows = list(csv_rows('Germany,1,2\n"the Netherlands, NL",3,4\n'))
    assert rows[1][0] == "the Netherlands, NL"
