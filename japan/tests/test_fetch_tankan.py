"""Locating the five-year general-price figure in the Tankan summary."""
from __future__ import annotations

import pandas as pd

from japan.fetch_tankan import N_BLOCKS, _ROW, _release_date

# Section 7 as pypdf extracts it: fifteen enterprise-group / horizon cells, each a
# previous and a current row of four numbers -- output price, its change, general price,
# its change. The last current row is all enterprises, all industries, five years ahead.
PAGE = "\n".join(
    ["7. Inflation Outlook of Enterprises",
     "(Reference) The Average of Enterprises' Inflation Outlook %, % points"]
    + [f"Previous projection {i}.1 0.1 {i}.2 0.1\n"
       f"Current projection {i}.3 0.2 {i}.4 0.2" for i in range(1, N_BLOCKS + 1)])


def test_the_last_current_row_is_the_five_year_general_price_figure():
    rows = _ROW.findall(PAGE)
    assert len(rows) == N_BLOCKS
    assert float(rows[N_BLOCKS - 1][2]) == float(f"{N_BLOCKS}.4")


def test_previous_projection_rows_are_ignored():
    for r in _ROW.findall(PAGE):
        assert not r[0].endswith(".1")          # ".1"/" .2" are the previous rows


def test_release_dates_come_from_the_filename():
    assert _release_date("/en/statistics/tk/gaiyo/2026/tka2606.pdf") == pd.Timestamp("2026-06-30")
    assert _release_date("/en/statistics/tk/gaiyo/2021/tka2112.pdf") == pd.Timestamp("2021-12-31")
    assert _release_date("/en/statistics/tk/gaiyo/2026/notes.pdf") is None


def test_negative_readings_parse():
    """Deflation-era rounds print negative outlooks; the row pattern must accept them."""
    line = "Current projection -0.3 -0.1 -0.2 -0.1"
    assert _ROW.findall(line) == [("-0.3", "-0.1", "-0.2", "-0.1")]
