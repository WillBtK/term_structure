"""Parsing the BoJ ``mei`` holdings workbook.

The bug these exist to prevent: the workbook names each bond type once and leaves the
rows beneath it blank, so the type column has to be forward-filled. Filling the MAPPED
label rather than the heading TEXT silently re-labelled every block we do not recognise
as the last one we did — "Floating-rate JGB" and "Inflation-indexed JGB" sit directly
below 40-year in the sheet and were being stamped "40-Year JGB". That both inflated the
holdings total the ownership driver reads (by 1.5% on average, 3.4% at the start of the
sample) and collided with genuine 40-year issue numbers.
"""
from __future__ import annotations

import io

import pandas as pd

from supply.fetch_jp import _parse_mei


def workbook(rows: list) -> bytes:
    """A minimal mei-shaped sheet: type, issue, amount, with the header text the
    locator anchors on. ``None`` in the type column means "same block as above",
    which is how the real workbook writes it."""
    frame = pd.DataFrame([[None, None, "Amount Outstanding"]] + rows)
    buf = io.BytesIO()
    frame.to_excel(buf, header=False, index=False)
    return buf.getvalue()


def test_an_unrecognised_block_is_dropped_not_relabelled():
    blob = workbook([["40-year JGB", 1, 100.0],
                     [None, 2, 200.0],                  # same block, type left blank
                     ["Floating-rate JGB", 1, 900.0],   # a block we deliberately exclude
                     [None, 2, 800.0]])
    out = _parse_mei(blob, "mei260831.xlsx")
    assert set(out["bond_type"]) == {"40-Year JGB"}
    assert out["amount"].sum() == 300.0        # not 2000.0
    assert not out.duplicated(["bond_type", "issue"]).any()


def test_a_blank_type_still_continues_its_own_block():
    """The ffill itself is load-bearing — dropping it entirely would lose most rows."""
    blob = workbook([["10-year JGB", 1, 10.0], [None, 2, 20.0], [None, 3, 30.0]])
    out = _parse_mei(blob, "mei260831.xlsx")
    assert len(out) == 3 and out["amount"].sum() == 60.0


def test_the_english_layout_that_repeats_the_type_on_every_row_still_parses():
    """The 2013-2016 archive names the type on every row; both layouts must work."""
    blob = workbook([["2-year JGB", 1, 5.0], ["2-year JGB", 2, 7.0]])
    out = _parse_mei(blob, "mei150130.xls")
    assert list(out["bond_type"]) == ["2-Year JGB"] * 2
