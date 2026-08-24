"""Offline tests for the Shiller loader and the long stock-bond correlation.

No network: the workbook parser is exercised on a synthetic sheet built to carry the
real file's awkward shape (banner rows above the header, notes below the data, YYYY.MM
float dates), and the return construction is checked against hand-computable cases.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest

from macro import shiller


def _workbook(rows: list[list], notes: bool = True) -> bytes:
    """A miniature ie_data.xls: banners, the Date header, rows, then trailing notes."""
    header = ["Date", "P", "D", "E", "CPI", "Fraction", "Rate GS10"]
    table = [["Stock Market Data Used in \"Irrational Exuberance\"", None, None,
              None, None, None, None],
             ["Robert J. Shiller", None, None, None, None, None, None],
             [None] * 7,
             header] + rows
    if notes:
        table += [[None] * 7, ["Notes: the trailing block the parser must drop",
                               None, None, None, None, None, None]]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame(table).to_excel(xl, sheet_name="Data", header=False, index=False)
    return buf.getvalue()


def _rows(n: int = 24, price: float = 100.0, div: float = 3.0, y: float = 5.0):
    out = []
    for i in range(n):
        stamp = 1871 + (i // 12) + ((i % 12) + 1) / 100.0
        out.append([round(stamp, 2), price, div, None, 100.0, None, y])
    return out


def test_parse_finds_header_and_drops_notes():
    df = shiller._parse(_workbook(_rows(24)))
    assert len(df) == 24
    assert list(df.columns) == ["price", "dividend", "cpi", "y10"]
    assert df.index[0] == pd.Timestamp("1871-01-01")
    assert df.index[-1] == pd.Timestamp("1872-12-01")


def test_month_recovered_from_the_fractional_stamp():
    """1871.10 is October, not "a tenth of the way through 1871"."""
    df = shiller._parse(_workbook([[1871.10, 100.0, 3.0, None, 100.0, None, 5.0],
                                   [1871.11, 100.0, 3.0, None, 100.0, None, 5.0],
                                   [1871.12, 100.0, 3.0, None, 100.0, None, 5.0]]))
    assert [d.month for d in df.index] == [10, 11, 12]


def test_missing_header_raises():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame([[1, 2], [3, 4]]).to_excel(xl, sheet_name="Data",
                                                header=False, index=False)
    with pytest.raises(ValueError, match="Date"):
        shiller._parse(buf.getvalue())


def test_flat_market_gives_carry_only():
    """Constant price and yield: equity earns its dividend, the bond earns its carry."""
    df = shiller._parse(_workbook(_rows(12, price=100.0, div=3.0, y=5.0)))
    r = shiller.returns(df)
    assert r["equity"].sub(0.03 / 12.0).abs().max() < 1e-12
    assert r["bond"].sub(0.05 / 12.0).abs().max() < 1e-12


def test_bond_return_is_carry_less_duration_times_yield_change():
    rows = [[1871.01, 100.0, 3.0, None, 100.0, None, 5.0],
            [1871.02, 100.0, 3.0, None, 100.0, None, 6.0]]
    r = shiller.returns(shiller._parse(_workbook(rows)))
    y, dy = 0.06, 0.01
    dur = (1.0 - (1.0 + y) ** (-shiller.TENOR)) / y
    assert r["bond"].iloc[0] == pytest.approx(y / 12.0 - dur * dy, rel=1e-12)
    assert r["bond"].iloc[0] < 0                     # a 1pp sell-off loses money


def test_correlation_window_length_and_sign():
    """A window where equity and the bond move together must come back positive."""
    rows, price, y = [], 100.0, 5.0
    for i in range(shiller.WINDOW + 12):
        # alternate: yields fall (bond gains) on the months equity rises
        up = i % 2 == 0
        price = price * (1.02 if up else 0.98)
        y = y - 0.1 if up else y + 0.1
        rows.append([round(1871 + (i // 12) + ((i % 12) + 1) / 100.0, 2),
                     round(price, 4), 3.0, None, 100.0, None, round(y, 4)])
    r = shiller.returns(shiller._parse(_workbook(rows)))
    c = r["equity"].rolling(shiller.WINDOW).corr(r["bond"]).dropna()
    assert len(c) == len(r) - shiller.WINDOW + 1
    assert c.min() > 0.9


def test_regime_means_centres_the_window():
    """The era table must not date a regime by when its window ended."""
    idx = pd.date_range("1960-01-01", "2000-12-01", freq="MS")
    c = pd.Series(range(len(idx)), index=idx, dtype=float)
    out = shiller.regime_means(c)
    assert {"era", "regime", "corr", "n"} <= set(out.columns)
    # a centred window shifts labels back by half the window, so the table must
    # reach eras the raw index alone would not cover
    assert out["n"].sum() > 0
    assert (out["era"] == "1946-1965").any()
