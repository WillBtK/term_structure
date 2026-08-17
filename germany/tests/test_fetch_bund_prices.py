"""Offline test for the Bundesbank per-ISIN workbook parser (no network).

Builds a workbook that mirrors the real layout confirmed via --probe: one sheet per
trading day named ``DD.MM.YYYY``, a title row, a header row (ISIN / Coupon /
Description / Maturity / Residual life / Issue volume / Price / Yield in % / Dirty
price), then one row per security. Asserts parse_workbook derives the date from the
sheet name, maps the columns, and converts percent yields to decimal.
"""
import io

import pandas as pd

from germany.fetch_bund_prices import parse_workbook


def _build_workbook(days):
    """days: {sheet_name(DD.MM.YYYY): [(isin, coupon, desc, maturity, yld_pct), ...]}."""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xl:
        for sheet, rows in days.items():
            title = [f"Prices and yields of listed Federal securities as of {sheet}"]
            header = ["ISIN", "Coupon", "Description\n1)", "Maturity",
                      "Residual\nlife\nY/M", "Issue\nvolume\n(EUR billion)",
                      "Price 2)", "Yield\nin %", "Dirty price"]
            data = [title]
            data.append(header)
            for isin, coup, desc, mat, yld in rows:
                data.append([isin, coup, desc, mat, "0/0", 20.0, 99.9, yld, 100.0])
            pd.DataFrame(data).to_excel(xl, sheet_name=sheet, header=False, index=False)
    return bio.getvalue()


def test_parse_workbook_maps_layout_and_date_from_sheet():
    days = {
        "03.08.2026": [
            ("DE0001102408", 0.0, "DBR 0 08/15/26", "2026-08-15", 2.401658),
            ("DE0001102416", 0.25, "DBR 0.25 02/15/27", "2027-02-15", 2.52098),
            ("DE0001135044", 6.5, "DBR 6.5 07/04/27", "2027-07-04", 2.638016),
        ],
        "04.08.2026": [
            ("DE0001102408", 0.0, "DBR 0 08/15/26", "2026-08-15", 2.463929),
            ("DE0001102416", 0.25, "DBR 0.25 02/15/27", "2027-02-15", 2.523065),
        ],
    }
    df = parse_workbook(_build_workbook(days))
    assert ["date", "isin", "maturity", "yield"] == list(df.columns)[:4]
    assert "name" in df.columns          # Description captured for instrument filtering
    assert len(df) == 5
    # date derives from the sheet name (two distinct trading days)
    assert sorted(df["date"].dt.strftime("%Y-%m-%d").unique()) == ["2026-08-03", "2026-08-04"]
    # percent -> decimal
    assert abs(df["yield"].max() - 0.02638016) < 1e-9
    # maturity parsed to datetime
    row = df[(df["isin"] == "DE0001102408") & (df["date"] == pd.Timestamp("2026-08-03"))]
    assert row["maturity"].iloc[0] == pd.Timestamp("2026-08-15")


def test_parse_workbook_skips_non_date_sheets():
    days = {
        "Notes": [("DE0001102408", 0.0, "x", "2026-08-15", 2.4)],   # not a date -> skipped
        "05.08.2026": [("DE0001102408", 0.0, "x", "2026-08-15", 2.4),
                       ("DE0001102416", 0.25, "y", "2027-02-15", 2.5)],
    }
    df = parse_workbook(_build_workbook(days))
    assert set(df["date"].dt.strftime("%Y-%m-%d")) == {"2026-08-05"}
    assert len(df) == 2
