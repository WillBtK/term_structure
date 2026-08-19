"""Offline test for the Bundesbank per-ISIN workbook parser (no network).

Builds a workbook that mirrors the real layout confirmed via --probe: one sheet per
trading day named ``DD.MM.YYYY``, a title row, a header row (ISIN / Coupon /
Description / Maturity / Residual life / Issue volume / Price / Yield in % / Dirty
price), then one row per security. Asserts parse_workbook derives the date from the
sheet name, maps the columns, and converts percent yields to decimal.
"""
import io

import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Accumulation. The index exposes only the most recent ~5 monthly workbooks and no
# reachable Bundesbank source backfills what scrolls off, so load_bund_yields must
# UNION the scrape into the versioned store rather than replace it. This is the
# invariant that broke silently before: the fetcher wrote only what it had just
# scraped, so the "history" never grew past the rolling window.
# ---------------------------------------------------------------------------
import germany.fetch_bund_prices as fbp
from validation import config as vcfg


def _panel(dates, isins, yld=0.02):
    return pd.DataFrame([
        {"date": pd.Timestamp(d), "isin": i, "maturity": pd.Timestamp("2030-01-01"),
         "yield": yld, "name": ""}
        for d in dates for i in isins])


def test_merge_unions_and_prefers_the_fresh_parse():
    old = _panel(["2026-01-02", "2026-01-03"], ["DE0001102408"], yld=0.01)
    new = _panel(["2026-01-03", "2026-01-06"], ["DE0001102408"], yld=0.02)
    out = fbp._merge(old, new)
    assert len(out) == 3                                  # 01-02, 01-03, 01-06
    got = out.set_index("date")["yield"]
    assert got[pd.Timestamp("2026-01-02")] == 0.01        # only the store had it
    assert got[pd.Timestamp("2026-01-03")] == 0.02        # fresh parse wins
    assert got[pd.Timestamp("2026-01-06")] == 0.02


def test_load_keeps_days_that_scrolled_off_the_index(tmp_path, monkeypatch):
    monkeypatch.setattr(vcfg, "BUND_PRICES_STORE", tmp_path / "store.parquet")
    monkeypatch.setattr(vcfg, "BUND_PRICES_RAW_PARQUET", tmp_path / "cache.parquet")
    # first run sees January; a later run sees only February (January has scrolled off)
    monkeypatch.setattr(fbp, "discover_workbook_urls", lambda limit=None: ["u1"])
    monkeypatch.setattr(fbp, "_get", lambda url, timeout=60: b"")
    monkeypatch.setattr(fbp, "parse_workbook",
                        lambda _c: _panel(["2026-01-02"], ["DE0001102408"]))
    first = fbp.load_bund_yields(refresh=True)
    assert set(first["date"]) == {pd.Timestamp("2026-01-02")}

    monkeypatch.setattr(fbp, "parse_workbook",
                        lambda _c: _panel(["2026-02-02"], ["DE0001102408"]))
    second = fbp.load_bund_yields(refresh=True)
    assert set(second["date"]) == {pd.Timestamp("2026-01-02"), pd.Timestamp("2026-02-02")}


def test_index_failure_falls_back_to_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(vcfg, "BUND_PRICES_STORE", tmp_path / "store.parquet")
    monkeypatch.setattr(vcfg, "BUND_PRICES_RAW_PARQUET", tmp_path / "cache.parquet")
    _panel(["2026-01-02"], ["DE0001102408"]).to_parquet(tmp_path / "store.parquet")

    def boom(limit=None):
        raise RuntimeError("index unreachable")
    monkeypatch.setattr(fbp, "discover_workbook_urls", boom)
    out = fbp.load_bund_yields(refresh=True)
    assert set(out["date"]) == {pd.Timestamp("2026-01-02")}


# ---------------------------------------------------------------------------
# The Bundesbank changed the workbook layout between the June and July 2026 files.
# Reading the OLD one with the new one's rules produced ISINs like "114183" and
# descriptions of "DE000" — which silently disabled drop_inflation_linked (it matches on
# the description), leaving real-yield linkers in a nominal fit and standing the gauge
# up at ~40 bp for those months against ~5 bp either side. Both layouts are pinned here.
# ---------------------------------------------------------------------------
def _build_old_layout(rows, date="01.04.2026"):
    """The pre-July workbook: the ISIN is SPLIT ACROSS THREE CELLS, the coupon column
    carries no header, the maturity is dd.mm.yyyy text, and the header spans three rows.
    ``rows``: [(isin, coupon, description, maturity_ddmmyyyy, yield_pct), ...]"""
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xl:
        data = [
            [None, f"Prices and yields of listed Federal securities as of {date}"]
            + [None] * 9,
            [None, None, None, None, None, None, "Residual", "Issue", "Price 3)", "", ""],
            [None, "ISIN 1)", None, None, "Description 2)  4)", "Maturity",
             "life", "volume", "as of", "Yield", '"Dirty price"'],
            [None, None, None, None, None, "date", "y/m", "€ billion", date, None, ""],
        ]
        for isin, coup, desc, mat, yld in rows:
            data.append([isin[:5], isin[5:11], int(isin[11]), coup, desc, mat,
                         "0 / 0", 20.0, 99.9, yld, 100.0])
        pd.DataFrame(data).to_excel(xl, sheet_name=date, header=False, index=False)
    return bio.getvalue()


OLD_ROWS = [
    ("DE0001141836", 0.0, "BO S 183", "10.04.2030", 2.090051),
    ("DE0001030567", 0.1, "Bund 15 index.", "15.04.2033", -0.440504),
    ("DE0001102408", 2.5, "Bund 26", "15.08.2036", 2.401658),
]


def test_parses_the_pre_july_layout_with_the_split_isin():
    df = parse_workbook(_build_old_layout(OLD_ROWS))
    assert len(df) == 3
    # the ISIN is rebuilt whole from its three cells, not truncated to the middle one
    assert set(df["isin"]) == {"DE0001141836", "DE0001030567", "DE0001102408"}
    row = df.set_index("isin").loc["DE0001102408"]
    assert row["name"] == "Bund 26"                    # not "DE000"
    assert row["coupon"] == pytest.approx(2.5)
    assert row["maturity"] == pd.Timestamp("2036-08-15")   # dd.mm.yyyy, not mm/dd
    assert row["yield"] == pytest.approx(0.02401658)
    assert df["date"].unique().tolist() == [pd.Timestamp("2026-04-01")]


def test_the_old_layout_linker_marker_is_recognised():
    """The old layout spells linkers 'Bund 15 index.', the new one 'DBRI ... I/L'. If
    only the new spelling matched, those months would keep real-yield bonds in a
    nominal fit — the exact failure that stood the shipped gauge up at ~40 bp."""
    from validation.bund_liquidity import drop_inflation_linked
    df = parse_workbook(_build_old_layout(OLD_ROWS))
    kept = drop_inflation_linked(df)
    assert set(kept["isin"]) == {"DE0001141836", "DE0001102408"}


def test_both_layouts_yield_the_same_schema():
    new = parse_workbook(_build_workbook({
        "01.04.2026": [("DE0001102408", 2.5, "DBR 2.5 08/15/36", "2036-08-15", 2.401658)],
    }))
    old = parse_workbook(_build_old_layout(OLD_ROWS[:1]))
    assert list(new.columns) == list(old.columns)
    assert new["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$").all()
    assert old["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$").all()
