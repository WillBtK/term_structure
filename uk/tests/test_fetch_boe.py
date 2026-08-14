"""Tests for the BoE GLC spot-curve parser (no network)."""
import io

import numpy as np
import pandas as pd

from uk import config as cfg
from uk.fetch_boe import parse_spot_workbook, discover_zip_url


def _fake_workbook() -> bytes:
    """Build a minimal xlsx mimicking the BoE '4. spot curve' layout:
    a title row, a 'years:' maturity header, then month-end dated rows (percent)."""
    # rows: 0 title, 1 'Maturity', 2 'months:', 3 'years:', 4 FAMEDATE, 5+ data
    data = [
        ["UK nominal spot curve", None, None, None],
        ["Maturity", None, None, None],
        ["months:", 12, 60, 120],
        ["years:", 1.0, 5.0, 10.0],
        ["FAMEDATE", "WU", "WU", "WU"],
        [pd.Timestamp("2020-01-31"), 0.55, 0.60, 0.75],
        [pd.Timestamp("2020-02-29"), np.nan, 0.42, 0.58],
    ]
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=cfg.SPOT_SHEET, header=False, index=False)
    return buf.getvalue()


def test_parse_spot_workbook_layout_and_units():
    df = parse_spot_workbook(_fake_workbook())
    assert list(df.columns) == [1.0, 5.0, 10.0]
    assert list(df.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    # percent -> decimal
    assert df.loc[pd.Timestamp("2020-01-31"), 10.0] == 0.0075
    # ragged: 1y missing on the second date
    assert np.isnan(df.loc[pd.Timestamp("2020-02-29"), 1.0])
    assert df.loc[pd.Timestamp("2020-02-29"), 5.0] == 0.0042


def test_discover_zip_url_fallback():
    # A nonsense zip name can't be discovered on the page -> stable media path.
    url = discover_zip_url("definitely-not-a-real-file-xyz.zip")
    assert url.endswith("definitely-not-a-real-file-xyz.zip")
    assert "yield-curves" in url
