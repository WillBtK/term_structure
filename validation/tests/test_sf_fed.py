"""Tests for the concrete FRBSF multi-sheet parser (no network).

Builds a tiny workbook mirroring FRBSF_Term_Model_Data.xlsx (one sheet per
quantity, columns like FTERMP02YR, values in decimal) and checks the reshape.
"""
import io

import numpy as np
import pandas as pd
import pytest

from validation.sf_fed import parse_frbsf_model_data


def _make_workbook_bytes() -> bytes:
    dates = pd.to_datetime(["2010-01-04", "2010-01-05"])
    sheets = {
        "Contents": pd.DataFrame({"x": ["FRBSF Term Premium Model Data"]}),
        "Fit Term Premium": pd.DataFrame(
            {"DATE": dates, "FTERMP02YR": [0.007883, 0.008], "FTERMP10YR": [0.018773, 0.019]}
        ),
        "Fit Avg Expected Rate": pd.DataFrame(
            {"DATE": dates, "FEXPTR02YR": [0.002784, 0.0028], "FEXPTR10YR": [0.022745, 0.0228]}
        ),
        "Fit Treasury Yields": pd.DataFrame(
            {"DATE": dates, "FYIELD02YR": [0.010667, 0.0108], "FYIELD10YR": [0.041518, 0.0418]}
        ),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


def test_parse_frbsf_shapes_and_maturities():
    tidy = parse_frbsf_model_data(_make_workbook_bytes())
    assert list(tidy.columns) == [
        "date",
        "maturity",
        "term_premium",
        "expected_rate",
        "fitted_yield",
    ]
    assert set(tidy["maturity"]) == {2.0, 10.0}
    assert len(tidy) == 4  # 2 dates x 2 maturities


def test_parse_frbsf_values_decimal_and_consistent():
    tidy = parse_frbsf_model_data(_make_workbook_bytes())
    row = tidy[(tidy["date"] == pd.Timestamp("2010-01-04")) & (tidy["maturity"] == 10.0)].iloc[0]
    assert row["term_premium"] == pytest.approx(0.018773)
    assert row["expected_rate"] == pytest.approx(0.022745)
    assert row["fitted_yield"] == pytest.approx(0.041518)
    # Decomposition identity: TP + expected rate == fitted yield (SF Fed convention)
    assert row["term_premium"] + row["expected_rate"] == pytest.approx(
        row["fitted_yield"], abs=1e-6
    )


def test_parse_frbsf_missing_sheet_raises():
    dates = pd.to_datetime(["2010-01-04"])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"DATE": dates, "FTERMP02YR": [0.01]}).to_excel(
            xw, sheet_name="Fit Term Premium", index=False
        )
    with pytest.raises(ValueError):
        parse_frbsf_model_data(buf.getvalue())
