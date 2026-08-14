"""Tests for the SNB rendoblim parser (no network)."""
import numpy as np
import pandas as pd
import pytest

from switzerland.fetch_snb import parse_snb_csv

FIXTURE = (
    '"CubeId";"rendoblim"\n'
    '"PublishingDate";"2025-09-01 14:29"\n'
    "\n"
    '"Date";"D0";"Value"\n'
    '"1988-01";"1J";"2.887"\n'
    '"1988-01";"10J";"4.092"\n'
    '"1988-01";"30J";\n'                 # 30y missing early (ragged)
    '"2015-01";"1J";"-0.75"\n'           # negative-rate era
    '"2015-01";"10J";"-0.05"\n'
    '"2015-01";"30J";"0.35"\n'
)


def test_parse_units_pivot_ragged():
    df = parse_snb_csv(FIXTURE)
    assert list(df.index) == [pd.Timestamp("1988-01-31"), pd.Timestamp("2015-01-31")]
    # percent -> decimal
    assert df.loc[pd.Timestamp("1988-01-31"), 1.0] == pytest.approx(0.02887)
    assert df.loc[pd.Timestamp("1988-01-31"), 10.0] == pytest.approx(0.04092)
    # 30y NaN in 1988, present in 2015 (ragged long end)
    assert np.isnan(df.loc[pd.Timestamp("1988-01-31"), 30.0])
    assert df.loc[pd.Timestamp("2015-01-31"), 30.0] == pytest.approx(0.0035)
    # negative short rate parsed
    assert df.loc[pd.Timestamp("2015-01-31"), 1.0] == pytest.approx(-0.0075)
