"""Tests for the SNB spot-curve parser, legacy + successor cube layouts (no network)."""
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


# Successor cube (rendeiduebd/m): D0 = bond CATEGORY, D1 = maturity. Only the Swiss
# Confederation category ("CHF") is ours — cantons/banks/corporates must be excluded.
FIXTURE_NEW = (
    '"CubeId";"rendeiduebd"\n'
    '"PublishingDate";"2026-08-03 14:30"\n'
    "\n"
    '"Date";"D0";"D1";"Value"\n'
    '"2026-07-30";"CHF";"1J";"0.02"\n'
    '"2026-07-30";"CHF";"10J";"0.47"\n'
    '"2026-07-30";"KTB";"10J";"0.90"\n'      # cantons — must be dropped
    '"2026-07-30";"IHB";"10J";"1.48"\n'      # corporates — must be dropped
    '"2026-07-31";"CHF";"1J";"0.03"\n'
    '"2026-07-31";"CHF";"10J";"0.475"\n'
    '"2026-07-31";"CHF";"30J";"0.585"\n'
    '"2026-07-31";"EUR";"10J";"2.60"\n'      # euro-area govt — must be dropped
)


def test_successor_layout_selects_confederation_only():
    df = parse_snb_csv(FIXTURE_NEW, freq="D")
    assert list(df.index) == [pd.Timestamp("2026-07-30"), pd.Timestamp("2026-07-31")]
    # the Confederation 10y, NOT the cantonal/corporate/euro-area rows at the same tenor
    assert df.loc[pd.Timestamp("2026-07-31"), 10.0] == pytest.approx(0.00475)
    assert df.loc[pd.Timestamp("2026-07-30"), 10.0] == pytest.approx(0.0047)
    assert df.loc[pd.Timestamp("2026-07-31"), 30.0] == pytest.approx(0.00585)
    # 30y absent on the first date -> ragged NaN, not a borrowed value
    assert np.isnan(df.loc[pd.Timestamp("2026-07-30"), 30.0])


def test_both_layouts_supported_by_one_parser():
    """The legacy (D0=maturity) and successor (D0=category, D1=maturity) cubes must both
    parse, so a cached legacy CSV still loads after the cube switch."""
    old = parse_snb_csv(FIXTURE)
    new = parse_snb_csv(FIXTURE_NEW, freq="D")
    assert not old.empty and not new.empty
    assert 10.0 in old.columns and 10.0 in new.columns
