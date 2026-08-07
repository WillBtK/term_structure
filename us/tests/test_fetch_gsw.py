"""Tests for GSW parsing against realistic feds200628 quirks (no network).

The real Fed Board file has:
  * ~9 descriptive preamble lines before the 'Date' header row,
  * missing-value sentinels (NA / ND / blank / #N/A),
  * early-history rows (pre-~1980) with only the 3-factor NS parameters, i.e.
    BETA3 / TAU2 blank.
"""
import numpy as np
import pandas as pd
import pytest

from us.fetch_gsw import parse_gsw_csv
from us.reconstruct import reconstruct_grid

# A compact but faithful fixture: preamble, header, then rows exercising each
# quirk. Columns mirror the real file's naming (subset).
FIXTURE = (
    "This is the descriptive preamble for the GSW feds200628 file.\n"
    "It spans several lines, some with, embedded commas, like this.\n"
    "Yields are continuously compounded, in percent.\n"
    "Date,BETA0,BETA1,BETA2,BETA3,TAU1,TAU2,SVENY10\n"
    # modern 4-factor Svensson row (golden params -> SVENY10 = 3.0051181545...)
    "2020-01-02,3.0,-1.0,2.0,-0.5,1.5,8.0,3.0051181545012904\n"
    # pre-1980 style: 3-factor NS only, BETA3 & TAU2 blank
    "1975-06-16,7.5,0.8,-1.2,,2.0,,\n"
    # a row using ND / NA sentinels for missing values
    "1976-01-05,ND,NA,ND,NA,ND,NA,ND\n"
    # a malformed date row -> dropped
    "not-a-date,3.0,-1.0,2.0,-0.5,1.5,8.0,3.0\n"
)


def test_preamble_skipped_and_dates_parsed():
    df = parse_gsw_csv(FIXTURE)
    # malformed-date row dropped; all-missing sentinel row dropped (all params NaN)
    assert list(df.index) == [pd.Timestamp("1975-06-16"), pd.Timestamp("2020-01-02")]
    assert df.index.name == "date"


def test_sentinels_coerced_to_nan():
    df = parse_gsw_csv(FIXTURE)
    # The 1976 row was all-sentinel -> dropped entirely by the all-NaN filter.
    assert pd.Timestamp("1976-01-05") not in df.index


def test_pre1980_ns_only_row_kept_in_params():
    df = parse_gsw_csv(FIXTURE)
    row = df.loc[pd.Timestamp("1975-06-16")]
    # NS params present, Svensson curvature absent.
    assert row["BETA0"] == pytest.approx(7.5)
    assert np.isnan(row["BETA3"])
    assert np.isnan(row["TAU2"])


def test_reconstruction_keeps_modern_drops_ns_only():
    df = parse_gsw_csv(FIXTURE)
    panel = reconstruct_grid(df, np.array([10.0]))
    # Only the modern 4-factor row can be reconstructed (TAU2 > 0 required).
    assert list(panel.index) == [pd.Timestamp("2020-01-02")]
    # And it matches the published SVENY10 (decimal).
    assert panel.iloc[0][10.0] == pytest.approx(3.0051181545012904 / 100.0, abs=1e-12)
