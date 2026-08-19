"""Tests for the MoF JGB parser (era dates, ragged tenors, units) — no network."""
import numpy as np
import pandas as pd
import pytest

from japan import config as cfg
from japan.fetch_mof import parse_jgb_csv

# Compact fixture in the real jgbcm_all.csv shape: title, header, era-dated rows
# with '-' for unavailable tenors. Columns: 1..10,15,20,25,30,40 (years).
_HDR = "基準日," + ",".join(f"{t}年" for t in cfg.MOF_TENORS)
FIXTURE = (
    "国債金利情報,,,,,,,,,,,,,,," + "(単位 : %)\n"
    + _HDR + "\n"
    # 1974: only 1-9y (10y+ '-')
    "S49.9.24,10.327,9.362,8.83,8.515,8.348,8.29,8.24,8.121,8.127,-,-,-,-,-,-\n"
    # 1990 (Heisei 2): 10y & 20y present, 15y/25y/30y/40y '-'
    "H2.9.28,8.30,8.40,8.35,8.30,8.25,8.20,8.15,8.12,8.11,8.11,-,8.60,-,-,-\n"
    # 2024 (Reiwa 6): full-ish
    "R6.3.19,0.10,0.18,0.20,0.30,0.45,0.60,0.72,0.80,0.88,0.79,1.30,1.55,1.70,1.85,2.10\n"
)


def test_era_dates_parse():
    assert cfg.parse_jp_date("S49.9.24") == pd.Timestamp(1974, 9, 24)
    assert cfg.parse_jp_date("H2.9.28") == pd.Timestamp(1990, 9, 28)
    assert cfg.parse_jp_date("R6.3.19") == pd.Timestamp(2024, 3, 19)


def test_parse_units_and_ragged():
    df = parse_jgb_csv(FIXTURE)
    assert list(df.index) == [pd.Timestamp(1974, 9, 24), pd.Timestamp(1990, 9, 28),
                              pd.Timestamp(2024, 3, 19)]
    # decimal units
    assert df.loc[pd.Timestamp(1974, 9, 24), 1.0] == pytest.approx(0.10327)
    # ragged: 10y missing in 1974, present in 1990
    assert np.isnan(df.loc[pd.Timestamp(1974, 9, 24), 10.0])
    assert df.loc[pd.Timestamp(1990, 9, 28), 10.0] == pytest.approx(0.0811)
    # '-' -> NaN (15y missing in 1990)
    assert np.isnan(df.loc[pd.Timestamp(1990, 9, 28), 15.0])
    assert df.loc[pd.Timestamp(2024, 3, 19), 40.0] == pytest.approx(0.021)


# Current-year file (jgbcm.csv): different title incl. year/month, and a trailing
# Japanese note line + blank line that must be skipped.
CURRENT_FIXTURE = (
    "国債金利情報 (令和8年8月),,,,,,,,,,,,,,,(単位 : %)\n"
    + _HDR + "\n"
    "R8.7.31,1.255,1.507,1.658,1.876,2.044,2.19,2.343,2.517,2.658,2.7,3.0,3.2,3.3,3.4,3.5\n"
    "R8.8.18,1.431,1.691,1.844,2.038,2.177,2.332,2.477,2.642,2.79,2.8,3.1,3.3,3.4,3.5,3.6\n"
    ",,,,,,,,,,,,,,,\n"
    "※最新のcsvデータがダウンロードできない場合、ブラウザのキャッシュを削除してください\n"
)


def test_current_year_parse_skips_note_and_blank():
    cur = parse_jgb_csv(CURRENT_FIXTURE)
    assert list(cur.index) == [pd.Timestamp(2026, 7, 31), pd.Timestamp(2026, 8, 18)]
    assert cur.loc[pd.Timestamp(2026, 8, 18), 10.0] == pytest.approx(0.028)


def test_history_plus_current_merge_prefers_current():
    """The load_yields merge rule: concat history + current-year, keep='last' on
    duplicate dates so the daily current-year file wins and extends the panel."""
    hist = parse_jgb_csv(FIXTURE)                       # ends 2024-03-19
    cur = parse_jgb_csv(CURRENT_FIXTURE)               # 2026-07-31, 2026-08-18
    merged = pd.concat([hist, cur])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    assert merged.index.max() == pd.Timestamp(2026, 8, 18)     # extended to the daily edge
    # an overlapping date would take the current-year value (keep='last'); check ordering
    assert merged.index.is_monotonic_increasing and merged.index.is_unique


def test_ycc_downweight_schedule():
    grid = cfg.COMMON_GRID
    dates = pd.to_datetime(["2015-06-30", "2019-08-31", "2023-09-30", "2025-06-30"])
    mult = cfg.ycc_downweight(dates, grid)
    j10 = list(grid).index(10.0)
    assert mult[0, j10] == 1.0            # pre-YCC: no downweight
    assert mult[1, j10] > 1.0             # tight YCC: heavy downweight
    assert mult[2, j10] > 1.0             # loose YCC (2023): still downweighted, lighter
    assert mult[1, j10] > mult[2, j10]    # tighter band -> heavier downweight
    assert mult[3, j10] == 1.0            # post-YCC (2025): none
    j30 = list(grid).index(30.0)
    assert mult[1, j30] == 1.0            # 30y never pegged
