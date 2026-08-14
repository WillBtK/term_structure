"""Japan-specific configuration for the term-premium pipeline.

Data: Ministry of Finance (MoF) JGB spot yields — published *directly* by
maturity (no Svensson parameters to reconstruct), Shift-JIS CSV, Japanese-era
dates. The engine in ``/model`` is country-agnostic; ``/japan`` produces the same
data contract (a date x maturity panel of zero yields in decimal).

Japan-specific handling (see validation/FINDINGS_japan.md):
- **Ragged maturities.** The curve reaches different tenors from different dates
  (10y only from 1986-07; 20y 1986-12; 30y 1999-09) — handled by the EKF's
  per-date maturity masking (as for the US 1960s long end).
- **QQE/YCC distortion.** Under Yield Curve Control (2016-09 → 2024-03) the BoJ
  price-pegged the 10y, so it was not a market price. We downweight the pegged
  maturities in the measurement over that window (heaviest while the band was
  tight), so the peg does not distort the estimated expected-rate path; the term
  premium is then reported against *observed* yields (the realised, policy-
  suppressed premium).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Data source: MoF "Historical Data / JGB interest rates" (full history CSV).
# ---------------------------------------------------------------------------
MOF_JGB_URL = os.environ.get(
    "MOF_JGB_URL",
    "https://www.mof.go.jp/jgbs/reference/interest_rate/data/jgbcm_all.csv")

# Column order after the date in jgbcm_all.csv (years).
MOF_TENORS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40]

# First available date per tenor (user-confirmed against MoF).
TENOR_START = {
    1: "1974-09-24", 2: "1974-09-24", 3: "1974-09-24", 4: "1974-09-24",
    5: "1974-09-24", 6: "1974-09-24", 7: "1974-09-24", 8: "1974-09-24",
    9: "1974-09-24", 10: "1986-07-05", 15: "1991-08-30", 20: "1986-12-01",
    25: "2004-03-22", 30: "1999-09-02", 40: "2007-11-06",
}

# Estimation grid (years): the liquid tenors that give factor + long-end
# identification. 10/20/30 join when available (ragged); 4/6/8/9/15/25/40 held
# out as redundant/thin. Sub-1y (T-bills) is a documented refinement.
COMMON_GRID = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)

# Sample start: 1986-07 (10y available) covers the 1989-90 bubble peak/collapse
# and the 1998 Trust Fund Bureau (VaR) shock. Extensible to 1974 using <=9y.
SAMPLE_START = "1986-07-05"

# ---------------------------------------------------------------------------
# Effective lower bound (decimal). NIRP policy rate was -0.10%; short JGB yields
# floored around -0.1% to -0.35%. The charter's -0.75% is a reversal-rate concept,
# deeper than the revealed floor — so we START near the NIRP rate and ESTIMATE the
# ELB by ML (well identified: rates sat at the bound for years).
# ---------------------------------------------------------------------------
ELB = -0.002
ELB_BAND = (-0.004, 0.000)

# ---------------------------------------------------------------------------
# BoJ policy dates (for the YCC downweighting schedule + annotation).
# ---------------------------------------------------------------------------
QQE_START = pd.Timestamp("2013-04-04")     # quantity easing (real premium compression)
NIRP_START = pd.Timestamp("2016-02-16")    # -0.10% on policy balances
YCC_START = pd.Timestamp("2016-09-21")     # 10y price peg begins
YCC_BAND_050 = pd.Timestamp("2022-12-20")  # band widened to +/-0.50%
YCC_BAND_100 = pd.Timestamp("2023-07-28")  # effective +/-1.0%
YCC_END = pd.Timestamp("2024-03-19")       # YCC + NIRP ended

# Maturities whose market price the 10y peg directly controlled (10y, and its
# immediate neighbours via arbitrage). The long end (20y/30y) stayed market-driven.
YCC_PEGGED_TENORS = (7.0, 10.0)


def ycc_downweight(dates, grid) -> np.ndarray:
    """Per-date, per-maturity measurement-variance *multiplier* implementing the
    YCC downweight. Returns (T, m): 1.0 normally, and a large factor on the pegged
    tenors during YCC — heaviest while the band was tight, tapering as it widened —
    so the EKF effectively ignores the pegged 10y as a market price. The factor
    multiplies h^2, so it scales with the estimated measurement variance.
    """
    dates = pd.DatetimeIndex(dates)
    grid = np.asarray(grid, dtype=float)
    mult = np.ones((len(dates), grid.size))
    peg_cols = [j for j, m in enumerate(grid) if m in YCC_PEGGED_TENORS]
    if not peg_cols:
        return mult
    for t, d in enumerate(dates):
        if d < YCC_START or d >= YCC_END:
            continue
        if d < YCC_BAND_050:      # tight band -> effectively ignore the peg
            f = 1.0e8
        elif d < YCC_BAND_100:    # +/-0.50% band
            f = 1.0e4
        else:                     # ~+/-1.0% band: 10y increasingly market-driven
            f = 1.0e2
        mult[t, peg_cols] = f
    return mult


# Cache paths (contents gitignored; directory tracked).
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV_PATH = DATA_DIR / "jgbcm_all.csv"
YIELDS_PARQUET_PATH = DATA_DIR / "mof_jgb_yields.parquet"


# Japanese-era -> Gregorian base years (era year n -> base + n).
_ERA_BASE = {"M": 1867, "T": 1911, "S": 1925, "H": 1988, "R": 2018}


def parse_jp_date(s: str) -> "pd.Timestamp":
    """Parse a MoF Japanese-era date like 'S49.9.24' or 'R6.3.19'."""
    s = s.strip()
    era, rest = s[0], s[1:]
    y, m, d = rest.split(".")
    return pd.Timestamp(_ERA_BASE[era] + int(y), int(m), int(d))
