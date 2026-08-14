"""UK-specific configuration for the term-premium pipeline.

Data: Bank of England **Government Liability Curve (GLC)** — the official fitted
UK government (gilt) yield curves, published as *spot rates directly* (VRP spline;
no parametric reconstruction, like Japan's MoF). Three curves come from the same
month-end archive: **nominal** (gilts), **real** (index-linked gilts), and
**implied inflation** (breakeven). Each is a zip of Excel workbooks split by era.

Coverage (month-end): nominal from **1970** (long end ragged; ~1979 out to 20y,
extending to 40y from 2016), real from the mid-1980s, all on a 0.5y-step grid.

UK-specific notes (see validation/FINDINGS_uk.md):
- **Mild ELB.** The Bank Rate floored at 0.10% (2020-21) and short gilt yields
  only briefly dipped slightly negative, so the ELB is weakly identified — we fix
  it (like the US) rather than estimate. The shadow-rate machinery barely binds.
- **Real curve has no ELB** (no "real cash"); the real TP is a plain-AFNS
  decomposition (bound pushed to -100%), as for US TIPS.
- **RPI indexation lag.** UK index-linked gilts are RPI-linked with an indexation
  lag, so the very short real end carries known near-term inflation; we start the
  real grid at 2y and caveat it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# BoE GLC month-end data (self-healing discovery + stable defaults).
# ---------------------------------------------------------------------------
BOE_YIELD_PAGE = "https://www.bankofengland.co.uk/statistics/yield-curves"
BOE_MEDIA_BASE = os.environ.get(
    "BOE_MEDIA_BASE",
    "https://www.bankofengland.co.uk/-/media/boe/files/statistics/yield-curves")
BOE_NOMINAL_ZIP = "glcnominalmonthedata.zip"
BOE_REAL_ZIP = "glcrealmonthedata.zip"
BOE_INFLATION_ZIP = "glcinflationmonthedata.zip"
# Daily GLC archives (same "4. spot curve" workbook layout, daily-dated rows).
BOE_NOMINAL_ZIP_D = "glcnominalddata.zip"
BOE_REAL_ZIP_D = "glcrealddata.zip"
BOE_INFLATION_ZIP_D = "glcinflationddata.zip"

# The spot-curve sheet inside each workbook, and the label marking the maturity
# (in years) header row.
SPOT_SHEET = "4. spot curve"
YEARS_LABEL = "years:"

# ---------------------------------------------------------------------------
# Estimation grids (years). Nominal covers the liquid gilt curve incl. the ragged
# long end (20y joins ~1979, 30y from 2016). Real starts at 2y (RPI-lag short end).
# ---------------------------------------------------------------------------
COMMON_GRID = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0], dtype=float)
# The BoE real (index-linked) curve has no sub-3y point (RPI-lag short end), so
# the real grid starts at 3y; the IRP overlaps the nominal at 3/5/7/10/20y.
REAL_GRID = np.array([3.0, 5.0, 7.0, 10.0, 20.0], dtype=float)

# Sample starts. Full nominal history reaches 1970; the modern production sample
# starts at UK **inflation targeting** (post-ERM, Oct 1992) — the UK analog of the
# US-1987 / Germany-1999 regime-clean modern sample. The pre-1992 stagflation era
# is a separate historical exhibit. Real (index-linked) history from the mid-1980s.
PROD_START = "1992-10-01"
HIST_START = "1979-01-01"
REAL_SAMPLE_START = "1985-01-01"

# ---------------------------------------------------------------------------
# Effective lower bound. UK is the mildest ZLB of the four; Bank Rate floored at
# 0.10% and short gilt yields barely went negative, so the ELB is weakly
# identified -> FIXED (like the US), not ML-estimated. Decimal.
# ---------------------------------------------------------------------------
ELB = -0.0035
ELB_REAL = -1.0        # no bound on real rates (never binds -> plain AFNS)

# ---------------------------------------------------------------------------
# Cache paths (contents gitignored; directory tracked).
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_DIR = DATA_DIR / "raw"
NOMINAL_PARQUET = DATA_DIR / "boe_nominal.parquet"
REAL_PARQUET = DATA_DIR / "boe_real.parquet"
INFLATION_PARQUET = DATA_DIR / "boe_inflation.parquet"
