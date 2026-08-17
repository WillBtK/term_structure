"""Canada-specific configuration for the term-premium pipeline.

Data: Bank of Canada fitted **zero-coupon** government curve (Bolder–Johnson–Metzler /
Bolder–Streliski method), published as spot zero-coupon yields directly at 0.25-year
steps from 0.25y to 30y via the legacy ``/stats/results`` CSV endpoint. Like the
SNB/MoF/BoE curves this is already a fitted spot curve — no parametric reconstruction,
just parse and grid. Daily from 1986; the 30y point is ragged (joins 1991).

Canada grazed its effective lower bound in 2009–10 and 2020–21 (the BoC held the
overnight rate at 0.25%, and the fitted short end dipped to ~0.05%), so it uses the
shadow-rate model. Because Canada only lightly touched the bound, the ELB is **weakly
identified** (like the US) — we calibrate it just below the observed floor (0%) and
ML-refine.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Bank of Canada "Yield curves for zero-coupon bonds" — the legacy stats CSV endpoint.
# GET <BOC_YC_URL>?lookupPage=<BOC_YC_LOOKUP>&startRange=<BOC_YC_START>&searchRange=all
# returns a clean CSV: "Date, ZC025YR, ZC050YR, ... , ZC3000YR" (0.25y..30y at 0.25y
# steps) with values ALREADY IN DECIMAL. NOTE: any invalid ``searchRange`` value
# 302-redirects to a maintenance page — use ``all`` (full history) or an empty value
# with dFrom/dTo. Reachable directly (no proxy allowlist needed).
BOC_YC_URL = os.environ.get("BOC_YC_URL", "https://www.bankofcanada.ca/stats/results/csv")
BOC_YC_LOOKUP = os.environ.get("BOC_YC_LOOKUP", "lookup_yield_curve.php")
BOC_YC_START = "1986-01-01"          # dataset start (the ``startRange`` param)

# Estimation grid (years). The BoC ZC curve carries every 0.25y node to 30y; we take
# the liquid spread incl. the long end (Canada is one of the few with a 30y point).
# 30y is ragged (joins 1991) — handled by the EKF's per-date masking.
COMMON_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)

# Modern sample: the BoC formally adopted inflation targeting in 1991, a clean
# monetary-policy regime break, and the full 30y curve begins the same year.
SAMPLE_START = "1991-01-01"

# Effective lower bound (decimal). Canada never set a negative policy rate; the BoC
# floored the overnight rate at 0.25% (2009–10, 2020–21) and the fitted zero-coupon
# short end bottomed near 0.05%. Calibrate the floor just below that (0%) and ML-refine;
# expect weak identification (Canada only grazed the bound), like the US.
ELB = 0.0
ELB_BAND = (-0.005, 0.0025)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV_PATH = DATA_DIR / "boc_yield_curve.csv"
YIELDS_PARQUET_PATH = DATA_DIR / "boc_yields.parquet"
