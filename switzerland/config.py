"""Switzerland-specific configuration for the term-premium pipeline.

Data: Swiss National Bank estimated government yield curve (Svensson–Nelson–Siegel
method), published as **spot rates directly** by maturity via the SNB data portal
cube ``rendoblim`` — so, like Japan/UK, there is no parametric reconstruction; we
just parse and grid. Monthly from 1988; the 30y point is ragged (from 1997-12).

Switzerland is the **deepest-ELB** market of the set: the SNB policy rate reached
−0.75% (2015-2022) and short CHF yields went well below zero, so the effective
lower bound is well identified — the definitive shadow-rate / ELB stress test. We
start the ELB at −0.75% and estimate it by ML.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# SNB data-portal cubes of Swiss Confederation spot rates (Svensson method).
# ``rendeiduebd`` = daily, ``rendeiduebm`` = monthly. These SUPERSEDE the old
# ``rendoblid``/``rendoblim`` cubes, which SNB froze at 2025-07-31 (last published
# 2025-09-01) — the fetch kept succeeding against a dead cube, silently serving a
# year-old curve. The successor is verified to be the same series: over the 7,534
# overlapping days every shared tenor matches the old cube to 0.000 bp. It also
# carries a NATIVE 10y (the old cube published 15y instead, so 10y had to be
# interpolated). Layout differs: D0 = bond category (we take ``CHF`` = Swiss
# Confederation bond issues), D1 = maturity code (``1J``..``30J``).
SNB_CUBE_URL = os.environ.get(
    "SNB_CUBE_URL", "https://data.snb.ch/api/cube/rendeiduebm/data/csv/en")
SNB_CUBE_URL_D = os.environ.get(
    "SNB_CUBE_URL_D", "https://data.snb.ch/api/cube/rendeiduebd/data/csv/en")
# D0 code selecting the Swiss Confederation curve in the successor cubes.
SNB_CONFED_CODE = os.environ.get("SNB_CONFED_CODE", "CHF")

# Estimation grid (years). SNB publishes 1-10, 15, 20, 30y; we take the liquid
# spread. 30y is ragged (joins 1997-12) — handled by the EKF's per-date masking.
COMMON_GRID = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)

SAMPLE_START = "1988-01-01"

# Effective lower bound (decimal). SNB policy rate floored at −0.75% (2015-2022),
# the deepest of the four majors; short CHF yields went clearly negative, so the
# ELB is well identified — start at −0.75% and ML-estimate.
ELB = -0.0075
ELB_BAND = (-0.010, -0.005)

DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV_PATH = DATA_DIR / "snb_rendoblim.csv"
RAW_CSV_PATH_D = DATA_DIR / "snb_rendoblid.csv"
YIELDS_PARQUET_PATH = DATA_DIR / "snb_yields.parquet"
YIELDS_PARQUET_PATH_D = DATA_DIR / "snb_yields_daily.parquet"
