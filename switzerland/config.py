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

# SNB data-portal cube of Swiss Confederation spot rates (Svensson method).
SNB_CUBE_URL = os.environ.get(
    "SNB_CUBE_URL", "https://data.snb.ch/api/cube/rendoblim/data/csv/en")

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
YIELDS_PARQUET_PATH = DATA_DIR / "snb_yields.parquet"
