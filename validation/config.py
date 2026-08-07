"""Configuration for benchmark validation.

Phase 1 gate: the plain (non-shadow-rate) AFNS fit on US GSW data must track the
SF Fed "Treasury Yield Premiums" series (same model family, same input) before
we add the shadow-rate extension.
"""
from __future__ import annotations

import os
from pathlib import Path

# Canonical quantities compared, all in DECIMAL (e.g. 0.0123).
QUANTITIES = ["term_premium", "expected_rate", "fitted_yield"]

# Maturities (years) on which the pass/fail gate is evaluated.
#
# The SF Fed "Treasury Yield Premiums" model estimates on the same eight
# maturities as our COMMON_GRID (3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y). Its published
# model-data workbook gives the full decomposition at integer maturities 1..10y,
# which overlaps our grid at {1, 2, 3, 5, 7, 10}y — the comparison runs on all of
# those. The pass/fail gate focuses on 2y and 10y, the canonical headline
# maturities for the term premium.
GATE_MATURITIES = [2.0, 10.0]

# ---------------------------------------------------------------------------
# Pass criteria (PROPOSED — confirm with the project owner once the first real
# numbers are in; see top-level README "Open items").
#   * term-premium correlation with SF Fed >= MIN_CORRELATION at gate maturities
#   * mean absolute term-premium difference <= MAX_MEAN_ABS_DIFF_BP at gate
#     maturities
# ---------------------------------------------------------------------------
MIN_CORRELATION = 0.98
MAX_MEAN_ABS_DIFF_BP = 15.0

# ---------------------------------------------------------------------------
# SF Fed Treasury Yield Premiums source.
#   Page: https://www.frbsf.org/research-and-insights/data-and-indicators/
#         treasury-yield-premiums/
#   File: FRBSF_Term_Model_Data.xlsx (full model output; verified 2026-08).
# Override via SF_FED_URL if the Bank moves the file; sf_fed.py falls back to its
# built-in FRBSF_MODEL_DATA_URL when this is empty.
# ---------------------------------------------------------------------------
SF_FED_URL = os.environ.get(
    "SF_FED_URL",
    "https://www.frbsf.org/wp-content/uploads/FRBSF_Term_Model_Data.xlsx",
)

DATA_DIR = Path(__file__).resolve().parent / "reports"
SF_FED_RAW_PATH = DATA_DIR / "sf_fed_raw"          # extension added by fetcher
SF_FED_PARQUET_PATH = DATA_DIR / "sf_fed_series.parquet"
REPORTS_DIR = DATA_DIR
