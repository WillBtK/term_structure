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
# The SF Fed "Treasury Yield Premiums" model *estimates* on the same eight
# maturities as our COMMON_GRID (3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y), but it only
# *publishes* the term-premium / expected-rate decomposition for the 2-year and
# 10-year yields. The gate therefore evaluates 2y and 10y — the maturities where
# a benchmark term premium actually exists. (The full-grid fitted-yield
# comparison in compare_afns still runs on every overlapping maturity.)
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
# The exact download URL and column layout MUST be confirmed against the live
# file (frbsf.org is egress-blocked in some environments). Set SF_FED_URL to the
# real download link; see sf_fed.py for the configurable column mapping.
# ---------------------------------------------------------------------------
SF_FED_URL = os.environ.get("SF_FED_URL", "")

DATA_DIR = Path(__file__).resolve().parent / "reports"
SF_FED_RAW_PATH = DATA_DIR / "sf_fed_raw"          # extension added by fetcher
SF_FED_PARQUET_PATH = DATA_DIR / "sf_fed_series.parquet"
REPORTS_DIR = DATA_DIR
