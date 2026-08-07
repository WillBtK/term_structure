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

# Maturities (years) on which the pass/fail gate is evaluated. The overall
# comparison reports every overlapping maturity; the gate focuses on 5y and 10y
# where the term premium is most economically meaningful.
GATE_MATURITIES = [5.0, 10.0]

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
