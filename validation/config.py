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
# Pass criteria.
#
# Fitted yields are held to a fitted-grade bar (they already pass): the curve
# fit should essentially reproduce SF Fed's.
MIN_CORRELATION_FIT = 0.98
#
# The TERM PREMIUM is judged against a realistic "close tracking" bar, not the
# fitted-yield grade. The premium is the weakly-identified P-vs-Q split
# (Kim-Orphanides; Bauer-Rudebusch-Wu), so an independent re-estimation cannot be
# expected to reproduce SF Fed's premium to a few bp even when the model, spec
# and data match. We require good co-movement and a bounded level gap:
#   * term-premium correlation with SF Fed >= MIN_CORRELATION_TP at gate maturities
#   * absolute mean (signed) level bias <= MAX_ABS_BIAS_BP_TP at gate maturities
MIN_CORRELATION_TP = 0.90
MAX_ABS_BIAS_BP_TP = 25.0

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

# ---------------------------------------------------------------------------
# External US term-premium benchmark series (for side-by-side plotting and as
# validation references). Pulled alongside the underlying rates. All parsed to the
# canonical decimal schema (date, maturity, ...) by validation/benchmarks.py.
#
#   ACM  — Adrian-Crump-Moench (NY Fed), 5-factor, daily 1961+, maturities 1-10y.
#          The ".csv" endpoint actually serves a legacy .xls workbook (sheets
#          "ACM Daily"/"ACM Monthly"; columns ACMTP/ACMRNY/ACMY 01..10, percent).
#   KW   — Kim-Wright (Fed Board 3-factor), FRED series THREEFYTP01..10 (term
#          premium, percent, daily 1990+); one keyless CSV per maturity.
#   DKW  — D'Amico-Kim-Wei (Fed Board), daily 1983+; 5y/10y/5f5 decomposition of
#          nominal + real. Nominal term premium = real.term.prem + inflation.risk.prem.
# Christensen-Rudebusch (SF Fed) is handled separately in validation/sf_fed.py.
# ---------------------------------------------------------------------------
ACM_URL = os.environ.get(
    "ACM_URL",
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.csv")
KIM_WRIGHT_FRED_BASE = os.environ.get(
    "KIM_WRIGHT_FRED_BASE", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=")
KIM_WRIGHT_IDS = [f"THREEFYTP{n}" for n in range(1, 11)]   # 1y..10y term premium (no zero-pad)
DKW_URL = os.environ.get(
    "DKW_URL",
    "https://www.federalreserve.gov/econres/notes/feds-notes/DKW_updates.csv")

ACM_RAW_PATH = DATA_DIR / "acm_raw.xls"
ACM_PARQUET_PATH = DATA_DIR / "acm_series.parquet"
KIM_WRIGHT_PARQUET_PATH = DATA_DIR / "kim_wright_series.parquet"
DKW_RAW_PATH = DATA_DIR / "dkw_updates.csv"
DKW_PARQUET_PATH = DATA_DIR / "dkw_series.parquet"

# ---------------------------------------------------------------------------
# ATSIX — Aruoba Term Structure of Inflation Expectations (Philadelphia Fed).
#   A smooth term structure of *survey-based* expected inflation, 3-120 months
#   ahead, monthly from 1998-01. Built (Aruoba 2020, JBES) from SPF + Blue Chip +
#   Michigan + other surveys via a dynamic Nelson-Siegel state-space model — i.e.
#   the same survey information DKW anchor to (incl. Blue Chip, which is otherwise
#   paywalled), delivered free as a ready term structure. `infexp{h}` = expected
#   AVERAGE inflation over the next h months (a horizon-average, matching a
#   zero-coupon yield convention and our model's expected_inflation(τ) object).
#   Used to anchor the joint nominal/real model's IRP level (US-only overlay; see
#   FINDINGS_phase3.md Strand E). The `Real` sheet (expected real short-rate path)
#   is a free cross-check on the real block.
# ---------------------------------------------------------------------------
ATSIX_URL = os.environ.get(
    "ATSIX_URL",
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/atsix/"
    "ATSIX_Vintages.xlsx?la=en&sc_lang=en&hash=EF53EB526F14CD928F9FCEB7F9A33946")
ATSIX_RAW_PATH = DATA_DIR / "atsix_vintages.xlsx"
ATSIX_PARQUET_PATH = DATA_DIR / "atsix_series.parquet"

# ---------------------------------------------------------------------------
# TIPS liquidity "noise" proxy (Hu-Pan-Wang 2013 applied to the TIPS cross-section),
# built from FRED individual TIPS securities via the official API. A price-based,
# non-circular liquidity observable to identify the joint model's IRP dynamics
# (validation/tips_liquidity.py). FRED_API_KEY read from the environment.
# ---------------------------------------------------------------------------
TIPS_NOISE_RAW_PARQUET = DATA_DIR / "tips_individual_yields.parquet"
TIPS_NOISE_PARQUET = DATA_DIR / "tips_noise_daily.parquet"

# ---------------------------------------------------------------------------
# German Bund liquidity "noise" gauge (validation/bund_liquidity.py) from the
# Bundesbank per-ISIN "Prices and yields of listed Federal securities" monthly
# workbooks (germany/fetch_bund_prices.py). Bundesbank index URL overridable.
# ---------------------------------------------------------------------------
BUND_PRICES_INDEX = os.environ.get(
    "BUND_PRICES_INDEX",
    "https://www.bundesbank.de/dynamic/action/en/service/federal-securities/"
    "prices-and-yields/810710/prices-and-yields-of-listed-federal-securities")
BUND_PRICES_RAW_PARQUET = DATA_DIR / "bund_individual_yields.parquet"
BUND_NOISE_PARQUET = DATA_DIR / "bund_noise_daily.parquet"
