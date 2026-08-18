"""US-specific configuration for the term-premium pipeline.

Everything country-specific about the US leg lives here: the data source, the
maturity grid we reconstruct onto, the effective lower bound, and cache paths.
The engine in ``/model`` is country-agnostic and reads none of this directly —
``/us`` uses it to produce the engine's data contract (see ``fetch_gsw`` /
``reconstruct``).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Maturity grid (years). Common across all four countries for comparability
# (top-level README): 3m / 6m / 1y / 2y / 3y / 5y / 7y / 10y.
# ---------------------------------------------------------------------------
COMMON_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)

# ---------------------------------------------------------------------------
# Effective lower bound for the (later) shadow-rate extension. US = -0.35%,
# fixed (top-level README). Stored in DECIMAL to match the yield convention.
# Not used by the plain-AFNS baseline; kept here so the US config is complete.
# ---------------------------------------------------------------------------
ELB = -0.0035

# ---------------------------------------------------------------------------
# Data source: Federal Reserve Board GSW ("The U.S. Treasury Yield Curve: 1961
# to the Present", Gurkaynak-Sack-Wright, FEDS 2006-28). Daily Svensson params.
#
# Override the URL via env var GSW_URL when a mirror is used or the Board moves
# the file again (it last moved on 2019-11-05).
#
# NOTE: in some execution environments www.federalreserve.gov is blocked by an
# egress proxy. ``fetch_gsw`` degrades to the local cache in that case; run the
# fetch where egress to federalreserve.gov is open (or point GSW_URL at a
# mirror) to refresh.
# ---------------------------------------------------------------------------
GSW_URL = os.environ.get(
    "GSW_URL",
    "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
)

# The six Svensson parameters as named in the feds200628 file.
# BETA0-3 are in PERCENT; TAU1-2 are in YEARS.
PARAM_COLUMNS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]

# Published fitted zero-coupon yields (percent) in the same file, used only as a
# reconstruction cross-check (see reconstruct.cross_check_against_published).
# SVENYNN = continuously-compounded zero yield at NN years.
PUBLISHED_YIELD_PREFIX = "SVENY"

# ---------------------------------------------------------------------------
# Cache paths. Contents are gitignored; the directory itself is tracked.
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
RAW_CSV_PATH = DATA_DIR / "feds200628_raw.csv"
PARAMS_PARQUET_PATH = DATA_DIR / "gsw_params.parquet"

# ---------------------------------------------------------------------------
# TIPS real-yield curve (GSW "feds200805", Gurkaynak-Sack-Wright 2010). Same
# Svensson parametric form as the nominal file (BETA0-3, TAU1-2), plus published
# fitted real zeros ``TIPSYNN`` and breakeven inflation ``BKEVENNN`` (percent).
# Used for the REAL term premium and the breakeven / inflation-risk-premium
# decomposition (see us/tips.py, us/inflation.py).
# ---------------------------------------------------------------------------
TIPS_URL = os.environ.get(
    "TIPS_URL",
    "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv",
)
# TIPS trade from 2y out (no short end), so the real grid starts at 2y. The IRP
# overlaps the nominal grid at 2/3/5/7/10y; 20y adds real long-end identification.
REAL_GRID = np.array([2.0, 3.0, 5.0, 7.0, 10.0, 20.0], dtype=float)
TIPS_PUBLISHED_YIELD_PREFIX = "TIPSY"      # fitted real zeros TIPSY02..TIPSY20
TIPS_BREAKEVEN_PREFIX = "BKEVEN"           # inflation compensation BKEVEN02..BKEVEN20

# The TIPS market is thin/illiquid 1999-2003 (unstable early fits — GSW caution
# against the early sample); the reliable real curve starts ~2004.
REAL_SAMPLE_START = "2004-01-01"

# NO effective lower bound on REAL rates: there is no "real cash" instrument, so
# real short rates go deeply negative (US real yields reached ~ -1.5% in 2012/2021).
# The real term premium is therefore a PLAIN-AFNS decomposition (shadow-rate off).
# We reuse the shadow pipeline with the bound pushed so far below the data it can
# never bind (Krippner option -> plain AFNS), i.e. estimate_elb=False, elb=ELB_REAL.
ELB_REAL = -1.0        # -100%: unreachable, guarantees the floor never binds

TIPS_RAW_CSV_PATH = DATA_DIR / "feds200805_raw.csv"
TIPS_PARQUET_PATH = DATA_DIR / "tips_params.parquet"
