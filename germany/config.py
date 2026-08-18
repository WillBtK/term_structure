"""Germany-specific configuration for the term-premium pipeline.

Everything country-specific about the Germany leg lives here: the data source
(Deutsche Bundesbank estimated term structure, Svensson method), the maturity
grid, the effective lower bound, and cache paths. The engine in ``/model`` is
country-agnostic — ``/germany`` produces the same data contract ``/us`` does
(a date x maturity panel of zero yields in decimal), via ``fetch_bundesbank``
and the shared Svensson reconstruction in ``us.reconstruct``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Common maturity grid (years), shared across all four countries: 3m..10y.
COMMON_GRID = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)

# ---------------------------------------------------------------------------
# Effective lower bound (decimal). Germany/euro area: -1.0% central — an ECB
# reversal-rate estimate (per the charter; provenance: discussions with Peter
# Praet). Bund yields went clearly negative (2014-2022), so unlike the US the
# ELB is *well identified* by ML here — we estimate it AND report the -1.0%
# calibration + a sensitivity band.
# ---------------------------------------------------------------------------
ELB = -0.010
ELB_BAND = (-0.015, -0.005)

# ---------------------------------------------------------------------------
# Sample windows. Full monthly history starts 1972-09 (Deutsche-Mark era). The
# *modern production* estimate uses the euro era only (ECB regime, from 1999),
# so 1970s-80s Bundesbank dynamics do not contaminate today's estimate — the
# German analogue of the US 1987+ choice.
# ---------------------------------------------------------------------------
HIST_START = "1972-09-01"     # full history (historical / stagflation exhibit)
PROD_START = "1999-01-01"     # euro era (modern production TP)

# ---------------------------------------------------------------------------
# Data source: Deutsche Bundesbank, "Term structure of interest rates on listed
# Federal securities (method by Svensson)". SDMX REST API, flow BBSIS; German
# general government (sector S1311, instrument A604 = listed Federal securities).
# Monthly (M) from 1972-09; daily (D) from 1997-08. BETA0-3 in PERCENT, TAU1-2
# in YEARS — same convention as GSW, so us.reconstruct reuses unchanged.
# ---------------------------------------------------------------------------
BUNDESBANK_API = os.environ.get(
    "BUNDESBANK_API", "https://api.statistiken.bundesbank.de/rest/data/BBSIS")

# key template: {FREQ}.I.ZST.{CODE}.EUR.S1311.B.A604.{POS}.R.A.A._Z._Z.A
_SECTOR = "EUR.S1311.B.A604"
_TAIL = "R.A.A._Z._Z.A"

# Svensson parameter series (canonical name -> Bundesbank position code).
PARAM_CODES = {"BETA0": "B0", "BETA1": "B1", "BETA2": "B2", "BETA3": "B3",
               "TAU1": "T1", "TAU2": "T2"}

# Published estimated spot yields (percent) at fixed maturities — used only as a
# reconstruction cross-check. Maturity (years) -> Bundesbank position code.
SPOT_CODES = {0.5: "R005X", 1.0: "R01XX", 2.0: "R02XX", 3.0: "R03XX",
              4.0: "R04XX", 5.0: "R05XX", 6.0: "R06XX", 7.0: "R07XX",
              8.0: "R08XX", 9.0: "R09XX", 10.0: "R10XX"}

PARAM_COLUMNS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def param_series_key(name: str, freq: str = "M") -> str:
    """Full SDMX series key for a Svensson parameter (``name`` in PARAM_CODES)."""
    return f"{freq}.I.ZST.{PARAM_CODES[name]}.{_SECTOR}._Z.{_TAIL}"


def spot_series_key(maturity: float, freq: str = "M") -> str:
    """Full SDMX series key for a published spot yield at ``maturity`` years."""
    return f"{freq}.I.ZST.ZI.{_SECTOR}.{SPOT_CODES[maturity]}.{_TAIL}"


# Cache paths (contents gitignored; directory tracked).
DATA_DIR = Path(__file__).resolve().parent / "data"
PARAMS_PARQUET_PATH = DATA_DIR / "bundesbank_params.parquet"
RAW_DIR = DATA_DIR / "raw"
