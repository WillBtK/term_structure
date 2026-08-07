"""Python -> Julia bridge for term-premium estimation.

The estimation engine is ``TermStructureModels.jl`` (MIT), driven via a thin
file-based subprocess boundary: we write the yield panel to a CSV, run
``model/engine_julia/estimate_tp.jl``, and read back the canonical term-premium
frame. This keeps the data layer (``/us``) and validation (``/validation``) in
Python while using the better-tracking Julia estimator (see
``validation/FINDINGS_phase1e.md``).

Environment:
- ``JULIA_BIN``       : path to the julia binary (default: ``~/.juliaup/bin/julia``
                        or ``julia`` on PATH).
- ``JULIA_DEPOT_PATH``: Julia depot with the packages installed (juliaup default
                        ``~/.julia`` if unset).
See ``model/README.md`` for install + the ``*.julialang.org`` / ``*.julialang.net``
egress requirement.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
JULIA_PROJECT = REPO_ROOT / "model" / "engine_julia"
JULIA_DRIVER = JULIA_PROJECT / "estimate_tp.jl"

_CANONICAL = ["date", "maturity", "term_premium", "expected_rate", "fitted_yield"]


def _julia_bin() -> str:
    for cand in (os.environ.get("JULIA_BIN"), str(Path.home() / ".juliaup/bin/julia"),
                 shutil.which("julia")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError(
        "Could not find the julia binary. Install Julia (see model/README.md) or "
        "set JULIA_BIN."
    )


def estimate_term_premium(
    panel: pd.DataFrame,
    maturities_years=(1.0, 2.0, 3.0, 5.0, 7.0, 10.0),
    iteration: int = 6000,
    burnin: int = 1000,
    upper_p: int = 24,
    timeout: int = 3600,
) -> pd.DataFrame:
    """Estimate the AFNS term premium via the Julia engine.

    Parameters
    ----------
    panel : DataFrame indexed by date, columns = maturities in YEARS, values =
        zero-coupon yields in DECIMAL (the ``/us`` panel from build_yield_panel).
    maturities_years : maturities (years) to decompose (must exist in the model's
        pricing horizon; SF Fed overlap is 1..10y).

    Returns the canonical long frame ``date,maturity,term_premium,expected_rate,
    fitted_yield`` in DECIMAL.
    """
    if not JULIA_DRIVER.exists():
        raise RuntimeError(f"Julia driver not found at {JULIA_DRIVER}")

    with tempfile.TemporaryDirectory() as td:
        panel_csv = Path(td) / "panel.csv"
        out_csv = Path(td) / "tp.csv"
        # Julia driver expects PERCENT; our panel is decimal.
        (panel * 100.0).rename_axis("date").to_csv(panel_csv)

        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        cmd = [
            _julia_bin(),
            f"--project={JULIA_PROJECT}",
            str(JULIA_DRIVER),
            str(panel_csv),
            str(out_csv),
            str(iteration),
            str(burnin),
            ",".join(str(m) for m in maturities_years),
            str(upper_p),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out_csv.exists():
            raise RuntimeError(
                f"Julia engine failed (rc={proc.returncode}).\n"
                f"stdout tail:\n{proc.stdout[-1500:]}\n"
                f"stderr tail:\n{proc.stderr[-1500:]}"
            )
        out = pd.read_csv(out_csv, parse_dates=["date"])
    return out[_CANONICAL]


def run(freq: str = "monthly", start=None, end=None, **kwargs) -> pd.DataFrame:
    """Build the US GSW panel and estimate the term premium via the Julia engine.

    Returns the canonical frame ready for validation.compare_afns.run_comparison.
    """
    from us.config import COMMON_GRID
    from us.fetch_gsw import build_yield_panel

    panel = build_yield_panel(grid=COMMON_GRID)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    if freq == "monthly":
        panel = panel.groupby([panel.index.year, panel.index.month]).tail(1)
    elif freq != "daily":
        raise ValueError(f"unknown freq {freq!r}")
    return estimate_term_premium(panel.dropna(), **kwargs)
