"""Reconstruct the zero-coupon yield grid from GSW Svensson parameters.

The Fed Board / GSW curve is stored parametrically (Svensson six-parameter
form); we reconstruct the yield at any maturity on demand rather than storing a
dense grid. This module implements the Svensson zero-coupon yield formula
*standalone* on purpose:

  * The vendored engine's numeric ``arbfree_dyn_ns.nss.ns_yield`` has a bug in
    its Svensson (beta3) term — it divides by ``lambda_ * T`` instead of
    ``lambda_svensson * T`` (see model/README.md). We do not want that in the
    data path.
  * Keeping this dependency-light (numpy only) makes it trivially testable
    without the heavy numba/sympy/IPython stack the engine needs.

Convention: BETA0-3 are in PERCENT and TAU1-2 in YEARS (as published). The
Svensson yield below therefore comes out in percent; ``reconstruct_grid``
converts to DECIMAL to match the engine's data contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMMON_GRID, PARAM_COLUMNS, PUBLISHED_YIELD_PREFIX


def _loading1(x: np.ndarray) -> np.ndarray:
    """(1 - e^-x) / x, with the well-defined limit -> 1 as x -> 0."""
    x = np.asarray(x, dtype=float)
    out = np.where(x == 0.0, 1.0, (1.0 - np.exp(-x)) / np.where(x == 0.0, 1.0, x))
    return out


def svensson_yield(
    n,
    beta0,
    beta1,
    beta2,
    beta3,
    tau1,
    tau2,
):
    """Svensson continuously-compounded zero-coupon yield (in PERCENT).

    y(n) = b0
         + b1 * L1(n/t1)
         + b2 * (L1(n/t1) - e^{-n/t1})
         + b3 * (L1(n/t2) - e^{-n/t2})

    where L1(x) = (1 - e^{-x}) / x. All arguments broadcast against each other,
    so ``n`` may be a scalar or array and the params may be scalars or arrays
    (e.g. one row per date).
    """
    n = np.asarray(n, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)
    beta1 = np.asarray(beta1, dtype=float)
    beta2 = np.asarray(beta2, dtype=float)
    beta3 = np.asarray(beta3, dtype=float)
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)

    x1 = n / tau1
    x2 = n / tau2
    l1 = _loading1(x1)
    slope = l1 - np.exp(-x1)               # NS curvature loading on tau1
    curv2 = _loading1(x2) - np.exp(-x2)    # Svensson second curvature on tau2
    return beta0 + beta1 * l1 + beta2 * slope + beta3 * curv2


def reconstruct_grid(
    params: pd.DataFrame,
    grid: np.ndarray = COMMON_GRID,
) -> pd.DataFrame:
    """Reconstruct a date x maturity panel of zero yields (DECIMAL).

    Parameters
    ----------
    params
        DataFrame indexed by date with columns ``BETA0..BETA3, TAU1, TAU2``
        (percent / years), e.g. the output of ``fetch_gsw.load_gsw_params``.
    grid
        Maturities in years. Defaults to the common 8-point grid.

    Returns
    -------
    DataFrame indexed by date, columns = ``grid`` (float years), values =
    zero-coupon yields in DECIMAL (e.g. 0.0423) — the engine's data contract.
    Rows with any missing/invalid parameter are dropped.
    """
    missing = [c for c in PARAM_COLUMNS if c not in params.columns]
    if missing:
        raise ValueError(f"params is missing required columns: {missing}")

    # Drop dates with missing params or non-positive taus (formula undefined).
    valid = params[PARAM_COLUMNS].notna().all(axis=1)
    valid &= (params["TAU1"] > 0) & (params["TAU2"] > 0)
    p = params.loc[valid, PARAM_COLUMNS]

    grid = np.asarray(grid, dtype=float)
    # Broadcast: dates on axis 0, maturities on axis 1.
    n = grid[np.newaxis, :]
    yields_pct = svensson_yield(
        n,
        p["BETA0"].to_numpy()[:, np.newaxis],
        p["BETA1"].to_numpy()[:, np.newaxis],
        p["BETA2"].to_numpy()[:, np.newaxis],
        p["BETA3"].to_numpy()[:, np.newaxis],
        p["TAU1"].to_numpy()[:, np.newaxis],
        p["TAU2"].to_numpy()[:, np.newaxis],
    )
    return pd.DataFrame(yields_pct / 100.0, index=p.index, columns=grid)


def cross_check_against_published(
    params: pd.DataFrame,
    tol_bp: float = 2.0,
) -> pd.DataFrame:
    """Sanity-check reconstruction against the file's own published SVENYNN.

    The feds200628 file publishes fitted zero yields ``SVENY01..SVENY30``
    (percent) alongside the parameters. When those columns are present we
    reconstruct at the same integer maturities and report the max absolute
    difference (in basis points) per maturity. A correct implementation should
    agree to well within a basis point.

    Returns a DataFrame with columns ``[maturity, max_abs_diff_bp, within_tol]``;
    empty if no SVENY columns are present.
    """
    sveny_cols = [
        c
        for c in params.columns
        if c.startswith(PUBLISHED_YIELD_PREFIX)
        and c[len(PUBLISHED_YIELD_PREFIX):].isdigit()
    ]
    if not sveny_cols:
        return pd.DataFrame(columns=["maturity", "max_abs_diff_bp", "within_tol"])

    rows = []
    for col in sorted(sveny_cols, key=lambda c: int(c[len(PUBLISHED_YIELD_PREFIX):])):
        mat = float(int(col[len(PUBLISHED_YIELD_PREFIX):]))
        recon = reconstruct_grid(params, np.array([mat])) * 100.0  # back to percent
        published = params.loc[recon.index, col]
        diff_bp = (recon.iloc[:, 0] - published).abs() * 100.0  # percent -> bp
        max_bp = float(diff_bp.max()) if len(diff_bp) else float("nan")
        rows.append(
            {
                "maturity": mat,
                "max_abs_diff_bp": max_bp,
                "within_tol": bool(max_bp <= tol_bp),
            }
        )
    return pd.DataFrame(rows)
