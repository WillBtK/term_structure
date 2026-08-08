"""Reconstruct the zero-coupon yield grid from GSW Svensson parameters.

The Fed Board / GSW curve is stored parametrically (Svensson six-parameter
form); we reconstruct the yield at any maturity on demand rather than storing a
dense grid. This module implements the Svensson zero-coupon yield formula
standalone (numpy only) — a small, self-contained, unit-tested reference for the
published GSW convention, cross-checked against the Fed's own published SVENY
columns to < 1 bp.

Convention: BETA0-3 are in PERCENT and TAU1-2 in YEARS (as published). The
Svensson yield below therefore comes out in percent; ``reconstruct_grid``
converts to DECIMAL to match the engine's data contract.

Early-history regime (1961-~1980): GSW fit the four-parameter Nelson-Siegel form
(BETA0-2, TAU1 only; BETA3/TAU2 blank) before enough long bonds existed for the
second Svensson hump. ``reconstruct_grid`` reconstructs those rows too (the NS
form is Svensson with BETA3=0), so the panel reaches 1961. It additionally masks
each maturity to NaN on dates before that maturity is published in the file's own
``SVENYNN`` columns (e.g. the 10y curve only becomes reliable from 1971-08) — so
the early long end is correctly *ragged*, for a missing-data-aware filter to
consume, rather than silently extrapolated beyond the traded universe.
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


_NS_CORE = ["BETA0", "BETA1", "BETA2", "TAU1"]


def _availability_cap(params: pd.DataFrame) -> "pd.Series | None":
    """Per-date longest published maturity (years) from the file's SVENYNN columns.

    Returns a Series indexed like ``params`` giving, per date, the largest integer
    maturity ``M`` with ``SVENY{M}`` non-null (0 if none / no such columns present,
    in which case masking is skipped by the caller). Used to keep the early long
    end ragged rather than extrapolated beyond the traded universe.
    """
    pref = PUBLISHED_YIELD_PREFIX
    cols = [c for c in params.columns
            if c.startswith(pref) and c[len(pref):].isdigit()]
    if not cols:
        return None
    mats = np.array([int(c[len(pref):]) for c in cols], dtype=float)   # (P,)
    present = params[cols].notna().to_numpy()                          # (T, P)
    caps = (present * mats[np.newaxis, :]).max(axis=1)                 # longest available
    return pd.Series(caps, index=params.index)


def reconstruct_grid(
    params: pd.DataFrame,
    grid: np.ndarray = COMMON_GRID,
    ns_fallback: bool = True,
    mask_by_availability: bool = True,
) -> pd.DataFrame:
    """Reconstruct a date x maturity panel of zero yields (DECIMAL).

    Parameters
    ----------
    params
        DataFrame indexed by date with columns ``BETA0..BETA3, TAU1, TAU2``
        (percent / years), e.g. the output of ``fetch_gsw.load_gsw_params``. If
        the file's ``SVENYNN`` columns are also present they are used for
        availability masking (see ``mask_by_availability``).
    grid
        Maturities in years. Defaults to the common 8-point grid.
    ns_fallback
        When True (default) reconstruct early Nelson-Siegel-only rows (BETA3/TAU2
        blank, ~pre-1980) using the four-parameter NS form (Svensson with
        BETA3=0), so the panel reaches 1961. When False, keep the historical
        behaviour of dropping those rows (full Svensson only).
    mask_by_availability
        When True (default) and SVENYNN columns are present, set each maturity to
        NaN on dates before it is published in the file (e.g. 10y before 1971-08),
        yielding a correctly ragged early long end.

    Returns
    -------
    DataFrame indexed by date, columns = ``grid`` (float years), values =
    zero-coupon yields in DECIMAL (e.g. 0.0423) — the engine's data contract.
    Rows lacking the core NS parameters (or with TAU1<=0) are dropped.
    """
    missing = [c for c in PARAM_COLUMNS if c not in params.columns]
    if missing:
        raise ValueError(f"params is missing required columns: {missing}")

    # Core NS parameters must be present with a positive first decay.
    valid = params[_NS_CORE].notna().all(axis=1) & (params["TAU1"] > 0)
    # The second Svensson hump is usable only with a valid BETA3 and TAU2>0.
    has_sven = (params["BETA3"].notna() & params["TAU2"].notna()
                & (params["TAU2"] > 0))
    if not ns_fallback:
        valid &= has_sven
    p = params.loc[valid].copy()
    ns_only = ~has_sven.loc[p.index].to_numpy()

    # For NS-only rows, zero the second hump (BETA3=0) and give TAU2 any positive
    # placeholder — the term BETA3*(...) vanishes, so svensson_yield reduces to NS.
    beta3 = np.where(ns_only, 0.0, p["BETA3"].to_numpy(dtype=float))
    tau2 = np.where(ns_only, 1.0, p["TAU2"].to_numpy(dtype=float))

    grid = np.asarray(grid, dtype=float)
    n = grid[np.newaxis, :]
    yields_pct = svensson_yield(
        n,
        p["BETA0"].to_numpy()[:, np.newaxis],
        p["BETA1"].to_numpy()[:, np.newaxis],
        p["BETA2"].to_numpy()[:, np.newaxis],
        beta3[:, np.newaxis],
        p["TAU1"].to_numpy()[:, np.newaxis],
        tau2[:, np.newaxis],
    )
    out = pd.DataFrame(yields_pct / 100.0, index=p.index, columns=grid)

    if mask_by_availability:
        caps = _availability_cap(params)
        if caps is not None:
            caps = caps.loc[out.index].to_numpy()[:, np.newaxis]      # (T,1)
            out = out.mask(grid[np.newaxis, :] > caps)                # NaN beyond cap
    return out


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
        # Validate the formula itself: reconstruct at every date and compare only
        # where the file publishes a value (availability masking is irrelevant
        # here and would NaN-out exactly the pre-availability dates we skip).
        recon = reconstruct_grid(params, np.array([mat]),
                                 mask_by_availability=False) * 100.0  # to percent
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
