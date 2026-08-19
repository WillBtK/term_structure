"""Shared cross-sectional fit behind every Hu-Pan-Wang "noise" gauge.

All three gauges — US nominal Treasuries, US TIPS, German Bunds — measure the same
thing: how far individual securities sit from the smooth curve their market implies. So
the fit belongs in one place, and the corrections below apply to all three.

Each was originally a 3-factor Nelson-Siegel OLS at ONE pinned decay, with the plain
RMS residual. Validating the US gauge against synthetic cross-sections priced exactly
off a known curve — where the true noise is zero, so anything reported is the method's
own error — showed that specification reporting **2-9 bp**, i.e. inventing illiquidity.
Three separable causes, each fixed here:

1. **Functional-form misfit.** A 3-factor Nelson-Siegel at a fixed decay cannot
   reproduce a 4-factor Svensson curve, and the gap moves with the shape of the curve —
   so the error is not even a constant offset one could subtract. Fixed by fitting the
   full Svensson form and PROFILING over a small grid of decay pairs: linear (ordinary
   or weighted least squares) given the decays, so it stays fast and deterministic,
   while the floor drops to 0.1-0.7 bp.

2. **The coupon effect.** Yield to maturity is not a function of maturity alone: two
   bonds maturing the same day with different coupons trade at different YTMs. Any
   smooth curve through a mixed-coupon cross-section is therefore mis-specified, and on
   the synthetic panel that alone accounted for 2.3 bp. Absorbed by carrying the coupon
   as a regressor, which leaves ~0.6 bp.

3. **Downward small-sample bias.** Plain RMS divides by n, not by the residual degrees
   of freedom, so it is biased low by sqrt((n-p)/n) — about 11% for five parameters on
   two dozen bonds, and worse on thin days. That made days with fewer securities look
   spuriously calm, which is backwards: those are usually the stressed ones. Fixed by
   dividing by the (effective) degrees of freedom.

The one construction immune to all of this is referencing a bond against ITSELF priced
off a published curve (`validation/ust_liquidity.gsw_model_yields`), which fits nothing
and cancels the coupon exactly. Where a central bank publishes daily curve parameters
that is the better measure; this module is what to use when it does not, or when a
curve-free number comparable across markets is wanted.
"""
from __future__ import annotations

import numpy as np

# Decay pairs to profile over. Wide enough to span the shapes a government curve takes
# (a short-end hump through a long-end slope) without being so fine that the fit starts
# absorbing the very scatter it is meant to measure.
LAMBDA1_GRID = (0.25, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6)
LAMBDA2_GRID = (0.04, 0.06, 0.09, 0.13, 0.19, 0.28, 0.40)

_N_BASE_PARAMS = 4          # beta0..beta3; the decays are profiled, not fitted


def svensson_design(tau: np.ndarray, lam1: float, lam2: float) -> np.ndarray:
    """Svensson regressors ``[1, slope, curvature1, curvature2]`` at fixed decays, so
    the fit given the decays is (weighted) least squares."""
    tau = np.asarray(tau, dtype=float)
    cols = [np.ones_like(tau)]
    for i, lam in enumerate((lam1, lam2)):
        lt = np.maximum(lam * tau, 1e-9)
        s = (1.0 - np.exp(-lt)) / lt
        if i == 0:
            cols.append(s)
        cols.append(s - np.exp(-lt))
    return np.column_stack(cols)


def residual_se(tau: np.ndarray, y: np.ndarray,
                coupon: np.ndarray | None = None,
                weights: np.ndarray | None = None,
                min_dof: float = 3.0) -> float:
    """Residual standard error of the best fixed-decay Svensson fit — the 'noise'.

    ``tau`` residual maturities in years, ``y`` yields in decimal. ``coupon`` (any
    consistent unit) adds the coupon regressor when it varies; pass None where the
    cross-section has no coupon dimension. ``weights`` gives weighted least squares —
    used by the TIPS gauge to taper near-maturity issues so basket changes do not step
    the series — and the SAME weights scale the residual, so a down-weighted outlier
    distorts neither the fit nor the level. Degrees of freedom use the EFFECTIVE
    (weighted) count. Returns NaN when there is too little to fit.
    """
    tau = np.asarray(tau, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(tau) if weights is None else np.asarray(weights, dtype=float)

    add_coupon = coupon is not None and np.ptp(np.asarray(coupon, dtype=float)) > 1e-12
    n_params = _N_BASE_PARAMS + int(add_coupon)
    dof = float(w.sum()) - n_params
    if dof < min_dof or tau.size <= n_params:
        return np.nan

    sw = np.sqrt(w)
    best = np.inf
    for lam1 in LAMBDA1_GRID:
        for lam2 in LAMBDA2_GRID:
            if lam2 >= lam1:
                continue
            X = svensson_design(tau, lam1, lam2)
            if add_coupon:
                X = np.column_stack([X, np.asarray(coupon, dtype=float)])
            beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            ssr = float(np.sum(w * (y - X @ beta) ** 2))
            best = min(best, ssr)
    return float(np.sqrt(best / dof))
