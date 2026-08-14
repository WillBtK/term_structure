"""Driver: estimate the Markov-switching AFNS on the long US sample (1961+) and
extract the regime-conditional term premium — the stagflation question.

Pipeline:
    build_yield_panel() (1961+, ragged early 10y)
      -> single-regime plain-AFNS fit (warm start for the regime model)
      -> regime_switching.estimate (S regimes, Kim filter QML)
      -> Kim smoother: which dates are the high-inflation regime
      -> regime_switching.regime_term_premium: TP level per regime

The high-inflation regime's term-premium level answers "how high can the term
premium get in a stagflationary environment."
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from model import ekf
from model import regime_switching as rs
from us.config import COMMON_GRID
from us.fetch_gsw import build_yield_panel

logger = logging.getLogger(__name__)


def _panel(start, end, panel=None):
    if panel is None:
        panel = build_yield_panel(grid=COMMON_GRID)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    panel = panel.groupby([panel.index.year, panel.index.month]).tail(1)
    return panel.dropna(how="all")   # keep ragged rows


def _warm_start(Y, grid, dt, S, maxiter=3000):
    """Single-regime plain-AFNS fit -> a 2-regime start (persistent/high-vol vs calm)."""
    from us.run_afns_ekf import PLAIN_ELB
    p1, _, _ = ekf.estimate(Y, grid, PLAIN_ELB, dt, maxiter=maxiter, n_per_year=24,
                            diagonal_K=True)
    kap = np.diag(p1.K); sig = p1.sigma
    # regime 0 = persistent & high-vol; regime 1 = calm (faster reversion, low vol).
    kappa = np.vstack([0.4 * kap, 1.8 * kap])[:S]
    sigma = np.vstack([1.6 * sig, 0.8 * sig])[:S]
    theta = np.tile(p1.theta, (S, 1))
    return rs.RSParams(kappa=kappa, theta=theta, sigma=sigma,
                       lambda_=p1.lambda_, h=p1.h, p_stay=np.full(S, 0.975))


def run_regimes(start="1961-06-01", end=None, S: int = 2, maxiter: int = 6000,
                warm_maxiter: int = 3000, restarts: int = 1, panel=None,
                mr_halflife: float = 3.5):
    panel = _panel(start, end, panel=panel)
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    dt = 1.0 / 12
    logger.info("regime-switching AFNS: %d obs (%s..%s), S=%d",
                len(panel), panel.index.min().date(), panel.index.max().date(), S)

    warm = _warm_start(Y, grid, dt, S, maxiter=warm_maxiter)
    logger.info("warm start ready (lambda=%.3f)", warm.lambda_)
    params, res, ll = rs.estimate(Y, grid, dt, S=S, warm_start=warm,
                                  maxiter=maxiter, n_per_year=24, restarts=restarts)
    logger.info("done: loglik=%.1f success=%s", ll, res.success)

    filt = rs.kim_filter(params, Y, grid, dt, smooth=True)
    # Regime detection is persistence-agnostic; the regime-conditional TP uses the
    # same mean-reversion floor as the production series so the stagflation-regime
    # premium is anchored (not the spurious negative a near-unit-root K^P gives).
    dec = rs.regime_term_premium(params, filt["states"], grid, mr_halflife=mr_halflife)

    idx = panel.index
    # Ex-ante (data-weighted) term premium: mix regime-conditional TP by smoothed prob.
    sm = filt["smoothed_prob"]                          # (T, S)
    tp_mixed = np.einsum("ts,tsm->tm", sm, dec["term_premium"])  # (T, m)
    return {
        "params": params, "loglik": ll, "success": bool(res.success),
        "panel": panel, "grid": grid, "dates": idx,
        "smoothed_prob": sm, "filtered_prob": filt["filtered_prob"],
        "states": filt["states"], "decomp": dec, "tp_mixed": tp_mixed,
    }


def summarize(result: dict) -> dict:
    """Identify the high-inflation regime and report its term-premium level."""
    p = result["params"]; grid = result["grid"]; idx = result["dates"]
    sm = result["smoothed_prob"]
    S = p.n_regimes
    # High-inflation regime = highest long-run level mean theta_L (or the most
    # persistent). Use theta_L.
    hi = int(np.argmax(p.theta[:, 0]))
    j10 = int(np.argmin(np.abs(grid - 10.0)))
    tp = result["decomp"]["term_premium"]              # (T,S,m)
    reg = sm.argmax(axis=1)
    is_hi = reg == hi
    # 1970s window
    mask70s = (idx >= pd.Timestamp("1970-01-01")) & (idx <= pd.Timestamp("1985-12-31"))
    return {
        "n_regimes": S,
        "hi_regime": hi,
        "kappa": np.round(p.kappa, 4).tolist(),
        "theta_pct": np.round(p.theta * 100, 3).tolist(),
        "sigma_bp": np.round(p.sigma * 1e4, 1).tolist(),
        "p_stay": np.round(p.p_stay, 4).tolist(),
        "hi_regime_months": int(is_hi.sum()),
        "hi_regime_share": round(float(is_hi.mean()), 3),
        "tp10_hi_regime_mean_pct": round(float(np.nanmean(tp[:, hi, j10]) * 100), 3),
        "tp10_lo_regime_mean_pct": round(float(np.nanmean(tp[:, 1 - hi, j10]) * 100), 3)
        if S == 2 else None,
        "tp10_mixed_1970s_mean_pct": round(float(np.nanmean(result["tp_mixed"][mask70s, j10]) * 100), 3),
        "tp10_mixed_post1990_mean_pct": round(
            float(np.nanmean(result["tp_mixed"][idx >= pd.Timestamp("1990-01-01"), j10]) * 100), 3),
        "hi_regime_frac_in_1970s": round(float(is_hi[mask70s].mean()), 3),
    }


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Estimate regime-switching AFNS (US, 1961+)")
    ap.add_argument("--start", default="1961-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--regimes", type=int, default=2)
    ap.add_argument("--maxiter", type=int, default=6000)
    args = ap.parse_args()
    r = run_regimes(start=args.start, end=args.end, S=args.regimes, maxiter=args.maxiter)
    import json
    print(json.dumps(summarize(r), indent=2, default=str))


if __name__ == "__main__":
    _main()
