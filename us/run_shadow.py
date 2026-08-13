"""Driver: estimate the US shadow-rate AFNS (Christensen-Rudebusch) and extract
the shadow short rate.

Pipeline:
    build_yield_panel()                 # US GSW zero yields (decimal), 8-pt grid
      -> ekf.estimate (shadow-rate AFNS, ELB from us.config.ELB)
      -> ekf.filter_states             # filtered latent factors
      -> shadow short rate s=L+S, fitted (ELB) yields, and the ELB "wedge"
         (fitted ELB yield minus the unbounded/plain-AFNS shadow yield).

Away from the bound the wedge -> 0 (collapse to plain AFNS); during the ZLB it is
large — the internal check the README calls for. Estimation is monthly by
default (the nonlinear EKF is heavier than the linear fit).
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from model import ekf
from model import shadow_rate as sr
from model import term_premium as tp
from us.config import COMMON_GRID, ELB
from us.fetch_gsw import build_yield_panel

logger = logging.getLogger(__name__)


def _panel(freq, start, end, panel=None):
    # `panel` (a prebuilt date x maturity frame) lets any country reuse this
    # driver; when None we build the US GSW panel (backward-compatible default).
    if panel is None:
        panel = build_yield_panel(grid=COMMON_GRID)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    if freq == "monthly":
        panel = panel.groupby([panel.index.year, panel.index.month]).tail(1)
    elif freq != "daily":
        raise ValueError(f"unknown freq {freq!r}")
    # Keep ragged rows (the EKF masks missing maturities per date); drop only
    # fully-empty dates.
    return panel.dropna(how="all")


def run_shadow(freq: str = "monthly", start="1987-01-01", end=None,
               elb: float | None = None, maxiter: int = 4000,
               warm_start: "ekf.ShadowParams | None" = None,
               diagonal_K: bool = True, estimate_elb: bool = False,
               panel=None):
    """Estimate the shadow-rate AFNS and return a results dict.

    `warm_start` (e.g. a prior fit) initializes the optimizer. `diagonal_K=True`
    is the independent-factor production spec. `estimate_elb=True` estimates the
    ELB by ML (Christensen-Rudebusch) instead of fixing it; `elb` is then the
    starting value. `panel` (a prebuilt zero-yield frame) makes this driver
    country-agnostic — pass another country's panel + its `elb`; default is US.
    """
    elb = ELB if elb is None else elb
    panel = _panel(freq, start, end, panel=panel)
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    dt = 7.0 / 365.2425 if freq == "daily" else 1.0 / 12
    logger.info("estimating shadow AFNS on %d obs (%s..%s) ELB=%.4f%s",
                len(panel), panel.index.min().date(), panel.index.max().date(), elb,
                " (ELB estimated)" if estimate_elb else "")

    npy = 24
    params, res, ll = ekf.estimate(Y, grid, elb, dt, warm_start=warm_start,
                                   maxiter=maxiter, n_per_year=npy,
                                   diagonal_K=diagonal_K, estimate_elb=estimate_elb)
    if estimate_elb:
        elb = float(params.elb)
    logger.info("estimation done: loglik=%.1f success=%s lambda=%.4f ELB=%.4f eig(K^P)=%s",
                ll, res.success, params.lambda_, elb,
                np.round(np.sort(np.linalg.eigvals(params.K).real), 4))

    xs, fits = ekf.filter_states(params, Y, grid, elb, dt, n_per_year=npy)
    shadow = xs[:, 0] + xs[:, 1]                    # s = L + S

    # ELB wedge: fitted ELB yields minus the same-state unbounded (shadow) yields.
    pre_free = sr.precompute(params.lambda_, np.diag(params.sigma), -1e6, grid, n_per_year=npy)
    free = np.array([sr.elb_yields(xs[t], pre_free) for t in range(len(xs))])
    wedge = fits - free                            # (T, m), ~0 away from bound

    # Shadow-model term premium: fitted (ELB) yield minus the expected-average
    # *floored* short rate under P (collapses to the plain-AFNS TP away from bound).
    expected = tp.expected_floored_short_rate_yields(
        xs, params.K, params.theta, np.diag(params.sigma), elb, sr.RHO, grid,
        n_per_year=npy)
    premium = fits - expected                       # (T, m)

    idx = panel.index
    shadow_df = pd.DataFrame({"date": idx, "shadow_short_rate": shadow})
    factors_df = pd.DataFrame(xs, index=idx, columns=["L", "S", "C"])
    fitted_df = pd.DataFrame(fits, index=idx, columns=grid)
    wedge_df = pd.DataFrame(wedge, index=idx, columns=grid)
    expected_df = pd.DataFrame(expected, index=idx, columns=grid)
    premium_df = pd.DataFrame(premium, index=idx, columns=grid)
    return {
        "params": params, "loglik": ll, "success": bool(res.success),
        "elb": elb, "grid": grid, "dates": idx,
        "shadow": shadow_df, "factors": factors_df, "fitted": fitted_df,
        "wedge": wedge_df, "expected_rate": expected_df, "term_premium": premium_df,
        "states": xs,
        "yield_fit_mad_bp": float(np.nanmean(np.abs(fits - Y)) * 1e4),
    }


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Estimate US shadow-rate AFNS")
    p.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    p.add_argument("--start", default="1987-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--elb", type=float, default=None, help="override ELB (decimal)")
    p.add_argument("--out", default=None, help="optional parquet prefix for outputs")
    args = p.parse_args()

    r = run_shadow(freq=args.freq, start=args.start, end=args.end, elb=args.elb)
    s = r["shadow"]
    print(f"loglik={r['loglik']:.1f} success={r['success']} "
          f"yield_fit_mad={r['yield_fit_mad_bp']:.2f}bp ELB={r['elb']}")
    print(f"shadow short rate: min={s.shadow_short_rate.min():.4f} "
          f"max={s.shadow_short_rate.max():.4f}")
    # year-end snapshot without relying on a specific resample alias
    ye = s.assign(year=s.date.dt.year).groupby("year").tail(1)
    print(ye.tail(20).round(4).to_string(index=False))
    if args.out:
        r["shadow"].to_parquet(f"{args.out}_shadow.parquet")
        r["fitted"].to_parquet(f"{args.out}_fitted.parquet")
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
