"""Production US term premium: 1987+ shadow-rate AFNS with Bauer-Rudebusch-Wu
bias-correction.

This is the **headline, cross-country-comparable** US term-premium series. It is
estimated on a **modern sample (1987+)** on purpose: a full 1961-present sample
would pull the physical long-run mean θ and persistence K^P toward the 1970s-80s
high-inflation world and so contaminate *today's* expected-rate path (and hence
today's term premium). The modern sample keeps those old-regime dynamics out;
**BRW bias-correction** then fixes the finite-sample persistence bias a short
sample would otherwise carry (in-sample it lifted the short-end SF-Fed term-premium
correlation from ~0.14 to ~0.83).

The 1970s / stagflation term premium is a *separate, clearly-labeled historical
exhibit* (see `validation/FINDINGS_phase3.md`, `FINDINGS_kim_validation.md`), not
this series. The two are deliberately not joined (the earlier 2008 splice is kept
in `us/splice.py` as an internal option, not the product).

Pipeline:
    run_shadow(1987+)                         # shadow-rate AFNS, ELB-respecting
      -> bias_correction.bias_correct(states) # BRW-correct the P-persistence
      -> expected_floored_short_rate_yields    # expected floored short rate, K^P_bc
      -> term premium = fitted (ELB) yield - expected
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from model import bias_correction as bcm
from model import shadow_rate as sr
from model import term_premium as tp
from us.run_shadow import run_shadow

logger = logging.getLogger(__name__)


def _assemble_tp(states, fitted, K, theta, Sigma, elb, grid, dates,
                 n_per_year: int = 24) -> pd.DataFrame:
    """Canonical tidy term-premium frame from a shadow fit + a P-dynamics K.

    ``fitted`` are the shadow (ELB) fitted yields (T,m); the expectations leg is
    the expected-average *floored* short rate under P with mean-reversion ``K``
    (pass the BRW-corrected K^P for the headline, or the raw K^P for the
    uncorrected comparison). Returns columns
    ``(date, maturity, term_premium, expected_rate, fitted_yield)`` in decimal.
    """
    states = np.asarray(states, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    grid = np.asarray(grid, dtype=float)
    expected = tp.expected_floored_short_rate_yields(
        states, K, theta, Sigma, elb, sr.RHO, grid, n_per_year=n_per_year)
    premium = fitted - expected
    rows = []
    for j, mat in enumerate(grid):
        rows.append(pd.DataFrame({
            "date": dates, "maturity": float(mat),
            "term_premium": premium[:, j], "expected_rate": expected[:, j],
            "fitted_yield": fitted[:, j],
        }))
    return pd.concat(rows, ignore_index=True)


def run_production_tp(start: str = "1987-01-01", end=None, maxiter: int = 5000,
                      n_boot: int = 2000, seed: int = 0,
                      warm_start=None, n_per_year: int = 24) -> dict:
    """Estimate the production US term premium (1987+ shadow AFNS + BRW).

    Returns both the bias-corrected (headline) and uncorrected canonical frames,
    the shadow short rate, factors, and the BRW diagnostics.
    """
    r = run_shadow(freq="monthly", start=start, end=end, maxiter=maxiter,
                   warm_start=warm_start)
    states, fitted = r["states"], r["fitted"].to_numpy(dtype=float)
    K, theta = r["params"].K, r["params"].theta
    Sigma = np.diag(r["params"].sigma)
    elb, grid, dates = r["elb"], r["grid"], r["dates"]
    dt = 1.0 / 12

    bc = bcm.bias_correct(states, dt, n_boot=n_boot, seed=seed)
    logger.info("BRW: spectral radius %.4f -> %.4f (delta=%.2f)",
                bc.rho_hat, bc.rho_bc, bc.delta)

    ours = _assemble_tp(states, fitted, bc.K_bc, theta, Sigma, elb, grid, dates,
                        n_per_year=n_per_year)                 # headline
    ours_uncorr = _assemble_tp(states, fitted, K, theta, Sigma, elb, grid, dates,
                               n_per_year=n_per_year)          # comparison
    return {
        "ours": ours, "ours_uncorrected": ours_uncorr, "bc": bc,
        "shadow": r["shadow"], "factors": r["factors"], "params": r["params"],
        "elb": elb, "grid": grid, "dates": dates, "loglik": r["loglik"],
        "success": r["success"], "yield_fit_mad_bp": r["yield_fit_mad_bp"],
    }


def validate_vs_sf_fed(frame: pd.DataFrame, make_plots: bool = False) -> dict:
    """Compare a canonical TP frame to the cached SF Fed benchmark."""
    from validation.compare_afns import run_comparison
    from validation.sf_fed import load_sf_fed_series
    return run_comparison(frame, load_sf_fed_series(), make_plots=make_plots)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Production US term premium (1987+ shadow + BRW)")
    ap.add_argument("--start", default="1987-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--maxiter", type=int, default=5000)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=None, help="parquet path for the headline frame")
    args = ap.parse_args()
    r = run_production_tp(start=args.start, end=args.end, maxiter=args.maxiter)
    print(f"loglik={r['loglik']:.1f} success={r['success']} "
          f"yield_fit_mad={r['yield_fit_mad_bp']:.2f}bp "
          f"rho {r['bc'].rho_hat:.4f}->{r['bc'].rho_bc:.4f}")
    if args.validate:
        for name, fr in [("bias-corrected", r["ours"]), ("uncorrected", r["ours_uncorrected"])]:
            g = validate_vs_sf_fed(fr)["metrics"]
            g = g[g["quantity"] == "term_premium"]
            print(f"  {name}:")
            for row in g.itertuples():
                print(f"    {row.maturity:g}y corr={row.correlation:.3f} bias={row.mean_bias_bp:+.1f}bp")
    if args.out:
        r["ours"].to_parquet(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
