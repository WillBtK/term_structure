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
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from model import bias_correction as bcm
from model import ekf
from model import shadow_rate as sr
from model import term_premium as tp
from us.run_shadow import run_shadow

logger = logging.getLogger(__name__)


@dataclass
class ProductionParams:
    """A frozen, serializable parameter set for the fast daily filter-forward.

    Produced (slowly) by `run_production_tp` on the re-estimation cadence and
    reused (cheaply) by `filter_forward` every day. `K_bc` is the BRW
    bias-corrected P mean-reversion used for the expectations leg; `params` is the
    estimated shadow-AFNS state-space fit.
    """
    params: "ekf.ShadowParams"
    K_bc: np.ndarray          # (3,3) bias-corrected K^P
    elb: float
    n_per_year: int = 24

    def to_json(self, path) -> None:
        p = self.params
        d = {"K": np.asarray(p.K).tolist(), "theta": np.asarray(p.theta).tolist(),
             "sigma": np.asarray(p.sigma).tolist(), "lambda_": float(p.lambda_),
             "h": float(p.h), "elb": float(self.elb),
             "K_bc": np.asarray(self.K_bc).tolist(), "n_per_year": int(self.n_per_year)}
        from pathlib import Path
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path) -> "ProductionParams":
        d = json.loads(open(path).read())
        params = ekf.ShadowParams(K=np.array(d["K"]), theta=np.array(d["theta"]),
                                  sigma=np.array(d["sigma"]), lambda_=d["lambda_"],
                                  h=d["h"], elb=d["elb"])
        return cls(params=params, K_bc=np.array(d["K_bc"]), elb=d["elb"],
                   n_per_year=d["n_per_year"])


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


def filter_forward(prod: ProductionParams, panel: pd.DataFrame,
                   dt: float = 1.0 / 12) -> pd.DataFrame:
    """**Fast daily path**: decompose a yield panel with *cached* params — no
    estimation. This is the operation run every day to publish the term premium.

    Runs one Kalman/EKF pass (the shadow-AFNS filter) forward over `panel` with
    the fixed `prod` parameters and decomposes into the canonical TP frame. Cost is
    milliseconds (last-12-months window) to a couple of seconds (full history);
    `dt` should match the panel frequency (1/12 monthly, ~1/252 daily) — `K^P` is
    continuous-time so any frequency is correct.
    """
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    xs, fits = ekf.filter_states(prod.params, Y, grid, prod.elb, dt,
                                 n_per_year=prod.n_per_year)
    return _assemble_tp(xs, fits, prod.K_bc, prod.params.theta,
                        np.diag(prod.params.sigma), prod.elb, grid, panel.index,
                        n_per_year=prod.n_per_year)


def run_production_tp(start: str = "1987-01-01", end=None, maxiter: int = 5000,
                      n_boot: int = 2000, seed: int = 0,
                      warm_start=None, n_per_year: int = 24,
                      panel=None, elb: float | None = None,
                      estimate_elb: bool = False) -> dict:
    """Estimate the production term premium (modern-sample shadow AFNS + BRW).

    Country-agnostic: pass another country's `panel` + `elb` (and
    `estimate_elb=True` where the bound is well identified, e.g. Germany/Japan);
    the default is the US 1987+ GSW panel with the fixed US ELB.

    Returns both the bias-corrected (headline) and uncorrected canonical frames,
    the shadow short rate, factors, and the BRW diagnostics.
    """
    r = run_shadow(freq="monthly", start=start, end=end, maxiter=maxiter,
                   warm_start=warm_start, panel=panel, elb=elb,
                   estimate_elb=estimate_elb)
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
    prod = ProductionParams(params=r["params"], K_bc=bc.K_bc, elb=elb,
                            n_per_year=n_per_year)
    return {
        "ours": ours, "ours_uncorrected": ours_uncorr, "bc": bc, "prod": prod,
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
