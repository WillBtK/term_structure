"""Driver: estimate a **plain AFNS** term premium via our own Python state-space
engine (`model.ekf`) and decompose it into fitted yield / expected-rate / term
premium.

This is the Phase-3 "workhorse guard" (plan Strand 0). The Julia engine
(`us.engine`) is Bayesian and returns only the term-premium series — it never
hands back the physical dynamics `K^P` we need for bootstrap bias-correction
(Bauer-Rudebusch-Wu) and it cannot be pushed to the ragged 1961+ sample or a
regime-switching likelihood. Our EKF *does* expose a point estimate of `K^P`,
handles those extensions, and — with the effective lower bound pushed far below
any yield (`elb = PLAIN_ELB`) — collapses exactly to plain AFNS:

    f_lb(τ) = ELB + (f_s-ELB)Φ(z) + ω φ(z),  z=(f_s-ELB)/ω  --->  f_s   as ELB→-∞

so the fitted ELB yields equal the plain-AFNS fitted yields and the shadow short
rate `L+S` is just the AFNS short rate. We then decompose with the *same*
`model.term_premium.expected_rate_yields` used to cross-check the Julia engine.

Before trusting this engine on 1961+/regimes, `validate_vs_sf_fed()` confirms its
1987+ term premium reproduces the Julia/SF-Fed-validated result within the
established gate — see `validation/FINDINGS_phase3.md`.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from model import ekf
from model import shadow_rate as sr
from model import term_premium as tp
from us.config import COMMON_GRID
from us.fetch_gsw import build_yield_panel

logger = logging.getLogger(__name__)

# ELB pushed to -100%: below any plausible yield, so the Krippner option floor is
# never active and f_lb == f_s exactly (plain AFNS). z=(f_s-elb)/ω ~ O(10+) makes
# Φ(z)=1, φ(z)=0 to machine precision, so no convexity distortion leaks in.
PLAIN_ELB = -1.0


def _panel(freq, start, end):
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


def decompose(params: "ekf.ShadowParams", panel: pd.DataFrame, dt: float,
              n_per_year: int = 24) -> dict:
    """Given estimated plain-AFNS params, produce the canonical decomposition.

    Returns a dict with the tidy long frame ``ours`` (date, maturity,
    term_premium, expected_rate, fitted_yield) plus the raw arrays.
    """
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    xs, fits = ekf.filter_states(params, Y, grid, PLAIN_ELB, dt, n_per_year=n_per_year)
    # Expected-average-short-rate yields under P; TP = fitted (Q) - expected (P).
    expected = tp.expected_rate_yields(xs, params.K, params.theta, sr.RHO, grid)
    premium = tp.term_premium(fits, expected)

    idx = panel.index
    rows = []
    for j, mat in enumerate(grid):
        rows.append(pd.DataFrame({
            "date": idx, "maturity": float(mat),
            "term_premium": premium[:, j], "expected_rate": expected[:, j],
            "fitted_yield": fits[:, j],
        }))
    ours = pd.concat(rows, ignore_index=True)
    return {
        "ours": ours, "states": xs, "fitted": fits, "expected": expected,
        "term_premium": premium, "grid": grid, "dates": idx,
        "yield_fit_mad_bp": float(np.nanmean(np.abs(fits - Y)) * 1e4),  # ragged-safe
    }


def run_afns(freq: str = "monthly", start="1987-01-01", end=None,
             maxiter: int = 4000, warm_start: "ekf.ShadowParams | None" = None,
             diagonal_K: bool = True) -> dict:
    """Estimate the plain-AFNS (EKF, bound pushed away) and decompose the premium.

    Independent-factor by default (`diagonal_K=True`), matching the SF Fed spec.
    """
    panel = _panel(freq, start, end)
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    dt = 7.0 / 365.2425 if freq == "daily" else 1.0 / 12
    npy = 24
    logger.info("estimating plain AFNS (EKF, elb=%.2f) on %d obs (%s..%s)",
                PLAIN_ELB, len(panel), panel.index.min().date(), panel.index.max().date())
    params, res, ll = ekf.estimate(Y, grid, PLAIN_ELB, dt, warm_start=warm_start,
                                   maxiter=maxiter, n_per_year=npy, diagonal_K=diagonal_K)
    logger.info("done: loglik=%.1f success=%s lambda=%.4f eig(K^P)=%s",
                ll, res.success, params.lambda_,
                np.round(np.sort(np.linalg.eigvals(params.K).real), 4))
    out = decompose(params, panel, dt, n_per_year=npy)
    out.update({"params": params, "loglik": ll, "success": bool(res.success),
                "panel": panel, "dt": dt})
    return out


def apply_bias_correction(result: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bauer-Rudebusch-Wu bias-correct the P-persistence of a plain-AFNS fit and
    re-decompose the term premium.

    Keeps the fitted (Q) yields and the unconditional mean ``θ`` fixed; only
    ``K^P`` changes (→ a more persistent expected-rate path, generally a higher /
    steeper premium). Returns ``{ours_bc, bc, expected_bc, term_premium_bc}``.
    """
    from model import bias_correction as bcm

    bc = bcm.bias_correct(result["states"], result["dt"], n_boot=n_boot, seed=seed)
    grid = result["grid"]
    expected_bc = tp.expected_rate_yields(result["states"], bc.K_bc,
                                          result["params"].theta, sr.RHO, grid)
    premium_bc = tp.term_premium(result["fitted"], expected_bc)

    idx = result["dates"]
    rows = []
    for j, mat in enumerate(grid):
        rows.append(pd.DataFrame({
            "date": idx, "maturity": float(mat),
            "term_premium": premium_bc[:, j], "expected_rate": expected_bc[:, j],
            "fitted_yield": result["fitted"][:, j],
        }))
    return {"ours_bc": pd.concat(rows, ignore_index=True), "bc": bc,
            "expected_bc": expected_bc, "term_premium_bc": premium_bc}


def validate_vs_sf_fed(result: dict, make_plots: bool = False) -> dict:
    """Compare the EKF plain-AFNS decomposition to the cached SF Fed benchmark.

    Returns the `compare_afns.run_comparison` dict (metrics, gate, ...).
    """
    from validation.compare_afns import run_comparison
    from validation.sf_fed import load_sf_fed_series
    bench = load_sf_fed_series()
    return run_comparison(result["ours"], bench, make_plots=make_plots)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Estimate US plain-AFNS term premium (EKF engine)")
    p.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    p.add_argument("--start", default="1987-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--maxiter", type=int, default=4000)
    p.add_argument("--validate", action="store_true", help="compare to cached SF Fed")
    p.add_argument("--out", default=None, help="optional parquet path for the tidy frame")
    args = p.parse_args()

    r = run_afns(freq=args.freq, start=args.start, end=args.end, maxiter=args.maxiter)
    print(f"loglik={r['loglik']:.1f} success={r['success']} "
          f"yield_fit_mad={r['yield_fit_mad_bp']:.2f}bp")
    if args.validate:
        cmp = validate_vs_sf_fed(r)
        print("SF Fed gate:", "PASS" if cmp["gate"]["passed"] else "FAIL")
        for d in cmp["gate"]["details"]:
            if d["status"] != "MISSING":
                print(f"  {d['maturity']:g}y: {d['status']} corr={d['correlation']:.4f} "
                      f"bias={d['mean_bias_bp']:+.1f}bp")
    if args.out:
        r["ours"].to_parquet(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
