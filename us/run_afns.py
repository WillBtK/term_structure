"""Driver: estimate plain AFNS on the US GSW panel and decompose the term premium.

Pipeline:
    build_yield_panel()            # US GSW zero yields, decimal, 8-point grid
      -> kalman_optimize(n=3)      # plain AFNS MLE (vendored engine)
      -> kalman_filter             # filtered states with the optimal params
      -> model.term_premium.decompose  # fitted / expected-rate / term premium
      -> canonical tidy frame      # feeds validation.compare_afns

Requires the full engine stack (numba/sympy/IPython) and Python <=3.12; set
ARBFREE_BASE_PATH to a writable dir and ensure `model/` is importable
(PYTHONPATH). AFNS MLE is nonlinear — we reuse the engine's
`initial_solution` and retry across its alternative starts, and fall back to the
best partial fit; convergence is logged.

Usage:
    python -m us.run_afns --freq monthly --start 1998-01-01
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import scipy.linalg

from us.config import COMMON_GRID
from us.fetch_gsw import build_yield_panel

logger = logging.getLogger(__name__)


def _resample(panel: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Subsample the daily panel. 'daily' keeps all rows; 'monthly' takes the
    last observation each month (month-end), which keeps MLE tractable."""
    if freq == "daily":
        return panel
    if freq == "monthly":
        # Last available observation each month, KEEPING its real (business-day)
        # date so it aligns with SF Fed's daily series. (resample("M").last()
        # would relabel to calendar month-end, which may be a weekend/holiday.)
        return panel.groupby([panel.index.year, panel.index.month]).tail(1)
    raise ValueError(f"unknown freq {freq!r}")


def estimate_afns(
    yields: pd.DataFrame,
    n: int = 3,
    time_step_size: float | None = None,
    yield_adjustment_term_active: bool = True,
    warm_start: dict | None = None,
):
    """Estimate plain AFNS (n=3) via the vendored engine.

    Tries a sequence of initial solutions and keeps the best by log-likelihood,
    breaking on the first that converges. If ``warm_start`` is given (a dict with
    ``K, theta, Sigma, H, lambda_`` and optional ``lambda_svensson``, e.g. the
    output of a prior lower-frequency fit), it is tried FIRST — which makes a
    large daily MLE tractable by starting near the optimum — with the engine's
    three ``initial_solution`` alternatives kept as fallbacks. Returns a dict of
    the optimal parameters and the filtered states, or raises if all fail.
    """
    from arbfree_dyn_ns.main import initial_solution
    from arbfree_dyn_ns.optimization import kalman_optimize, compute_Q
    from arbfree_dyn_ns.kalman import kalman_filter
    from arbfree_dyn_ns.nss import B_matrix, C_vector

    assert n in (3, 5)
    grid = np.asarray(yields.columns, dtype=float)
    N = yields.shape[1]
    if time_step_size is None:
        # infer from the median spacing of the index
        dt_days = np.median(np.diff(yields.index.values).astype("timedelta64[D]").astype(float))
        time_step_size = float(dt_days) / 365.2425

    # Build the ordered list of (label, theta0, K0, H0, Sigma0, lam0, lam_sv0).
    starts = []
    if warm_start is not None:
        starts.append(
            (
                "warm",
                np.asarray(warm_start["theta"], dtype=float).ravel(),
                np.asarray(warm_start["K"], dtype=float),
                np.asarray(warm_start["H"], dtype=float),
                np.asarray(warm_start["Sigma"], dtype=float),
                float(warm_start["lambda_"]),
                warm_start.get("lambda_svensson"),
            )
        )
    for alt in range(3):
        theta0, K0, H0, Sigma0, lam0, lam_sv0 = initial_solution(
            N, n, grid, yield_adjustment_term_active, alternative=alt
        )
        starts.append((f"alt{alt}", theta0, K0, H0, Sigma0, lam0, lam_sv0))

    best = None
    for label, theta0, K0, H0, Sigma0, lam0, lam_sv0 in starts:
        try:
            logger.info("kalman_optimize: start=%s", label)
            out = kalman_optimize(
                yields,
                theta0,
                K0,
                H0,
                Sigma0,
                lambda_initial=lam0,
                diag_restrictions=(1, 2, 3),
                spd_restrictions=(2, 3),
                lambda_svensson_initial=lam_sv0,
                svensson_active=(n == 5),
                yield_adjustment_term_active=yield_adjustment_term_active,
                out=None,
                time_step_size=time_step_size,
            )
        except Exception:
            logger.exception("kalman_optimize failed for start=%s", label)
            continue

        optres, H_opt, K_opt, Sigma_opt, lam_opt, lam_sv_opt, theta_opt = out[:7]
        ll = -float(optres.fun)  # optres.fun is the negative log-likelihood
        logger.info("start=%s converged=%s loglik=%.2f", label, optres.success, ll)
        if best is None or ll > best["loglik"]:
            best = dict(
                loglik=ll,
                success=bool(optres.success),
                K=np.asarray(K_opt, dtype=float),
                theta=np.asarray(theta_opt, dtype=float).ravel(),
                Sigma=np.asarray(Sigma_opt, dtype=float),
                lambda_=float(lam_opt),
                lambda_svensson=(float(lam_sv_opt) if lam_sv_opt is not None else None),
                H=np.asarray(H_opt, dtype=float),
            )
        if optres.success:
            break

    if best is None:
        raise RuntimeError("AFNS estimation failed for all initial solutions.")

    # Filtered states with the optimal parameters.
    lam, lam_sv, Sigma, K, theta = (
        best["lambda_"], best["lambda_svensson"], best["Sigma"], best["K"], best["theta"],
    )
    B = np.asarray(B_matrix(lam, grid, lam_sv).values, dtype=float)
    c = np.asarray(C_vector(lam, grid, None, Sigma, lam_sv), dtype=float)
    A = scipy.linalg.expm(-K * time_step_size)
    Q = compute_Q(K, Sigma, time_step_size)
    x, ll, *_ = kalman_filter(yields, theta, A, best["H"], Q, B, c=c)
    best["states"] = np.asarray(x, dtype=float)
    best["time_step_size"] = time_step_size
    return best


def _panel(freq, start, end):
    panel = build_yield_panel(grid=COMMON_GRID)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]
    return _resample(panel, freq).dropna()


def run_afns(
    freq: str = "monthly",
    start=None,
    end=None,
    n: int = 3,
    warm_start: dict | None = None,
    return_fit: bool = False,
):
    """Full run -> canonical tidy frame (date, maturity, term_premium,
    expected_rate, fitted_yield) in decimal, on COMMON_GRID. If ``return_fit``,
    returns (frame, fit_dict) so the parameters can warm-start a later run."""
    from term_premium import decompose

    panel = _panel(freq, start, end)
    logger.info("estimating AFNS on %d obs (%s..%s), freq=%s, n=%d",
                len(panel), panel.index.min().date(), panel.index.max().date(), freq, n)

    fit = estimate_afns(panel, n=n, warm_start=warm_start)
    logger.info("estimation done: loglik=%.2f success=%s lambda=%.4f",
                fit["loglik"], fit["success"], fit["lambda_"])

    frame = decompose(
        states=fit["states"],
        K=fit["K"],
        theta=fit["theta"],
        Sigma=fit["Sigma"],
        lambda_=fit["lambda_"],
        grid=np.asarray(panel.columns, dtype=float),
        lambda_svensson=fit["lambda_svensson"],
        dates=panel.index,
    )
    return (frame, fit) if return_fit else frame


def run_daily_warmstarted(start="1987-01-01", end=None, n: int = 3):
    """Fit monthly (fast) to get a good starting point, then warm-start a daily
    full-sample fit. Returns (daily_frame, monthly_fit, daily_fit)."""
    _, monthly_fit = run_afns(freq="monthly", start=start, end=end, n=n, return_fit=True)
    logger.info("monthly warm-start ready (loglik=%.2f); starting daily fit",
                monthly_fit["loglik"])
    daily_frame, daily_fit = run_afns(
        freq="daily", start=start, end=end, n=n, warm_start=monthly_fit, return_fit=True
    )
    return daily_frame, monthly_fit, daily_fit


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Estimate US AFNS and decompose term premium")
    p.add_argument("--freq", default="monthly", choices=["monthly", "daily"])
    p.add_argument("--start", default="1998-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--n", type=int, default=3, choices=[3, 5])
    p.add_argument("--out", default=None, help="optional parquet path for the result")
    args = p.parse_args()

    ours = run_afns(freq=args.freq, start=args.start, end=args.end, n=args.n)
    print(ours.head())
    print(f"rows={len(ours)} maturities={sorted(ours['maturity'].unique())}")
    if args.out:
        ours.to_parquet(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
