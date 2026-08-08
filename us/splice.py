"""Two-model splice: regime-switching AFNS (pre-2008) + shadow-rate AFNS (2008+).

Per the project charter, the Kim-filter collapsing approximation and the
shadow-rate nonlinearity are **not** combined in one estimation (that compounds
two approximate/nonlinear techniques and breaks the weekly/monthly re-estimate +
daily-filter-forward cadence). Instead two separately-estimated models are joined
at a fixed splice date:

- **< 2008-12-31**: regime-switching AFNS (`model.regime_switching`), regime-mixed
  factors and term premium (validated in `validation/FINDINGS_kim_validation.md`).
  Frozen history — never re-estimated.
- **>= 2008-12-31**: shadow-rate AFNS (`us.run_shadow`) — the live model
  (re-estimate slow, filter forward daily), which respects the ELB.

A **common λ** is shared (the regime leg fixes λ to the shadow leg's estimate) so
the level/slope/curvature factors are on the same basis at the seam. Away from the
bound the shadow term premium collapses to the plain-AFNS term premium, so the two
legs coincide over their overlap — `continuity_check` quantifies that.

The splice date (2008-12-31, the ZLB onset) is the first date the shadow model does
something the regime model can't, and cleanly separates frozen history from the
live model.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from model import regime_switching as rs
from us.config import COMMON_GRID
from us.run_regimes import _panel as _regime_panel
from us.run_shadow import run_shadow

logger = logging.getLogger(__name__)

SPLICE_DATE = pd.Timestamp("2008-12-31")
OVERLAP_START = pd.Timestamp("1987-01-01")   # where the two legs overlap


def estimate_shadow_leg(start="1987-01-01", end=None, maxiter=4000,
                        warm_start=None) -> dict:
    """Estimate the post-2008 shadow-rate AFNS (covers the overlap too)."""
    r = run_shadow(freq="monthly", start=start, end=end, maxiter=maxiter,
                   warm_start=warm_start)
    return {
        "dates": r["dates"], "grid": r["grid"], "lambda_": float(r["params"].lambda_),
        "factors": r["factors"], "term_premium": r["term_premium"],
        "shadow": r["shadow"], "params": r["params"], "loglik": r["loglik"],
        "success": r["success"],
    }


def _regime_warm_start(Y, grid, lambda_fixed, S=2):
    """Vol-contrasted 2-regime start (persistent/high-vol vs calm), λ fixed."""
    lvl = float(np.nanmean(Y[:, -1])); slp = float(np.nanmean(Y[:, 0] - Y[:, -1]))
    return rs.RSParams(
        kappa=np.array([[0.03, 0.30, 0.60], [0.20, 0.50, 0.90]])[:S],
        theta=np.tile([lvl, slp, 0.0], (S, 1)),
        sigma=np.array([[0.020, 0.026, 0.032], [0.007, 0.009, 0.011]])[:S],
        lambda_=lambda_fixed, h=0.001, p_stay=np.full(S, 0.96))


def estimate_regime_leg(lambda_fixed: float, start="1961-06-01", end="2008-12-31",
                        S: int = 2, maxiter: int = 3000, restarts: int = 1,
                        n_per_year: int = 12) -> dict:
    """Estimate the pre-2008 regime-switching AFNS with λ fixed to the shadow leg.

    Returns regime-mixed factors and term premium (mixed by the smoothed regime
    probabilities), plus the smoothed probabilities and the fitted params.
    """
    panel = _regime_panel(start, end)
    grid = np.asarray(panel.columns, dtype=float)
    Y = panel.to_numpy(dtype=float)
    dt = 1.0 / 12
    logger.info("regime leg: %d obs (%s..%s), lambda fixed=%.4f",
                len(panel), panel.index.min().date(), panel.index.max().date(),
                lambda_fixed)
    warm = _regime_warm_start(Y, grid, lambda_fixed, S)
    params, res, ll = rs.estimate(Y, grid, dt, S=S, warm_start=warm, maxiter=maxiter,
                                  n_per_year=n_per_year, restarts=restarts,
                                  fix_lambda=lambda_fixed)
    filt = rs.kim_filter(params, Y, grid, dt, n_per_year=n_per_year, smooth=True)
    sm = filt["smoothed_prob"]
    mixed_factors = rs.regime_mixed(filt["states"], sm)                 # (T,3)
    dec = rs.regime_term_premium(params, filt["states"], grid, n_per_year=n_per_year)
    mixed_tp = rs.regime_mixed(dec["term_premium"], sm)                 # (T,m)

    idx = panel.index
    return {
        "dates": idx, "grid": grid,
        "factors": pd.DataFrame(mixed_factors, index=idx, columns=["L", "S", "C"]),
        "term_premium": pd.DataFrame(mixed_tp, index=idx, columns=grid),
        "smoothed_prob": pd.DataFrame(sm, index=idx),
        "params": params, "loglik": ll, "success": bool(res.success),
    }


def splice(regime_leg: dict, shadow_leg: dict,
           splice_date=SPLICE_DATE) -> pd.DataFrame:
    """Join the two legs into one continuous series (regime < seam, shadow >=)."""
    grid = shadow_leg["grid"]
    rf = regime_leg["factors"]; rtp = regime_leg["term_premium"]
    sf = shadow_leg["factors"]; stp = shadow_leg["term_premium"]
    sh = shadow_leg["shadow"].set_index("date")["shadow_short_rate"]

    pre_mask = rf.index < splice_date
    post_mask = sf.index >= splice_date

    def _leg(fac, tpf, sr_series, mask):
        out = fac.loc[mask, ["L", "S", "C"]].copy()
        out["shadow_short_rate"] = (sr_series.reindex(out.index)
                                    if sr_series is not None
                                    else out["L"] + out["S"])
        for j, mat in enumerate(grid):
            out[f"tp_{mat:g}y"] = tpf.loc[mask].iloc[:, j].values
        return out

    pre = _leg(rf, rtp, None, pre_mask)          # pre-2008: shadow rate = L+S (no bound)
    post = _leg(sf, stp, sh, post_mask)
    joined = pd.concat([pre, post]).sort_index()
    joined.index.name = "date"
    return joined


def continuity_check(regime_leg: dict, shadow_leg: dict,
                     splice_date=SPLICE_DATE, overlap_start=OVERLAP_START) -> dict:
    """Compare the two legs' L/S/C over their overlap and at the seam."""
    rf = regime_leg["factors"]; sf = shadow_leg["factors"]
    rtp = regime_leg["term_premium"]; stp = shadow_leg["term_premium"]
    grid = shadow_leg["grid"]
    j10 = int(np.argmin(np.abs(grid - 10.0)))

    common = rf.index.intersection(sf.index)
    common = common[(common >= overlap_start) & (common <= splice_date)]
    r = rf.loc[common]; s = sf.loc[common]

    per_factor = {}
    for c in ["L", "S", "C"]:
        a, b = r[c].to_numpy(), s[c].to_numpy()
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else np.nan
        per_factor[c] = {"corr": round(corr, 4),
                         "max_abs_gap_bp": round(float(np.max(np.abs(a - b))) * 1e4, 2),
                         "mean_abs_gap_bp": round(float(np.mean(np.abs(a - b))) * 1e4, 2)}

    # Seam: both legs evaluated at the splice date (regime covers it; shadow too).
    def _at(df, dt):
        idx = df.index[df.index <= dt]
        return df.loc[idx[-1]] if len(idx) else None
    r_seam = _at(rf, splice_date); s_seam = _at(sf, splice_date)
    r_tp10 = float(rtp.loc[rtp.index <= splice_date].iloc[-1, j10])
    s_tp10 = float(stp.loc[stp.index >= splice_date].iloc[0, j10])

    return {
        "overlap_n": int(len(common)),
        "overlap_window": [str(common.min().date()), str(common.max().date())] if len(common) else None,
        "per_factor": per_factor,
        "seam_date": str(splice_date.date()),
        "regime_seam_LSC": [round(float(r_seam[c]), 5) for c in ["L", "S", "C"]] if r_seam is not None else None,
        "shadow_seam_LSC": [round(float(s_seam[c]), 5) for c in ["L", "S", "C"]] if s_seam is not None else None,
        "seam_gap_LSC_bp": [round((float(r_seam[c]) - float(s_seam[c])) * 1e4, 2)
                            for c in ["L", "S", "C"]] if (r_seam is not None and s_seam is not None) else None,
        "regime_10y_tp_pct": round(r_tp10 * 100, 3),
        "shadow_10y_tp_pct": round(s_tp10 * 100, 3),
        "tp10_seam_gap_bp": round((r_tp10 - s_tp10) * 1e4, 2),
    }


def run_splice(shadow_maxiter=4000, regime_maxiter=3000, regime_restarts=1) -> dict:
    """Estimate both legs (shadow first, then regime with λ fixed to it) and splice."""
    shadow_leg = estimate_shadow_leg(maxiter=shadow_maxiter)
    regime_leg = estimate_regime_leg(lambda_fixed=shadow_leg["lambda_"],
                                     maxiter=regime_maxiter, restarts=regime_restarts)
    joined = splice(regime_leg, shadow_leg)
    cont = continuity_check(regime_leg, shadow_leg)
    return {"spliced": joined, "continuity": cont,
            "regime_leg": regime_leg, "shadow_leg": shadow_leg}


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Regime/shadow term-premium splice (US)")
    ap.add_argument("--out", default=None, help="parquet path for the spliced series")
    args = ap.parse_args()
    r = run_splice()
    import json
    print(json.dumps(r["continuity"], indent=2, default=str))
    if args.out:
        r["spliced"].to_parquet(args.out)
        print("wrote", args.out)


if __name__ == "__main__":
    _main()
