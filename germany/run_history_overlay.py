"""Germany 1972+ monthly historical term-premium overlay for the Atlas.

Produces a small committed JSON (`viz/history/germany_hist_tp.json`) that the Atlas
build attaches to DE:nominal as a labelled `history` overlay (see
`viz/thin_daily._attach_germany_history`). It is a **separate historical exhibit**,
NOT the daily production series and NOT spliced onto it.

Method: plain AFNS on the full monthly Bundesbank Svensson history (fast; the ELB
barely binds over the mostly-positive-rate 1972+ span), with the SAME modern
expectations-leg treatment as the daily line — sample-mean theta + the
mean-reversion floor (3.5y half-life) — so the decomposition is consistent with the
production series it sits beside. The daily shadow-rate machinery is intentionally
not used here (its quadrature is far slower and equivalent away from the bound).

Regenerate (rarely — the monthly history is near-static):
    python -m germany.run_history_overlay
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from germany import config as cfg
from germany.fetch_bundesbank import build_yield_panel
from model import shadow_rate as sr
from model import term_premium as tp
from us.run_afns_ekf import run_afns

logger = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parent.parent / "viz" / "history" / "germany_hist_tp.json"
MR_HALFLIFE_YEARS = 3.5
TENORS = tuple(float(t) for t in cfg.COMMON_GRID)   # full curve (0.25y..10y) for the splice


def build(maxiter: int = 4000, out: Path = OUT) -> dict:
    """Estimate the 1972+ monthly historical TP and write the Atlas overlay JSON."""
    panel = build_yield_panel(grid=cfg.COMMON_GRID, freq="M")
    r = run_afns(start=cfg.HIST_START, panel=panel, maxiter=maxiter)
    states = np.asarray(r["states"], float)
    fitted = np.asarray(r["fitted"], float)
    grid = list(np.asarray(r["grid"], float))
    dates = [str(d.date()) for d in r["dates"]]
    K = r["params"].K
    theta = states.mean(axis=0)                                    # sample-mean theta (modern convention)
    K_expect = tp.mean_reversion_floor(K, float(np.log(2) / MR_HALFLIFE_YEARS))
    expected = tp.expected_rate_yields(states, K_expect, theta, sr.RHO, np.asarray(grid))
    premium = fitted - expected                                    # (T, m), decimal
    # OBSERVED yield = the actual Bundesbank monthly curve (the estimation input), aligned
    # to the estimation dates. The expected rate then "drops out": exp = obs − tp, derived
    # downstream, so we don't ship a separately-modelled expected path.
    obs_panel = panel.reindex(pd.to_datetime([d for d in r["dates"]]))
    def _col(mat, arr):
        j = grid.index(mat)
        return [round(float(arr[t, j] * 100), 4) for t in range(len(dates))]
    def _obs_col(mat):
        s = obs_panel[float(mat)] if float(mat) in obs_panel.columns else None
        return ([None if pd.isna(v) else round(float(v) * 100, 4) for v in s]
                if s is not None else [None] * len(dates))
    out_d = {"country": "DE", "instrument": "nominal", "freq": "monthly",
             "start": cfg.HIST_START, "dates": dates,
             "method": ("plain AFNS + mean-reversion floor (3.5y half-life), full-sample "
                        "fit on 1972+ monthly Bundesbank Svensson curve; separate historical "
                        "exhibit, NOT the daily production series"),
             # tenors = term premium (kept for the global factor); obs = the actual
             # observed yield. The explorer derives expected = obs − term premium.
             "tenors": {"%g" % m: _col(m, premium) for m in TENORS},
             "obs": {"%g" % m: _obs_col(m) for m in TENORS}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_d, separators=(",", ":")))
    t10 = out_d["tenors"]["10"]
    logger.info("wrote %s (%d months); 10y TP last=%.2f%% min=%.2f%% max=%.2f%% (loglik=%.0f)",
                out, len(dates), t10[-1], min(t10), max(t10), r["loglik"])
    return out_d


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()


if __name__ == "__main__":
    _main()
