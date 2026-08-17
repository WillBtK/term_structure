"""Japan 1974+ monthly historical term-premium overlay — the 9y backfill that
extends the JGB 10y term premium back to 1974.

MoF publishes the 10y JGB yield only from 1986-07, but the ≤9y tenors run back to
1974-09 (see ``japan.config``). So we estimate a plain-AFNS term premium on the
monthly MoF curve with a **9y-inclusive grid** (10y ragged, NaN before 1986-07), read
off the **9-year** premium 1974+, and — downstream (``model.global_factor`` /
``viz.thin_daily``) — splice it onto the production 10y at the earliest overlap month
(1986-07) with a one-time level adjustment, giving an estimated 10y premium back to 1974.

Method mirrors the German 1972+ overlay (``germany.run_history_overlay``): plain AFNS
(EKF, bound pushed away) with the modern expectations-leg treatment — sample-mean theta
+ the mean-reversion floor (3.5y half-life). The estimation window is capped at 2015-12,
BEFORE the BoJ's negative-rate/YCC era, so the plain (no-floor) fit stays valid; we only
use the 9y premium for 1974–1986 anyway (the modern period comes from the production line).

Regenerate (rarely — MoF history is near-static; Japan is egress-blocked from the dev
sandbox so run on a GitHub runner):
    python -m japan.run_history_overlay
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from japan.fetch_mof import build_yield_panel
from model import shadow_rate as sr
from model import term_premium as tp
from us.run_afns_ekf import run_afns

logger = logging.getLogger(__name__)
OUT = Path(__file__).resolve().parent.parent / "viz" / "history" / "japan_hist_tp.json"
HIST_START = "1974-09-24"
HIST_END = "2015-12-31"                       # pre-negative-rate / pre-YCC: plain AFNS valid
MR_HALFLIFE_YEARS = 3.5
GRID = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 10.0])   # 9y back to 1974; 10y ragged (1986-07+)
TENORS = (9.0, 10.0)


def build(maxiter: int = 4000, out: Path = OUT) -> dict:
    """Estimate the 1974+ monthly historical JGB TP (9y + ragged 10y) and write the JSON."""
    panel = build_yield_panel(grid=GRID, freq="M", start=HIST_START, refresh=True)
    panel = panel[panel.index <= __import__("pandas").Timestamp(HIST_END)]
    r = run_afns(start=HIST_START, panel=panel, maxiter=maxiter)
    states = np.asarray(r["states"], float)
    fitted = np.asarray(r["fitted"], float)
    grid = list(np.asarray(r["grid"], float))
    dates = [str(d.date()) for d in r["dates"]]
    K = r["params"].K
    theta = states.mean(axis=0)                                    # sample-mean theta
    K_expect = tp.mean_reversion_floor(K, float(np.log(2) / MR_HALFLIFE_YEARS))
    expected = tp.expected_rate_yields(states, K_expect, theta, sr.RHO, np.asarray(grid))
    premium = fitted - expected                                    # (T, m), decimal
    out_d = {"country": "JP", "instrument": "nominal", "freq": "monthly",
             "start": HIST_START, "end": HIST_END, "dates": dates,
             "method": ("plain AFNS + mean-reversion floor (3.5y half-life), monthly MoF "
                        "JGB curve 1974-09..2015-12 with a 9y-inclusive grid (10y ragged "
                        "from 1986-07); the 9y premium backfills the 10y before 1986 via a "
                        "one-time level splice — NOT the daily production series"),
             "tenors": {"%g" % m: [round(float(premium[t, grid.index(m)] * 100), 4)
                                   for t in range(len(dates))] for m in TENORS}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_d, separators=(",", ":")))
    t9 = out_d["tenors"]["9"]
    logger.info("wrote %s (%d months); 9y TP last=%.2f%% min=%.2f%% max=%.2f%% (loglik=%.0f success=%s)",
                out, len(dates), t9[-1], min(t9), max(t9), r["loglik"], r["success"])
    return out_d


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build()


if __name__ == "__main__":
    _main()
