"""Germany production term premium: euro-era (1999+) shadow-rate AFNS + BRW.

Thin country wrapper over the shared, country-agnostic pipeline
(`us.production_tp.run_production_tp`): it supplies the German Bundesbank panel
and the German ELB. As with the US, the modern series is estimated on a modern
sample (the euro era) so pre-1999 Deutsche-Mark-era dynamics do not contaminate
today's estimate. Unlike the US, Bund yields went clearly negative, so the ELB is
**estimated by ML** (well identified) alongside the -1.0% calibration.
"""
from __future__ import annotations

import logging

from germany import config as cfg
from germany.fetch_bundesbank import build_yield_panel
from us.production_tp import run_production_tp

logger = logging.getLogger(__name__)


def run_germany_production(start: str = cfg.PROD_START, end=None,
                           elb: float = cfg.ELB, estimate_elb: bool = True,
                           maxiter: int = 5000, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Euro-era German production TP (shadow AFNS + BRW). `estimate_elb=True`
    (default) fits the ELB by ML from the negative-rate era; pass a fixed `elb`
    for a calibration/sensitivity run."""
    panel = build_yield_panel(grid=cfg.COMMON_GRID, freq="M")
    return run_production_tp(start=start, end=end, panel=panel, elb=elb,
                             estimate_elb=estimate_elb, maxiter=maxiter,
                             n_boot=n_boot, seed=seed)
