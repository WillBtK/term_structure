"""Canada production term premium: shadow-rate AFNS + BRW on the BoC ZC curve.

Thin wrapper over the shared pipeline (`us.production_tp.run_production_tp`). The only
country-specific inputs are the Bank of Canada zero-coupon panel and the ELB. Canada
only grazed its lower bound (2009–10, 2020–21), so the ELB is weakly identified — we
calibrate it just below the observed floor (0%) and ML-refine.
"""
from __future__ import annotations

import logging

from canada import config as cfg
from canada.fetch_boc import build_yield_panel
from us.production_tp import run_production_tp

logger = logging.getLogger(__name__)


def run_canada_production(start: str = cfg.SAMPLE_START, end=None,
                          elb: float = cfg.ELB, estimate_elb: bool = True,
                          maxiter: int = 4000, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """CAD production TP (shadow AFNS + BRW). ELB calibrated near 0 and ML-refined
    (weakly identified — Canada only lightly touched the bound)."""
    panel = build_yield_panel(grid=cfg.COMMON_GRID, freq="M", start=start)
    return run_production_tp(start=start, end=end, panel=panel, elb=elb,
                             estimate_elb=estimate_elb, maxiter=maxiter,
                             n_boot=n_boot, seed=seed)
