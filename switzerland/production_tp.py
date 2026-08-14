"""Switzerland production term premium: shadow-rate AFNS + BRW on the SNB curve.

Thin wrapper over the shared pipeline (`us.production_tp.run_production_tp`). The
only country-specific inputs are the SNB CHF panel and the (deepest) ELB, which is
ML-estimated — Switzerland is the strongest test of the ELB machinery (SNB policy
rate reached −0.75%).
"""
from __future__ import annotations

import logging

from switzerland import config as cfg
from switzerland.fetch_snb import build_yield_panel
from us.production_tp import run_production_tp

logger = logging.getLogger(__name__)


def run_switzerland_production(start: str = cfg.SAMPLE_START, end=None,
                               elb: float = cfg.ELB, estimate_elb: bool = True,
                               maxiter: int = 4000, n_boot: int = 2000,
                               seed: int = 0) -> dict:
    """CHF production TP (shadow AFNS + BRW). ELB ML-estimated by default
    (well identified — the deepest negative-rate market)."""
    panel = build_yield_panel(grid=cfg.COMMON_GRID, freq="M", start=start)
    return run_production_tp(start=start, end=end, panel=panel, elb=elb,
                             estimate_elb=estimate_elb, maxiter=maxiter,
                             n_boot=n_boot, seed=seed)
