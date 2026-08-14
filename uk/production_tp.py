"""UK production term premium — nominal (shadow AFNS, mild fixed ELB) and real
(plain AFNS, no ELB), on the BoE GLC curves.

Thin country wrappers over the shared pipeline (`us.production_tp.run_production_tp`,
BRW-auto + sample-mean-theta + observed-based premium). Only the BoE panels and
the (mild, fixed) ELB differ. The inflation-risk-premium decomposition reuses the
country-agnostic `us.inflation.inflation_decomposition`.
"""
from __future__ import annotations

import logging

from uk import config as cfg
from uk.fetch_boe import build_yield_panel
from us.production_tp import run_production_tp

logger = logging.getLogger(__name__)


def run_uk_nominal(start: str = cfg.PROD_START, end=None, maxiter: int = 4000,
                   n_boot: int = 2000, seed: int = 0, grid=None) -> dict:
    """UK nominal gilt term premium (shadow AFNS, ELB fixed — mild/weakly identified
    UK ZLB). Default sample = inflation-targeting era (1992-10+)."""
    panel = build_yield_panel("nominal", grid=cfg.COMMON_GRID if grid is None else grid,
                              freq="M", start=start)
    return run_production_tp(start=start, end=end, panel=panel, elb=cfg.ELB,
                             estimate_elb=False, maxiter=maxiter, n_boot=n_boot, seed=seed)


def run_uk_real(start: str = cfg.PROD_START, end=None, maxiter: int = 4000,
                n_boot: int = 2000, seed: int = 0, grid=None) -> dict:
    """UK real (index-linked) term premium — plain AFNS via the non-binding bound
    (real rates have no ELB). Real grid 3-20y (no RPI-lag short end)."""
    panel = build_yield_panel("real", grid=cfg.REAL_GRID if grid is None else grid,
                              freq="M", start=start)
    return run_production_tp(start=start, end=end, panel=panel, elb=cfg.ELB_REAL,
                             estimate_elb=False, maxiter=maxiter, n_boot=n_boot, seed=seed)
