"""US **real** term premium: plain-AFNS on the TIPS real curve (shadow-rate off).

Real short rates have no ELB, so this is the plain-AFNS decomposition — obtained
by reusing the shared shadow pipeline (`us.production_tp.run_production_tp`) with
the bound pushed to `config.ELB_REAL` (-100%), where the Krippner option collapses
to plain AFNS. Same BRW "auto" bias-correction, sample-mean-theta expectations
leg, and observed-based premium as the nominal production series; the only
country/curve-specific inputs are the real panel and the (non-binding) bound.

Output is the **real term premium** = real yield - expected-average real short
rate, on the TIPS grid (2-20y), 2004+ (the reliable TIPS sample).
"""
from __future__ import annotations

import logging

from us import config as cfg
from us.production_tp import run_production_tp
from us.tips import build_real_panel

logger = logging.getLogger(__name__)


def run_real_tp(start: str = cfg.REAL_SAMPLE_START, end=None, maxiter: int = 4000,
                n_boot: int = 2000, seed: int = 0, grid=None) -> dict:
    """US real term premium (plain-AFNS via non-binding bound). Returns the same
    dict shape as `run_production_tp` (headline `ours` frame is the real TP)."""
    panel = build_real_panel(grid=cfg.REAL_GRID if grid is None else grid,
                             freq="M", start=start)
    return run_production_tp(start=start, end=end, panel=panel, elb=cfg.ELB_REAL,
                             estimate_elb=False, maxiter=maxiter, n_boot=n_boot, seed=seed)
