"""Japan production term premium: MoF JGB shadow-rate AFNS + BRW, with the
Yield-Curve-Control measurement downweight.

Thin country wrapper over the shared, country-agnostic pipeline
(`us.production_tp.run_production_tp`): it supplies the MoF JGB panel, the
Japanese ELB (ML-estimated — JGB yields sat clearly at the bound for years), and
the YCC measurement-variance downweight (`japan.config.ycc_downweight`).

Japan-specific handling:
- **ELB estimated by ML.** NIRP took the policy rate to -0.10% and short JGB
  yields floored around -0.1% to -0.35%; the bound is well identified, so we fit
  it (starting near the NIRP rate) rather than calibrate it.
- **YCC downweight.** Under Yield Curve Control (2016-09 -> 2024-03) the BoJ
  price-pegged the 10y, so it was not a market price. We inflate its measurement
  variance over that window (heaviest while the band was tight) so the peg does
  not distort the estimated expected-rate path. The term premium is then reported
  against **observed** yields at *all* tenors — including the pegged 10y — which
  is exactly the realised, policy-suppressed premium the user wants to see:
  negative out to 10y under QQE/YCC, normalising as the band widened and YCC ended.
- **Ragged maturities.** 10y from 1986-07, 20y from 1986-12, 30y from 1999-09 —
  handled by the EKF's per-date maturity masking (as for the early US long end).

The term premium reported by the shared pipeline is `fitted (ELB) yield -
expected floored short rate`; because the fit tracks observed yields closely
away from the (down-weighted) peg, this is the premium against observed yields.
"""
from __future__ import annotations

import logging

from japan import config as cfg
from japan.fetch_mof import build_yield_panel
from us.production_tp import run_production_tp

logger = logging.getLogger(__name__)


def run_japan_production(start: str = cfg.SAMPLE_START, end=None,
                         elb: float = cfg.ELB, estimate_elb: bool = True,
                         downweight_ycc: bool = True,
                         maxiter: int = 5000, n_boot: int = 2000,
                         seed: int = 0) -> dict:
    """Japan production TP (shadow AFNS + BRW) on the ragged MoF JGB panel.

    `estimate_elb=True` (default) fits the ELB by ML from the NIRP era.
    `downweight_ycc=True` (default) applies the YCC measurement downweight to the
    BoJ-pegged tenors over 2016-09..2024-03; set False for a no-downweight
    sensitivity run.
    """
    panel = build_yield_panel(grid=cfg.COMMON_GRID, freq="M", start=start)
    var_mult_fn = cfg.ycc_downweight if downweight_ycc else None
    # Neutral-rate shifting endpoint (japan.neutral_anchor); inert until the survey/r*
    # CSVs are committed. Aligned to the monthly estimation dates.
    from japan.neutral_anchor import build_nominal_neutral
    endpoint, _ = build_nominal_neutral(panel.index)
    return run_production_tp(start=start, end=end, panel=panel, elb=elb,
                             estimate_elb=estimate_elb, maxiter=maxiter,
                             n_boot=n_boot, seed=seed, var_mult_fn=var_mult_fn,
                             endpoint=endpoint)
