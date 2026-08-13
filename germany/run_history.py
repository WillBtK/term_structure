"""Germany historical / stagflation exhibit (full 1972+ monthly history).

Separate, clearly-labeled exhibit — *not* joined to the modern production series
(`germany/production_tp.py`). Two lenses, mirroring the US historical exhibit:
regime-switching (regime dating over the Deutsche-Mark + euro eras) and a
single-regime bias-corrected fit (a historical term-premium level). Thin wrappers
over the shared country-agnostic drivers with the German Bundesbank panel.
"""
from __future__ import annotations

from germany import config as cfg
from germany.fetch_bundesbank import build_yield_panel
from us.run_afns_ekf import run_afns, apply_bias_correction
from us.run_regimes import run_regimes


def _panel():
    return build_yield_panel(grid=cfg.COMMON_GRID, freq="M")


def run_germany_regimes(start: str = cfg.HIST_START, S: int = 2,
                        maxiter: int = 4000, restarts: int = 1,
                        warm_maxiter: int = 2000):
    """Markov-switching AFNS regime dating on the full German history."""
    return run_regimes(start=start, S=S, maxiter=maxiter, restarts=restarts,
                       warm_maxiter=warm_maxiter, panel=_panel())


def run_germany_single(start: str = cfg.HIST_START, maxiter: int = 4000,
                       n_boot: int = 2000):
    """Single-regime plain-AFNS on the full history + BRW bias-correction."""
    r = run_afns(start=start, maxiter=maxiter, panel=_panel())
    bc = apply_bias_correction(r, n_boot=n_boot)
    return r, bc
