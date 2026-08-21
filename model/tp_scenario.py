"""Scenario -> term premium: the assembled chain.

    terminal core inflation
        -> (Okun 1971)              inflation uncertainty          model/okun.py
        -> (Brixton et al. 2023)    stock-bond correlation         model/stock_bond.py
        -> (fair value)             equilibrium term premium       here
        -> (error correction)       term premium at horizon h      here

Two deliberate design choices.

**Fair value plus a speed, not a level regression.** A scenario asks two separable
questions — where does the premium settle in this world, and how much of that has
happened by the horizon I care about. An error-correction form answers both and keeps
them visible: ``TP*`` is the destination, ``lam`` is the speed. A level regression would
silently conflate them.

**Horizon matters, and not the way one might guess.** Fitted on the four markets whose
samples span the regime change, the 3y premium reverts at about 0.49/year (half-life
~1 year) and the 10y at about 0.32/year (~1.8 years). So a 3-year scenario horizon closes
roughly 87% of the gap at the 3y tenor but only ~68% at the 10y tenor, while a 10-year
horizon is essentially fully converged at both. The 10y-horizon answer is therefore close
to pure fair value; the 3y-horizon answer still depends materially on where the premium
starts today.

Pure: no I/O, no network. Fitted parameters live in ``validation/FINDINGS_tp_scenario.md``
and are reproduced by ``validation/tp_scenario_fit.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.okun import OkunParams
from model.stock_bond import SBCParams


@dataclass(frozen=True)
class FairValueParams:
    """``tp_star = intercept + b_sbc * sbc + b_gap * output_gap`` (premium in PERCENT)."""

    intercept: float
    b_sbc: float
    b_gap: float = 0.0
    lam: float = 0.3            # annual error-correction speed toward tp_star
    r2: float = float("nan")
    n: int = 0

    def fair_value(self, sbc, output_gap=0.0):
        """Equilibrium term premium implied by a correlation and an output gap."""
        return (self.intercept
                + self.b_sbc * np.asarray(sbc, dtype=float)
                + self.b_gap * np.asarray(output_gap, dtype=float))

    def project(self, tp_today, tp_star, horizon_years):
        """Partial adjustment from today's premium toward fair value over the horizon.

        Geometric convergence at ``lam`` per year, so the share of the gap closed is
        ``1 - (1-lam)**h``. Returns ``tp_star`` exactly when the horizon is long enough
        for that share to reach one.
        """
        h = np.asarray(horizon_years, dtype=float)
        closed = 1.0 - (1.0 - self.lam) ** h
        return np.asarray(tp_today, dtype=float) + closed * (
            np.asarray(tp_star, dtype=float) - np.asarray(tp_today, dtype=float))

    def share_closed(self, horizon_years) -> float:
        """Fraction of the gap to fair value closed by ``horizon_years``."""
        return float(1.0 - (1.0 - self.lam) ** float(horizon_years))


def scenario_term_premium(terminal_core_inflation, tp_today, horizon_years,
                          okun: OkunParams, sbc_model: SBCParams, fv: FairValueParams,
                          sigma_g: float, output_gap: float = 0.0) -> dict:
    """Run the whole chain for one scenario and return every intermediate.

    Intermediates are returned, not just the answer, because the point of routing through
    an observable is that each step can be shown and challenged separately. An investment
    committee that disagrees with the output should be able to see whether it disagrees
    about the inflation uncertainty, the correlation, or the premium mapping.

    ``sigma_g`` must be measured the same way it was in estimation — the correlation model
    is a RATIO, so substituting a differently-scaled growth-volatility proxy silently
    rescales the answer.
    """
    sigma_pi = okun.sigma(terminal_core_inflation)
    sbc = sbc_model.predict(sigma_pi, sigma_g)
    tp_star = fv.fair_value(sbc, output_gap)
    tp_h = fv.project(tp_today, tp_star, horizon_years)
    return {
        "terminal_core_inflation": float(terminal_core_inflation),
        "sigma_pi": float(sigma_pi),
        "sigma_g": float(sigma_g),
        "ratio": float(sigma_pi / sigma_g),
        "stock_bond_corr": float(sbc),
        "tp_fair_value": float(tp_star),
        "tp_today": float(tp_today),
        "horizon_years": float(horizon_years),
        "share_closed": fv.share_closed(horizon_years),
        "tp_at_horizon": float(tp_h),
    }


def fit_fair_value(tp, sbc, output_gap=None) -> FairValueParams:
    """Least-squares fair-value regression. ``lam`` is fitted separately by
    :func:`fit_error_correction` and set on the returned params by the caller."""
    tp = np.asarray(tp, dtype=float)
    cols = [np.asarray(sbc, dtype=float)]
    if output_gap is not None:
        cols.append(np.asarray(output_gap, dtype=float))
    X = np.column_stack([np.ones(len(tp))] + cols)
    beta, *_ = np.linalg.lstsq(X, tp, rcond=None)
    resid = tp - X @ beta
    r2 = 1.0 - (resid @ resid) / ((tp - tp.mean()) ** 2).sum()
    b_gap = float(beta[2]) if output_gap is not None else 0.0
    return FairValueParams(float(beta[0]), float(beta[1]), b_gap, r2=r2, n=len(tp))


def fit_error_correction(d_tp, gap_to_fair) -> float:
    """Annual convergence speed ``lam`` from ``dTP_t = lam * (TP*_{t-1} - TP_{t-1})``."""
    y = np.asarray(d_tp, dtype=float)
    x = np.asarray(gap_to_fair, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 8:
        raise ValueError(f"need at least 8 observations to fit, got {len(y)}")
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])
