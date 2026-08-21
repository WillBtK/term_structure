"""The inflation shape of the term premium, fitted the practical way.

This is the production relationship the scenario tool uses, and it deliberately mixes
horizons. That is a considered choice, not an oversight:

* **The inflation shape is fitted on the FULL history**, back to the 1960s-70s, even
  though the term premium there comes from a model whose parameters were estimated on a
  later, regime-clean sample. The high-inflation branch cannot be seen any other way --
  inside the regime-clean windows almost nothing sits above 5% inflation, and what does
  is all disinflation. Accepting a noisier premium in exchange for the variation that
  identifies the shape is the right trade for a tool whose job is to say where the
  premium settles in a 4% world.
* **The balance-sheet term is fitted ONLY inside the regime-clean window**, on the
  residual from the inflation fit. It is a control on a specific, recent, well-measured
  episode; there is no case for extending it back through data where it means nothing.

Two kinks, one of them chosen rather than searched. The first is fixed at 2% -- the
inflation target in every market this covers -- so the story is "flat while inflation is
at target, rising once it clears". The second is searched, and lands near 3.5%: above it
the marginal effect drops sharply, because by then the regime has already repriced. Both
branches are needed; on the full history dropping the second costs 0.09 of R2 at the ten
-year point.

The numbers this produces are approximations and are meant to be. Projecting a term
premium ten years out from a macro scenario carries far larger errors than the third
decimal place of a slope coefficient, and a model that pretends otherwise is precisely
wrong rather than approximately right.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_KINK = 2.0          # the inflation target, imposed rather than searched
DEFAULT_HALFLIFE_M = 24    # EWMA memory for the inflation level, months

# Each market's regime-clean production window (README), used ONLY for the supply term.
CLEAN_START = {"US": "1987-01-01", "DE": "1984-01-01",
               "JP": "2000-05-01", "UK": "1995-11-01"}


@dataclass(frozen=True)
class InflationShape:
    """TP = intercept + slope_mid*max(pi-2,0) + d_slope*max(pi-kink2,0)."""
    intercept: float
    slope_mid: float
    d_slope: float
    kink2: float
    r2: float
    n: int

    @property
    def slope_high(self) -> float:
        """Marginal effect above the upper kink."""
        return self.slope_mid + self.d_slope

    def premium(self, inflation):
        """Term premium in percent at a SETTLED inflation level.

        The scenario path flatlines, so the EWMA of a settled level is that level --
        the memory parameter matters for fitting, not for the projection.
        """
        pi = np.asarray(inflation, dtype=float)
        return (self.intercept + self.slope_mid * np.maximum(pi - TARGET_KINK, 0.0)
                + self.d_slope * np.maximum(pi - self.kink2, 0.0))


def _ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    return b, r, 1.0 - (r @ r) / ((y - y.mean()) ** 2).sum()


def fit_shape(tp: pd.Series, inflation_ewma: pd.Series,
              kink2_grid=np.arange(2.75, 8.01, 0.25)) -> InflationShape:
    """Fit the two-branch shape, searching only the UPPER kink."""
    d = pd.DataFrame({"tp": tp, "pi": inflation_ewma}).dropna()
    pi, y = d["pi"].values, d["tp"].values
    best = None
    for k2 in kink2_grid:
        c2 = np.maximum(pi - k2, 0.0)
        if c2.std() == 0:
            continue
        b, r, r2 = _ols(y, [np.maximum(pi - TARGET_KINK, 0.0), c2])
        if best is None or (r @ r) < best[0]:
            best = (r @ r, InflationShape(b[0], b[1], b[2], float(k2), r2, len(d)))
    if best is None:
        raise ValueError("no usable upper kink -- inflation never clears the grid")
    return best[1]


UPPER_KINK = 5.0           # imposed, not searched -- see fit_constrained

# The high-inflation marginal effect is floored at this fraction of the market's own
# 2-to-5% slope. This is a judgement, applied openly. The data does say the response
# flattens above 5%, but a flattening all the way to zero would mean inflation risk
# simply stops being priced once inflation is high -- which no investor believes and no
# committee would accept. Since every high-inflation observation we have comes from a
# disinflation with improving credibility, the flat reading is exactly what a sample of
# descents would produce, and it is the wrong analogue for a scenario going the other
# way. Flooring keeps the curve monotone in inflation by construction rather than by
# luck, and it binds only where the fit had gone implausibly flat.
HIGH_SLOPE_FLOOR = 1.0 / 3.0


def fit_constrained(tp_by_market: dict, pi_by_market: dict,
                    identified=("US", "UK")) -> dict:
    """Fit every market with a SHARED upper branch and its own level and sensitivity.

    Letting each market choose its own high-inflation slope fails where a market has
    little high-inflation history: Germany has 34 monthly observations above 5% and its
    fitted upper slope comes out NEGATIVE, implying the premium falls as inflation rises
    past 5% and turning the three-year premium negative at 6%. That is not a finding, it
    is a fit running out of data.

    So the upper slope is estimated once, pooled across the markets that can identify it
    (the US and UK have 158 and 177 such observations), and imposed on the rest. Each
    market keeps its own intercept and its own 2-to-5% sensitivity, which is where the
    real cross-market differences live and where every market has data.

    Returns ``{market: InflationShape}`` plus ``{"_shared_high": slope}``.
    """
    pool_y, pool_x = [], []
    for c in identified:
        d = pd.DataFrame({"tp": tp_by_market[c], "pi": pi_by_market[c]}).dropna()
        pi = d["pi"].values
        pool_y.append(d["tp"].values)
        pool_x.append(np.column_stack([np.maximum(pi - TARGET_KINK, 0.0),
                                       np.maximum(pi - UPPER_KINK, 0.0)]))
    y = np.concatenate(pool_y)
    X = np.vstack(pool_x)
    b, _, _ = _ols(y, [X[:, 0], X[:, 1]])
    g_fitted = float(b[1] + b[2])              # the shared marginal effect above 5%

    out = {"_shared_high_fitted": g_fitted}
    for c, tp in tp_by_market.items():
        d = pd.DataFrame({"tp": tp, "pi": pi_by_market[c]}).dropna()
        pi = d["pi"].values
        mid = np.maximum(pi - TARGET_KINK, 0.0) - np.maximum(pi - UPPER_KINK, 0.0)
        adj = d["tp"].values - g_fitted * np.maximum(pi - UPPER_KINK, 0.0)
        bb, r, _ = _ols(adj, [mid])
        # Floor the upper branch, then re-fit the level so the curve still passes
        # through the data rather than being shifted by the adjustment.
        g = max(g_fitted, HIGH_SLOPE_FLOOR * float(bb[1]))
        adj = d["tp"].values - g * np.maximum(pi - UPPER_KINK, 0.0)
        bb, r, _ = _ols(adj, [mid])
        fitted = (bb[0] + bb[1] * mid + g * np.maximum(pi - UPPER_KINK, 0.0))
        resid = d["tp"].values - fitted
        r2 = 1.0 - (resid @ resid) / ((d["tp"].values - d["tp"].values.mean()) ** 2).sum()
        out[c] = InflationShape(float(bb[0]), float(bb[1]), float(g - bb[1]),
                                UPPER_KINK, float(r2), len(d))
    return out


def fit_supply(residual: pd.Series, duration_share: pd.Series, start: str) -> tuple:
    """Supply coefficient on the inflation-fit residual, inside the clean window.

    Returns ``(coefficient, t_stat, n)``. The regressor is the 12-month change in the
    central bank's ten-year-equivalent share -- the FLOW, not the level, because the
    level is too persistent to identify anything (see FINDINGS_supply.md).
    """
    d = pd.DataFrame({"r": residual, "s": duration_share}).dropna(how="all")
    d["flow"] = d["s"].diff(12)
    d = d.dropna(subset=["r", "flow"])
    d = d[d.index >= pd.Timestamp(start)]
    if len(d) < 40:
        return np.nan, np.nan, len(d)
    b, r, _ = _ols(d["r"].values, [d["flow"].values])
    dof = max(len(d) - 2, 1)
    A = np.column_stack([np.ones(len(d)), d["flow"].values])
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r) / dof)
    return float(b[1]), float(b[1] / se[1]), len(d)


def ewma_level(core_yoy: pd.Series, halflife_m: int = DEFAULT_HALFLIFE_M) -> pd.Series:
    """Trailing inflation level. Two-year memory by default -- long enough to mean
    'the prevailing regime', short enough that the 2021-24 episode registers as the
    high-inflation event it was rather than being averaged away."""
    return core_yoy.ewm(halflife=halflife_m, min_periods=24).mean()


def scenario_premium(tp_today: float, expectations_today: float,
                     expectations_scenario, slope: float) -> "float | np.ndarray":
    """Scenario premium ANCHORED to today's observed level.

        TP(scenario) = TP_today + slope * (pi_scenario - pi_today)

    The model explains how the premium MOVES with long-run inflation expectations. It
    does not explain the whole level, and today it under-predicts the US ten-year premium
    by about 1.4pp and the UK by 1.1pp. That gap is the fiscal and issuance environment --
    deficits, heavy supply, a shrinking official bid -- which is real but which this tool
    deliberately does not forecast.

    Anchoring absorbs it without pretending to model it. The scenario answers "where does
    the premium go RELATIVE to today", today's level comes from the observed premium, and
    the standing assumption is stated rather than buried: the fiscal stance is unchanged
    from today. A user who wants a fiscal scenario overrides this intercept by hand; the
    alternative -- forecasting deficits to forecast a premium -- imports an error far
    larger than the one it removes.

    Below the 2% kink the premium is flat, so the difference is taken on the kinked
    variable rather than on raw expectations.
    """
    a = np.maximum(np.asarray(expectations_scenario, dtype=float) - TARGET_KINK, 0.0)
    b = max(float(expectations_today) - TARGET_KINK, 0.0)
    return float(tp_today) + slope * (a - b)
