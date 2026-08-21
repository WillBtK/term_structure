"""Term premium as a US level plus a country spread.

Four independent driver models fit each market's LEVEL respectably and then explain
almost none of its CHANGE -- which is the only thing a scenario ever uses. Regressed on
twelve-month changes the standalone models return R2 0.046 for the UK, 0.062 for Germany
and 0.081 for Japan. They describe where a premium has been, not where it is going.

Putting the US in as the common level fixes that, because the US is the global rate
setter and the literature has said so for two decades. On twelve-month changes the US
term premium alone explains 0.57 of the UK, 0.49 of Germany and 0.30 of Japan, and
adding each market's drivers RELATIVE to the US takes those to 0.66, 0.56 and 0.35. On
levels the same structure lifts the UK from 0.62 to 0.84, Germany from 0.82 to 0.93 and
Japan from 0.82 to 0.91, roughly halving the RMSE in each.

    TP_i = a_i + b_i * TP_US + c_i * (driver_i - driver_US)

One correction to the motivating intuition, because it changes how the model should be
sold rather than whether to build it. Co-movement is NOT stronger now. Across the four
markets PC1 takes 93.8% of the variance in levels over the whole sample, but that is the
shared collapse in premia from the 1980s; in rolling ten-year windows it falls from 0.89
through 2000 to 0.65 through 2024, and on monthly CHANGES it is 0.53. The US beta has
fallen with it: on 2010+ levels the UK reads 0.65, Germany 0.54 and Japan 0.24, against
0.98, 0.89 and 0.78 over the full sample. Japan has very nearly decoupled -- twelve-month
changes since 2010 give it a US beta of 0.12 -- which is what yield-curve control was
for.

The structure still wins, and wins by MORE on the recent sample (2010+ levels: UK 0.53 ->
0.82, Germany 0.35 -> 0.76, Japan 0.62 -> 0.83). But it wins because the relative drivers
capture the divergence, not because the markets have converged. A committee should be
told the US sets the level and each market's own QE and fiscal path decide how far it
trades away from it -- not that the four markets have become one.

Two things this structure repairs on its own. Germany's fiscal coefficient, unidentified
in isolation and pinned at zero by the sign constraint, comes back at +1.8bp per point of
RELATIVE debt/GDP with t=+18.2; the UK's at +3.3bp with t=+3.4. Only Japan still needs
the calibrated Laubach term. And the price-insensitive driver becomes relative QE scale,
which is the comparison an investor actually makes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from model.tp_drivers import (DEBT_COEF, DRIVERS, MARKETS, SIGNS, build_panel, design,
                              fit_market)

logger = logging.getLogger(__name__)

SPREAD_MARKETS = tuple(c for c in MARKETS if c != "US")

# The US beta is bounded below by zero -- a market whose premium moves AGAINST the US
# level is not a spread market -- and above by 1.5, which no sample comes near.
BETA_BOUNDS = (0.0, 1.5)
MIN_T = 2.0


@dataclass(frozen=True)
class RelativeModel:
    """One market's premium as a beta to the US plus relative drivers."""
    country: str
    intercept: float
    beta_us: float
    coef: dict                       # relative-driver coefficients, FITTED
    tstat: dict = field(default_factory=dict)
    # Channels this market's own history cannot identify. Each is applied to the
    # relative driver measured FROM TODAY, so it contributes nothing at the base case --
    # the historical fit stays honest -- and still responds when a scenario moves it.
    adjust: dict = field(default_factory=dict)
    base: dict = field(default_factory=dict)
    borrowed: tuple = ()             # inflation channels taken from the US
    debt_calibrated: bool = False
    r2: float = float("nan")
    r2_recent: float = float("nan")
    rmse: float = float("nan")
    n: int = 0

    def premium(self, us: dict, own: dict, tp_us: float) -> float:
        """This market's modelled premium, given the US premium and both driver sets."""
        rel = _relative(own, us)
        return (self.intercept + self.beta_us * float(tp_us)
                + sum(self.coef.get(k, 0.0) * rel[k] for k in DRIVERS)
                + sum(g * (rel[k] - self.base[k]) for k, g in self.adjust.items()))

    def scenario(self, us_today: dict, us_scenario: dict, own_today: dict,
                 own_scenario: dict, tp_us_today: float, tp_us_scenario: float,
                 tp_today: float) -> float:
        """Anchored on today's observed premium, exactly as the US model is.

        The model supplies the change and the market supplies the level, so whatever the
        four drivers and the US beta fail to explain is carried forward untouched. An
        unchanged scenario returns today's premium to the decimal.
        """
        return float(tp_today) + (self.premium(us_scenario, own_scenario, tp_us_scenario)
                                  - self.premium(us_today, own_today, tp_us_today))


def _relative(own: dict, us: dict) -> dict:
    """Driver differences, with inflation kinked at the target on each side first."""
    from model.tp_drivers import TARGET_KINK
    out = {}
    for k in DRIVERS:
        a, b = float(own[k]), float(us[k])
        if k == "expected_inflation":
            a, b = max(a - TARGET_KINK, 0.0), max(b - TARGET_KINK, 0.0)
        out[k] = a - b
    return out


def build_pair(country: str, panels: dict | None = None) -> pd.DataFrame:
    """One market joined to the US, with the relative drivers already differenced."""
    panels = panels or {}
    own = panels.get(country)
    own = build_panel(country) if own is None else own
    us = panels.get("US")
    us = build_panel("US") if us is None else us
    j = own.join(us[["tp"] + list(DRIVERS)].add_prefix("us_"), how="inner")
    x, xu = design(j), design(j[[f"us_{k}" for k in DRIVERS]]
                              .rename(columns={f"us_{k}": k for k in DRIVERS}))
    for k in DRIVERS:
        j[f"rel_{k}"] = x[k] - xu[k]
    return j


def fit_relative(country: str, pair: pd.DataFrame | None = None,
                 us_model=None) -> RelativeModel:
    """Sign-constrained fit of one market's spread to the US.

    A channel is FITTED only where this market's own history identifies it. Where the
    sign constraint pins it at zero -- Germany's relative inflation, Japan's relative
    debt -- forcing a number into the historical fit is what wrecked the earlier attempt:
    2.5bp against Japan's hundred-point climb in relative debt demands 2.5pp of premium
    that never appeared, and the fit goes negative. So those channels are carried as
    ADJUSTMENTS measured from today's relative driver. They are worth nothing at the base
    case, which leaves the fit reporting what the data actually supports, and they still
    respond when a scenario moves them, which is the whole point of having them.

    The inflation fallback is the US's own coefficient, because a relative driver's
    coefficient IS the response to a move in this market's driver with the US held still.
    The fiscal fallback is the Laubach mid-point.
    """
    p = build_pair(country) if pair is None else pair
    us_model = us_model or fit_market("US")
    y = p["tp"].values
    base = {k: float(p[f"rel_{k}"].iloc[-1]) for k in DRIVERS}

    coef, tstat, b_us, a = _solve(p, list(DRIVERS), y)
    dead = [k for k in DRIVERS if abs(coef[k]) < 1e-9 or tstat[k] < MIN_T]
    if dead:                                    # refit on what is left, cleanly
        live = [k for k in DRIVERS if k not in dead]
        coef, tstat, b_us, a = _solve(p, live, y)

    adjust, borrowed = {}, []
    for k in dead:
        if k == "debt_gdp":
            adjust[k] = DEBT_COEF
        else:
            adjust[k] = us_model.coef[k]
            borrowed.append(k)

    r = y - _hat(p, coef, b_us, a)
    recent = np.asarray(p.index >= "2010-01-01")
    return RelativeModel(
        country=country, intercept=a, beta_us=b_us, coef=coef, tstat=tstat,
        adjust=adjust, base=base, borrowed=tuple(borrowed),
        debt_calibrated="debt_gdp" in adjust,
        r2=float(1 - (r @ r) / ((y - y.mean()) ** 2).sum()),
        r2_recent=float(1 - (r[recent] ** 2).sum()
                        / ((y[recent] - y[recent].mean()) ** 2).sum()),
        rmse=float(np.sqrt((r ** 2).mean())), n=len(p))


def _solve(p: pd.DataFrame, keys: list, y: np.ndarray) -> tuple:
    A = np.column_stack([np.ones(len(p)), p["us_tp"].values]
                        + [p[f"rel_{k}"].values for k in keys])
    lo = np.array([-np.inf, BETA_BOUNDS[0]]
                  + [0.0 if SIGNS[k] > 0 else -np.inf for k in keys])
    hi = np.array([np.inf, BETA_BOUNDS[1]]
                  + [np.inf if SIGNS[k] > 0 else 0.0 for k in keys])
    b = lsq_linear(A, y, bounds=(lo, hi)).x
    r = y - A @ b
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r) / max(len(y) - A.shape[1], 1))
    coef = {k: float(b[i + 2]) for i, k in enumerate(keys)}
    tstat = {k: float(abs(b[i + 2] / se[i + 2])) if se[i + 2] > 0 else 0.0
             for i, k in enumerate(keys)}
    return coef, tstat, float(b[1]), float(b[0])


def _hat(p: pd.DataFrame, coef: dict, beta: float, a: float) -> np.ndarray:
    """The FITTED premium. Adjustments are deliberately absent: they are measured from
    today, so over history they would impose swings that never happened -- 2.5bp against
    Japan's hundred-point climb in relative debt is 2.5pp of premium the market never
    charged. They contribute nothing at today's values and so change nothing here, and
    everything in a scenario, which is exactly where they belong."""
    out = a + beta * p["us_tp"].values
    for k, g in coef.items():
        out = out + g * p[f"rel_{k}"].values
    return out


def fit_all_relative(markets: tuple = SPREAD_MARKETS) -> dict:
    us_model = fit_market("US")
    panels = {c: build_panel(c) for c in ("US",) + tuple(markets)}
    return {c: fit_relative(c, build_pair(c, panels), us_model) for c in markets}


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    us_model = fit_market("US")
    print(f"US, the common level: R2 {us_model.r2:.3f}  post-2000 "
          f"{us_model.r2_post2000:+.3f}  RMSE {us_model.rmse:.2f}pp  (unchanged)\n")
    panels = {c: build_panel(c) for c in MARKETS}
    for c in SPREAD_MARKETS:
        p = build_pair(c, panels)
        m = fit_relative(c, p, us_model)
        print(f"{c} = a + b*US + relative drivers,  n={m.n}, "
              f"{p.index[0].date()}..{p.index[-1].date()}")
        print(f"  R2 {m.r2:.3f}   2010+ {m.r2_recent:+.3f}   RMSE {m.rmse:.2f}pp")
        print(f"  {'US term premium':24} {m.beta_us:+.3f}")
        for k in DRIVERS:
            if k in m.coef:
                print(f"  {'rel ' + k:24} {m.coef[k]:+.4f}  (t={m.tstat[k]:5.1f})")
            else:
                tag = "calibrated, from today" if k == "debt_gdp" else "from US, from today"
                print(f"  {'rel ' + k:24} {m.adjust[k]:+.4f}  {tag}")
        print()


if __name__ == "__main__":
    _main()
