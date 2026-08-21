"""US ten-year term premium: the driver model behind the scenario tool.

Five drivers, and the reason there are five is that one is not enough. Long-run expected
inflation explains the great disinflation and almost nothing since -- it has sat near
2.3% for twenty-five years while the premium swung through four points -- so a model
built on it alone reads today as 1.4pp rich, which is not a view anyone holds. What
moved the premium after expectations became anchored was how UNCERTAIN inflation was,
how much paper there was, and who was holding it.

    TP = a + b1*max(pi-2,0) + b2*sigma_pi + b3*official + b4*fed + b5*debt

Two choices worth stating.

**Foreign holdings are split into OFFICIAL and PRIVATE.** They are not the same buyer.
A reserve manager holds Treasuries because they are reserves; a private foreign investor
holds them because the yield is attractive, and so responds to the price rather than
setting it. Tested, that is exactly what the data says: official enters at -0.064
(t=-12.6) and private at -0.029 (t=-1.2). Only the price-insensitive holder compresses
the premium, and the official coefficient lands within a whisker of the Federal
Reserve's own (-0.066) -- which is what you would expect if the mechanism is duration
absorbed by someone who does not care what it costs. The two have also diverged sharply:
official ownership has fallen from 24.6% of GDP in 2013 to 12.1%, while private foreign
holdings have risen to a record 16.6%. A combined measure hides that completely.

**Inflation enters as the SURVEY EXPECTATION, not a moving average of realised core.**
A discounted average of past inflation reads the mid-1970s as 14.8% -- a level nobody
ever expected to persist -- and implies a premium far above anything the market priced.
What prices a ten-year bond is what holders expect inflation to average over its life.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

logger = logging.getLogger(__name__)

TARGET_KINK = 2.0

# The US fit that motivated the split.
US_DRIVERS = ("expected_inflation", "inflation_uncertainty",
              "foreign_official", "fed_holdings", "debt_gdp")

# The specification that travels. The Federal Reserve and the foreign official sector
# enter the US fit at -0.069 and -0.064 -- statistically the same coefficient -- so
# combining them costs almost nothing and buys one specification for all four markets,
# which is worth more in front of a committee than a fifth of a percentage point of fit.
DRIVERS = ("expected_inflation", "inflation_uncertainty",
           "price_insensitive", "debt_gdp")

CORE = DRIVERS[:3]                 # always estimated
MARKETS = ("US", "UK", "DE", "JP")

# The sign of every driver is known before a single regression is run, and where the
# data disagrees the data is wrong. Unconstrained, Germany reads MORE debt as LOWER
# premium (-0.062, t=-11.0) and Japan reads MORE inflation uncertainty as lower premium
# (-0.478, t=-6.4). Both are real features of those samples -- Bunds got scarcer as
# issuance rose because the Eurosystem bought faster, and Japan's premium fell for
# twenty years while debt climbed -- and both are useless for projecting forward. A
# committee shown a model that says a bigger deficit cheapens the bond will stop
# listening, and rightly. So the signs are imposed and the fit works around them.
SIGNS = {"expected_inflation": +1, "inflation_uncertainty": +1,
         "price_insensitive": -1, "debt_gdp": +1,
         "foreign_official": -1, "fed_holdings": -1, "foreign_private": -1}

# The fiscal coefficient is CALIBRATED, not estimated, and it does not enter the
# historical fit at all.
#
# No modern sample identifies it. In Germany and Japan the premium fell for two decades
# while debt climbed -- because the Eurosystem and the Bank of Japan were absorbing the
# issuance faster than it arrived -- so the regression reads more debt as a cheaper
# bond, and a sign constraint then pins the coefficient at exactly zero. Forcing a
# positive coefficient into the fit instead is worse: 2.5bp against Japan's 90-point
# climb in the debt ratio demands 2.3pp of premium that never appeared, and the fit
# collapses (R2 -1.17).
#
# So each market is asked whether its own history identifies a fiscal effect, and only
# the two that do keep one in the fit. The US answers +1.4bp per point of debt/GDP
# (t=+6.7) and the UK +2.3bp (t=+3.4). Germany and Japan cannot answer, and rather than
# forcing a coefficient onto them the model carries the fiscal channel there as a
# scenario adjustment measured from today's debt ratio: 2.5bp per point, the mid-point
# of Laubach (2009), whose 3-4bp is the standard reference and which the CBO's own
# review puts at 2-3bp.
#
# The calibration is not an assumption pulled from the air -- it sits between the two
# numbers the markets that CAN measure it came back with. Every market ends up with a
# live fiscal channel and none of them has to pretend its own history produced one.
DEBT_COEF = 0.025
MIN_DEBT_T = 2.0

# A market estimates a coefficient only where the driver actually moves in its sample.
# Japan has SIX observations out of 388 above the inflation target, so its inflation
# coefficient is fitted on six points and comes back at +8.1 -- 810bp of premium per
# point of expected inflation. Where a driver's effective standard deviation is less
# than this fraction of the US's, or where the sign constraint binds at zero, the market
# takes the US coefficient instead: the US is the only sample here that spans a full
# inflation cycle, which is exactly the information the other market lacks.
MIN_VARIATION = 0.20

# FRED identifiers. Holdings are shares of GDP so the model is scale-free.
SERIES = {
    "foreign_official": "BOGZ1FL263061130Q",   # Z.1, quarterly from 1945
    "foreign_private": "BOGZ1FL263061145Q",
    "fed_holdings": "HBFRGDQ188S",
    "debt_public": "FYGFDPUN",
    "gdp": "GDP",
}


@dataclass(frozen=True)
class DriverModel:
    intercept: float
    coef: dict
    tstat: dict = field(default_factory=dict)
    r2: float = float("nan")
    r2_post2000: float = float("nan")
    rmse: float = float("nan")
    n: int = 0

    drivers: tuple = DRIVERS
    country: str = ""
    borrowed: tuple = ()
    driver_sd: dict = field(default_factory=dict)
    debt_fitted: bool = True
    debt_base: float = 0.0

    def premium(self, **kw) -> float:
        """Modelled term premium in percent, from a full set of driver values."""
        missing = set(self.drivers) - set(kw)
        if missing:
            raise TypeError(f"premium() missing drivers: {sorted(missing)}")
        x = dict(kw)
        x["expected_inflation"] = max(float(x["expected_inflation"]) - TARGET_KINK, 0.0)
        # Where the fiscal term is calibrated rather than fitted it is measured FROM
        # today's debt ratio, so it contributes nothing at the base case and only what a
        # scenario moves. Where it is fitted it is already part of the level.
        x["debt_gdp"] = float(x["debt_gdp"]) - (0.0 if self.debt_fitted
                                                else self.debt_base)
        return self.intercept + sum(self.coef[k] * float(x[k]) for k in self.drivers)

    def fit_error(self, panel: pd.DataFrame) -> pd.Series:
        """Modelled minus actual over the estimation sample -- the chart the review
        needs, since the coefficients alone do not show where a market goes wrong."""
        x = design(panel, self.drivers)
        return (self.intercept
                + sum(self.coef[k] * x[k] for k in self.drivers)) - panel["tp"]

    def scenario(self, today: dict, scenario: dict, tp_today: float) -> float:
        """Scenario premium ANCHORED to today's observed level.

        The model supplies the CHANGE in the premium; today's observed premium supplies
        the level. Left to converge on its own fair value the model would forecast the
        US premium falling 74bp inside a year, because it does not explain the standing
        fiscal and structural level -- and the fitted adjustment speed of 0.92 a year
        would close that gap almost immediately. Anchoring holds the unexplained part
        constant instead, which makes every scenario a move FROM today rather than
        toward a fair value we cannot defend. It also means the three-year and ten-year
        horizons return the same premium; they differ only in the policy path and the
        holding period.
        """
        return float(tp_today) + (self.premium(**scenario) - self.premium(**today))


def _us_holdings(index: pd.DatetimeIndex) -> pd.DataFrame:
    """US price-insensitive ownership, split and combined, as a percent of GDP."""
    from validation.fred_api import fred_series
    gdp = fred_series(SERIES["gdp"]).reindex(index).ffill() * 1000       # $bn -> $mn
    out = pd.DataFrame(index=index)
    for k in ("foreign_official", "foreign_private"):
        out[k] = fred_series(SERIES[k]).reindex(index).ffill() / gdp * 100
    out["fed_holdings"] = fred_series(SERIES["fed_holdings"]).reindex(index).ffill()
    out["price_insensitive"] = out["fed_holdings"] + out["foreign_official"]
    return out


def build_panel(country: str = "US") -> pd.DataFrame:
    """Monthly panel of one market's term premium and its drivers."""
    from macro.debt_gdp import history as debt_history
    from macro.fetch_core_cpi import core_inflation_vol
    from macro.inflation_expectations import expectations
    from macro.tp_history import spliced_tp

    d = pd.DataFrame({"tp": spliced_tp(10.0)[country],
                      "expected_inflation": expectations(country)}).dropna()
    if country == "US":
        d = d.join(_us_holdings(d.index))
    else:
        from supply.price_insensitive import price_insensitive
        pi = price_insensitive(country)["total"]
        d["price_insensitive"] = pi.reindex(d.index.union(pi.index)).ffill().reindex(
            d.index)
    # macro.debt_gdp is annual and indexed by calendar year; place each observation at
    # the start of its year and carry it forward through the months.
    debt = debt_history(country)
    debt.index = pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) for y in debt.index])
    d["debt_gdp"] = debt.reindex(d.index.union(debt.index)).ffill().reindex(d.index)
    d["inflation_uncertainty"] = core_inflation_vol()[country].reindex(d.index).ffill()
    keep = ["tp", *dict.fromkeys((*DRIVERS, *(US_DRIVERS if country == "US" else ())))]
    return d[[c for c in keep if c in d]].dropna()


def fit(panel: pd.DataFrame | None = None, drivers: tuple = DRIVERS,
        country: str = "US", constrain: bool = True) -> DriverModel:
    p = build_panel(country) if panel is None else panel
    X = np.column_stack([
        np.ones(len(p)),
        np.maximum(p["expected_inflation"].values - TARGET_KINK, 0.0),
        *[p[k].values for k in drivers[1:]]])
    y = p["tp"].values
    if constrain:
        lo = np.array([-np.inf] + [0.0 if SIGNS[k] > 0 else -np.inf for k in drivers])
        hi = np.array([np.inf] + [np.inf if SIGNS[k] > 0 else 0.0 for k in drivers])
        b = lsq_linear(X, y, bounds=(lo, hi)).x
    else:
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r) / max(len(p) - X.shape[1], 1))
    post = np.asarray(p.index >= "2000-01-01")
    return DriverModel(
        intercept=float(b[0]),
        coef={k: float(b[i + 1]) for i, k in enumerate(drivers)},
        tstat={k: float(b[i + 1] / se[i + 1]) for i, k in enumerate(drivers)},
        r2=float(1 - (r @ r) / ((y - y.mean()) ** 2).sum()),
        r2_post2000=float(1 - (r[post] ** 2).sum()
                          / ((y[post] - y[post].mean()) ** 2).sum()),
        rmse=float(np.sqrt((r ** 2).mean())), n=len(p),
        drivers=drivers, country=country)


def fit_all(markets: tuple = MARKETS) -> dict:
    """One specification, fitted market by market."""
    return {c: fit(build_panel(c), country=c) for c in markets}


def design(p: pd.DataFrame, drivers: tuple = DRIVERS) -> pd.DataFrame:
    """The regressors as the model actually sees them.

    Inflation enters kinked at the target -- ``max(pi - 2, 0)`` -- so any standardising
    or pooling has to be done on the kinked series, not on the level. Standardising the
    level instead is what made Japan's inflation coefficient read +8.07: almost none of
    its history sits above the kink, so the effective regressor barely moves while the
    level moves plenty.
    """
    out = p[list(drivers)].copy()
    out["expected_inflation"] = np.maximum(out["expected_inflation"] - TARGET_KINK, 0.0)
    return out


def _refit_intercept(p: pd.DataFrame, coef: dict, drivers: tuple) -> tuple:
    """Given fixed slopes, the intercept that centres the residual, and the fit."""
    x = {k: p[k].values for k in drivers}
    x["expected_inflation"] = np.maximum(x["expected_inflation"] - TARGET_KINK, 0.0)
    yhat = sum(coef[k] * x[k] for k in drivers)
    a = float((p["tp"].values - yhat).mean())
    r = p["tp"].values - (a + yhat)
    return a, r


def _constrained(X_: pd.DataFrame, y: np.ndarray, cols: list) -> tuple:
    """Sign-constrained least squares with a free intercept; returns (coef, tstat)."""
    A = np.column_stack([np.ones(len(y))] + [X_[k].values for k in cols])
    lo = np.array([-np.inf] + [0.0 if SIGNS[k] > 0 else -np.inf for k in cols])
    hi = np.array([np.inf] + [np.inf if SIGNS[k] > 0 else 0.0 for k in cols])
    b = lsq_linear(A, y, bounds=(lo, hi)).x
    r = y - A @ b
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r) / max(len(y) - A.shape[1], 1))
    return ({k: float(b[i + 1]) for i, k in enumerate(cols)},
            {k: float(b[i + 1] / se[i + 1]) if se[i + 1] > 0 else 0.0
             for i, k in enumerate(cols)})


def fit_market(country: str, panel: pd.DataFrame | None = None,
               reference: DriverModel | None = None) -> DriverModel:
    """The production fit for one market.

    Three estimated coefficients, one calibrated. Debt enters at ``DEBT_COEF`` and is
    partialled out of the premium before fitting, so it neither competes with the
    ownership driver for the same trend nor gets pinned at zero by the sign constraint.
    The remaining three are estimated under their known signs, and any that the market's
    own history cannot identify falls back to the US.
    """
    p = build_panel(country) if panel is None else panel
    X_ = design(p)
    y = p["tp"].values
    debt_base = float(p["debt_gdp"].iloc[-1])

    # Does this market's own history identify a fiscal effect with the right sign?
    trial = _constrained(X_, y, list(DRIVERS))
    debt_fitted = trial[1].get("debt_gdp", 0.0) >= MIN_DEBT_T
    est = list(DRIVERS) if debt_fitted else list(CORE)

    coef, tstat = _constrained(X_, y, est)
    borrowed = []
    if reference is not None:
        ref_sd = reference.driver_sd
        for k in est:
            weak = float(X_[k].std()) < MIN_VARIATION * ref_sd.get(k, np.inf)
            if weak or abs(coef[k]) < 1e-9:
                coef[k], tstat[k] = reference.coef[k], float("nan")
                borrowed.append(k)
    if not debt_fitted:
        coef["debt_gdp"], tstat["debt_gdp"] = DEBT_COEF, float("nan")

    a, r = _refit_intercept(p, coef, est)
    yy = p["tp"].values
    post = np.asarray(p.index >= "2000-01-01")
    return DriverModel(
        intercept=a, coef=coef, tstat=tstat,
        r2=float(1 - (r @ r) / ((yy - yy.mean()) ** 2).sum()),
        r2_post2000=float(1 - (r[post] ** 2).sum()
                          / ((yy[post] - yy[post].mean()) ** 2).sum()),
        rmse=float(np.sqrt((r ** 2).mean())), n=len(p), drivers=DRIVERS,
        country=country, borrowed=tuple(borrowed),
        driver_sd={k: float(X_[k].std()) for k in DRIVERS},
        debt_fitted=debt_fitted, debt_base=debt_base)


def fit_all(markets: tuple = MARKETS, panels: dict | None = None) -> dict:
    """Every market's production fit, the US first so the others can borrow from it."""
    panels = panels or {c: build_panel(c) for c in markets}
    us = fit_market("US", panels["US"])
    return {c: (us if c == "US" else fit_market(c, panels[c], reference=us))
            for c in markets}


def _report(m: DriverModel, p: pd.DataFrame) -> None:
    print(f"\n{m.country} 10y term premium, n={m.n}, "
          f"{p.index.min().date()}..{p.index.max().date()}")
    print(f"  R2 {m.r2:.3f}   post-2000 {m.r2_post2000:+.3f}   RMSE {m.rmse:.2f}pp")
    for k in m.drivers:
        tag = ("  calibrated" if (k == "debt_gdp" and not m.debt_fitted) else
               "  from US   " if k in m.borrowed else
               f"  (t={m.tstat[k]:+5.1f})")
        print(f"  {k:22} {m.coef[k]:+.4f}{tag}   today {p[k].iloc[-1]:7.2f}")
    tp0 = float(p["tp"].iloc[-1])
    today = {k: float(p[k].iloc[-1]) for k in m.drivers}
    print(f"  actual {tp0:+.2f}%   model {m.premium(**today):+.2f}%   "
          f"residual {tp0 - m.premium(**today):+.2f}pp (held constant in scenarios)")


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    us = build_panel("US")
    split = fit(us, drivers=US_DRIVERS, country="US")
    print("US, Fed and foreign official split out separately (the diagnostic fit):")
    _report(split, us)
    print("\n" + "=" * 72)
    print("Production fit: three estimated coefficients, debt calibrated at "
          f"{DEBT_COEF * 100:.1f}bp per point of debt/GDP")
    panels = {c: build_panel(c) for c in MARKETS}
    for c, m in fit_all(MARKETS, panels).items():
        _report(m, panels[c])


if __name__ == "__main__":
    _main()
