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

logger = logging.getLogger(__name__)

TARGET_KINK = 2.0
DRIVERS = ("expected_inflation", "inflation_uncertainty",
           "foreign_official", "fed_holdings", "debt_gdp")

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

    def premium(self, expected_inflation, inflation_uncertainty,
                foreign_official, fed_holdings, debt_gdp) -> float:
        """Modelled term premium in percent."""
        x = {"expected_inflation": max(float(expected_inflation) - TARGET_KINK, 0.0),
             "inflation_uncertainty": float(inflation_uncertainty),
             "foreign_official": float(foreign_official),
             "fed_holdings": float(fed_holdings),
             "debt_gdp": float(debt_gdp)}
        return self.intercept + sum(self.coef[k] * x[k] for k in DRIVERS)

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


def build_panel() -> pd.DataFrame:
    """Monthly panel of the term premium and its five drivers."""
    from macro.fetch_core_cpi import core_inflation_yoy
    from macro.inflation_expectations import expectations
    from macro.tp_history import spliced_tp
    from validation.fred_api import fred_series

    d = pd.DataFrame({"tp": spliced_tp(10.0)["US"],
                      "expected_inflation": expectations("US")}).dropna()
    gdp = fred_series(SERIES["gdp"]).reindex(d.index).ffill() * 1000     # $bn -> $mn
    d["foreign_official"] = fred_series(SERIES["foreign_official"]).reindex(
        d.index).ffill() / gdp * 100
    d["foreign_private"] = fred_series(SERIES["foreign_private"]).reindex(
        d.index).ffill() / gdp * 100
    d["fed_holdings"] = fred_series(SERIES["fed_holdings"]).reindex(d.index).ffill()
    pub = fred_series(SERIES["debt_public"])
    d["debt_gdp"] = (pub / (fred_series(SERIES["gdp"]).reindex(pub.index).ffill() * 1000)
                     * 100).reindex(d.index).ffill()
    d["inflation_uncertainty"] = core_inflation_yoy()["US"].rolling(60).std().reindex(
        d.index).ffill()
    return d.dropna()


def fit(panel: pd.DataFrame | None = None) -> DriverModel:
    p = build_panel() if panel is None else panel
    X = np.column_stack([
        np.ones(len(p)),
        np.maximum(p["expected_inflation"].values - TARGET_KINK, 0.0),
        *[p[k].values for k in DRIVERS[1:]]])
    y = p["tp"].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (r @ r) / max(len(p) - X.shape[1], 1))
    post = np.asarray(p.index >= "2000-01-01")
    return DriverModel(
        intercept=float(b[0]),
        coef={k: float(b[i + 1]) for i, k in enumerate(DRIVERS)},
        tstat={k: float(b[i + 1] / se[i + 1]) for i, k in enumerate(DRIVERS)},
        r2=float(1 - (r @ r) / ((y - y.mean()) ** 2).sum()),
        r2_post2000=float(1 - (r[post] ** 2).sum()
                          / ((y[post] - y[post].mean()) ** 2).sum()),
        rmse=float(np.sqrt((r ** 2).mean())), n=len(p))


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = build_panel()
    m = fit(p)
    print(f"US 10y term premium, n={m.n}, {p.index.min().date()}..{p.index.max().date()}")
    print(f"  R2 {m.r2:.3f}   post-2000 {m.r2_post2000:+.3f}   RMSE {m.rmse:.2f}pp\n")
    for k in DRIVERS:
        print(f"  {k:22} {m.coef[k]:+.4f}  (t={m.tstat[k]:+5.1f})   today "
              f"{p[k].iloc[-1]:6.2f}")
    tp0 = float(p["tp"].iloc[-1])
    today = {k: float(p[k].iloc[-1]) for k in DRIVERS}
    print(f"\n  actual {tp0:+.2f}%   model {m.premium(**today):+.2f}%   "
          f"residual {tp0 - m.premium(**today):+.2f}pp (held constant in scenarios)")


if __name__ == "__main__":
    _main()
