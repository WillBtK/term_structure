"""Is the fiscal FLOW a better term-premium driver than the fiscal STOCK?

The production models use debt/GDP. The case for a flow measure instead is Laubach
(2009) -- who finds roughly 25bp per point of projected deficit against 3-4bp per point
of debt -- and the mechanism: preferred habitat (Vayanos-Vila, Greenwood-Vayanos) says
what moves the premium is the duration the price-sensitive market must ABSORB, which is
an issuance flow, not an outstanding stock. It is the same stock-versus-flow question
the QE work raised, transplanted to the fiscal side.

Three candidates, all as a percent of GDP:

    debt_gdp   the stock                        (what the models use today)
    deficit    general government net borrowing (the literal reading of the hypothesis)
    d_debt     the twelve-month change in the stock

The third is the one worth watching. It is a flow, but not the accrual deficit: it is
the change in outstanding debt, which differs through valuation, off-budget financing
and cash-versus-accrual timing. It is also the flow preferred-habitat theory actually
names -- the paper that has to be absorbed -- and, practically, it needs no new data
source and inherits the committed WEO/national projections for free, whereas a deficit
path would need its own forecast.

Scored on real-time refits at the tool's three-year horizon, each market in its
PRODUCTION specification -- the US standalone, the others as a US-relative spread --
with the residual decaying at the production rho.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from model.tp_drivers import (MARKETS, RESIDUAL_PERSISTENCE, TARGET_KINK, build_panel)

logger = logging.getLogger(__name__)

# General government net lending/borrowing, percent of GDP. The US uses the federal
# balance, which reaches back to 1929 rather than 1980.
DEFICIT_SERIES = {"US": "FYFSGDA188S", "UK": "GGNLBAGBA188N",
                  "DE": "GGNLBADEA188N", "JP": "GGNLBAJPA188N"}
CORE = ["ei", "inflation_uncertainty", "price_insensitive"]
CORE_SIGNS = [1, 1, -1]
SPECS = {"debt": ["debt_gdp"], "deficit": ["deficit"], "d_debt": ["d_debt"],
         "both": ["debt_gdp", "d_debt"]}
HORIZON = 36


def _annual_to_monthly(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    s = s.copy()
    s.index = pd.DatetimeIndex([pd.Timestamp(int(t.year), 1, 1) for t in s.index])
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def market_panel(country: str, panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """The driver panel plus both fiscal flow candidates."""
    from validation.fred_api import fred_series
    p = (build_panel(country) if panel is None else panel).copy()
    p["ei"] = np.maximum(p["expected_inflation"] - TARGET_KINK, 0.0)
    # Sign-flipped so a bigger deficit is a bigger positive number, like debt.
    p["deficit"] = -_annual_to_monthly(
        fred_series(DEFICIT_SERIES[country]).dropna(), p.index)
    p["d_debt"] = p["debt_gdp"].diff(12)
    return p


def _fit(train: pd.DataFrame, cols: list, signs: list, relative: bool) -> np.ndarray:
    base = [np.ones(len(train))] + ([train["us_tp"].values] if relative else [])
    A = np.column_stack(base + [train[c].values for c in cols])
    lo = np.array([-np.inf] + ([0.0] if relative else [])
                  + [0.0 if s > 0 else -np.inf for s in signs])
    hi = np.array([np.inf] + ([1.5] if relative else [])
                  + [np.inf if s > 0 else 0.0 for s in signs])
    return lsq_linear(A, train["tp"].values, bounds=(lo, hi)).x


def _value(b: np.ndarray, q: pd.DataFrame, cols: list, i: int, relative: bool) -> float:
    k = 1 if relative else 0
    out = b[0] + (b[1] * q["us_tp"].iloc[i] if relative else 0.0)
    return out + sum(b[j + 1 + k] * q[c].iloc[i] for j, c in enumerate(cols))


def _frame(country: str, extra: list, panels: dict) -> tuple:
    """(aligned frame, column names, is_relative) in the production specification."""
    if country == "US":
        cols = CORE + extra
        return panels["US"][cols + ["tp"]].dropna(), cols, False
    keep = CORE + ["debt_gdp", "deficit", "d_debt", "tp"]
    j = panels[country].join(panels["US"][keep].add_prefix("us_"), how="inner")
    for k in CORE + extra:
        j["rel_" + k] = j[k] - j["us_" + k]
    cols = ["rel_" + k for k in CORE + extra]
    return j[cols + ["us_tp", "tp"]].dropna(), cols, True


def horse_race(panels: dict | None = None, horizon: int = HORIZON,
               rho: float = RESIDUAL_PERSISTENCE) -> pd.DataFrame:
    """Out-of-sample RMSE by fiscal specification, real-time annual refits."""
    warnings.filterwarnings("ignore")
    panels = panels or {c: market_panel(c) for c in MARKETS}
    rows = []
    for c in MARKETS:
        row = {"market": c}
        for label, extra in SPECS.items():
            q, cols, rel = _frame(c, extra, panels)
            signs = CORE_SIGNS + [1] * len(extra)
            errs = []
            for t0 in pd.date_range(q.index[0] + pd.DateOffset(years=8),
                                    q.index[-1] - pd.DateOffset(months=horizon),
                                    freq="12MS"):
                tr = q.loc[:t0]
                if len(tr) < 80:
                    continue
                i0 = q.index.get_indexer([t0], method="pad")[0]
                i1 = i0 + horizon
                if i1 >= len(q):
                    break
                try:
                    b = _fit(tr, cols, signs, rel)
                except Exception:                                   # noqa: BLE001
                    continue
                res0 = q["tp"].iloc[i0] - _value(b, q, cols, i0, rel)
                errs.append(_value(b, q, cols, i1, rel)
                            + rho ** (horizon / 12) * res0 - q["tp"].iloc[i1])
            row[label] = (float(np.sqrt(np.mean(np.array(errs) ** 2)))
                          if errs else float("nan"))
            row["n"] = len(errs)
        rows.append(row)
    return pd.DataFrame(rows).set_index("market")


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    panels = {c: market_panel(c) for c in MARKETS}
    r = horse_race(panels)
    print("Out-of-sample RMSE (pp) at the 3-year horizon, production spec per market:\n")
    print(r.round(2).to_string())
    cols = [c for c in r.columns if c != "n"]
    print("\nbest per market: "
          + ", ".join(f"{m} {r.loc[m, cols].idxmin()}" for m in r.index))


if __name__ == "__main__":
    _main()
