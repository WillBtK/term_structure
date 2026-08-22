"""Out-of-sample discipline for the scenario driver models.

Two tests, both answering the question an investment committee will actually ask:
"if you had built this earlier, would it have worked?"

1. FROZEN-2015: fit every model on data through 2015 only, then drive it with the
   REALIZED drivers for the following decade and compare the anchored projection with
   what happened. This spans peak QE, YCC, COVID and the 2022 repricing -- the hardest
   window available.

2. ROLLING 3-YEAR: refit annually on an expanding window and project 36 months ahead,
   scored against the random walk (premium unchanged). Three years is the tool's
   primary forecast horizon, and no-change is the only benchmark that matters --
   an anchored model that cannot beat it adds nothing unconditionally.

What the numbers mean -- and what they do not. The tool is a CONDITIONAL mapping:
the user supplies the macro scenario and the model prices it. Matching the random
walk unconditionally (US 0.98, UK 0.95) is the expected result, not a failure;
nothing in the bond-risk-premium literature beats no-change at three years
(Thornton-Valente 2012). The anchoring design is what makes this safe: an unchanged
scenario returns today's premium exactly, so the model contributes no drift of its
own -- only the translation of deliberate scenario deviations, at sensitivities that
match the event-study literature (QE -6.5bp per percent of GDP vs Ihrig et al's
4-8bp; fiscal 1.4-2.5bp per debt point vs Laubach's 3-4bp).

Germany and Japan fail the frozen test for a reason worth keeping on the record:
frozen at 2015, Germany's fit had seen ZERO months of PSPP (it began March 2015) and
Japan's had never seen BoJ holdings above ~30% of GDP, so both extrapolated
coefficients into driver ranges outside their samples. Japan's random-walk RMSE of
0.50pp is yield-curve control itself -- no-change was enforced policy, unbeatable and
uninformative. Today's fits contain a complete QE expansion and contraction, which is
precisely the variation those coefficients needed.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from model.tp_drivers import MARKETS, build_panel, design, fit_market
from model.tp_relative import SPREAD_MARKETS, _hat, build_pair, fit_relative

logger = logging.getLogger(__name__)

FREEZE = "2015-12-31"
HORIZON = 36


def _us_fitted(m, xrow) -> float:
    out = m.intercept + sum(m.coef[k] * xrow[k] for k in m.drivers
                            if m.debt_fitted or k != "debt_gdp")
    if not m.debt_fitted and "debt_gdp" in m.coef:
        out += m.coef["debt_gdp"] * (xrow["debt_gdp"] - m.debt_base)
    for k, g in m.adjust.items():
        # measured-from-base channels: the backtest hands them the realised driver,
        # exactly as a scenario would
        if k in xrow.index and pd.notna(xrow[k]):
            out += g * (xrow[k] - m.base[k])
    return out


def _spread_fitted(m, pair: pd.DataFrame, i: int) -> float:
    out = m.intercept + m.beta_us * pair["us_tp"].iloc[i]
    for k, g in m.coef.items():
        out += g * pair[f"rel_{k}"].iloc[i]
    for k, g in m.adjust.items():
        out += g * (pair[f"rel_{k}"].iloc[i] - m.base[k])
    return out


def frozen_decade(panels: dict | None = None, cut: str = FREEZE) -> pd.DataFrame:
    """Fit through ``cut``, project the rest of the sample on realized drivers."""
    panels = panels or {c: build_panel(c) for c in MARKETS}
    rows = []
    us = panels["US"]
    m_us = fit_market("US", us.loc[:cut])
    x = design(us)
    f = pd.Series([_us_fitted(m_us, x.iloc[i]) for i in range(len(x))], index=us.index)
    a = us.loc[:cut].index[-1]
    proj = us["tp"].loc[a] + (f - f.loc[a])
    oos = us.index > cut
    e = proj[oos] - us["tp"][oos]
    rows.append({"market": "US", "rmse": float(np.sqrt((e ** 2).mean())),
                 "end_error": float(e.iloc[-1])})
    for c in SPREAD_MARKETS:
        p = build_pair(c, panels)
        tr = p.loc[:cut]
        m = fit_relative(c, tr, m_us)
        fs = pd.Series([_spread_fitted(m, p, i) for i in range(len(p))], index=p.index)
        a = tr.index[-1]
        proj = p["tp"].loc[a] + (fs - fs.loc[a])
        oos = p.index > cut
        e = proj[oos] - p["tp"][oos]
        rows.append({"market": c, "rmse": float(np.sqrt((e ** 2).mean())),
                     "end_error": float(e.iloc[-1])})
    return pd.DataFrame(rows).set_index("market")


def residual_persistence_grid(panels: dict | None = None,
                              horizons=(36, 120),
                              rhos=(1.0, 0.9, 0.8, 0.7, 0.5, 0.25, 0.0)) -> pd.DataFrame:
    """How much of today's UNEXPLAINED premium should survive to the horizon?

    Full anchoring says all of it, which asserts every shock is permanent. Zero says
    none, which asserts the premium reverts to the model's fair value -- for the US
    today, that the ten-year yield is 71bp rich. This scores the whole range out of
    sample on real-time refits, so the residual is the one a user would actually have
    seen rather than an in-sample artefact. (Estimated in sample the residual looks
    far more mean-reverting than it is: the regression forces it orthogonal to the
    drivers over the whole span, so a full-sample AR(1) puts the US half-life at four
    months. That number is not usable and is why this test exists.)
    """
    warnings.filterwarnings("ignore")
    panels = panels or {c: build_panel(c) for c in MARKETS}
    pairs = {c: build_pair(c, panels) for c in MARKETS if c != "US"}
    rows = []
    for horizon in horizons:
        for c in MARKETS:
            p = panels["US"] if c == "US" else pairs[c]
            x = design(panels["US"]) if c == "US" else None
            start = "2004-12-01" if c == "US" else "2009-12-01"
            errs = {r: [] for r in rhos}
            for t0 in pd.date_range(start, p.index[-1] - pd.DateOffset(months=horizon),
                                    freq="12MS"):
                tr = p.loc[:t0]
                if len(tr) < 100:
                    continue
                try:
                    if c == "US":
                        m = fit_market("US", tr)
                        fit = pd.Series([_us_fitted(m, x.iloc[i]) for i in range(len(x))],
                                        index=p.index)
                    else:
                        m = fit_relative(c, tr, fit_market("US", panels["US"].loc[:t0]))
                        fit = pd.Series(_hat(p, m.coef, m.beta_us, m.intercept),
                                        index=p.index)
                except Exception:                               # noqa: BLE001
                    continue
                i0 = p.index.get_indexer([t0], method="pad")[0]
                i1 = i0 + horizon
                if i1 >= len(p):
                    break
                res0 = p["tp"].iloc[i0] - fit.iloc[i0]
                for r in rhos:
                    errs[r].append(fit.iloc[i1] + r ** (horizon / 12) * res0
                                   - p["tp"].iloc[i1])
            if not errs[rhos[0]]:
                continue
            row = {"market": c, "horizon_y": horizon // 12, "n": len(errs[rhos[0]])}
            row.update({f"rho={r}": float(np.sqrt(np.mean(np.array(errs[r]) ** 2)))
                        for r in rhos})
            rows.append(row)
    return pd.DataFrame(rows).set_index(["horizon_y", "market"])


def rolling_vs_random_walk(panels: dict | None = None,
                           horizon: int = HORIZON) -> pd.DataFrame:
    """Expanding-window annual refits, ``horizon`` months ahead, vs no-change."""
    warnings.filterwarnings("ignore")
    panels = panels or {c: build_panel(c) for c in MARKETS}
    out = []

    def score(market, errs, rws):
        e, r = np.array(errs), np.array(rws)
        out.append({"market": market, "n": len(e),
                    "model_rmse": float(np.sqrt((e ** 2).mean())),
                    "rw_rmse": float(np.sqrt((r ** 2).mean())),
                    "ratio": float(np.sqrt((e ** 2).mean()) / np.sqrt((r ** 2).mean()))})

    us = panels["US"]
    x = design(us)
    errs, rws = [], []
    for t0 in pd.date_range("2004-12-01", us.index[-1] - pd.DateOffset(months=horizon),
                            freq="12MS"):
        tr = us.loc[:t0]
        if len(tr) < 120:
            continue
        m = fit_market("US", tr)
        i0 = us.index.get_indexer([t0], method="pad")[0]
        i1 = i0 + horizon
        if i1 >= len(us):
            break
        proj = us["tp"].iloc[i0] + (_us_fitted(m, x.iloc[i1]) - _us_fitted(m, x.iloc[i0]))
        errs.append(proj - us["tp"].iloc[i1])
        rws.append(us["tp"].iloc[i0] - us["tp"].iloc[i1])
    score("US", errs, rws)

    for c in SPREAD_MARKETS:
        p = build_pair(c, panels)
        errs, rws = [], []
        for t0 in pd.date_range("2009-12-01",
                                p.index[-1] - pd.DateOffset(months=horizon), freq="12MS"):
            tr = p.loc[:t0]
            if len(tr) < 100:
                continue
            try:
                m = fit_relative(c, tr, fit_market("US", us.loc[:t0]))
            except Exception:                                       # noqa: BLE001
                continue
            i0 = p.index.get_indexer([t0], method="pad")[0]
            i1 = i0 + horizon
            if i1 >= len(p):
                break
            proj = p["tp"].iloc[i0] + (_spread_fitted(m, p, i1) - _spread_fitted(m, p, i0))
            errs.append(proj - p["tp"].iloc[i1])
            rws.append(p["tp"].iloc[i0] - p["tp"].iloc[i1])
        if errs:
            score(c, errs, rws)
    return pd.DataFrame(out).set_index("market")


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    panels = {c: build_panel(c) for c in MARKETS}
    print(f"Frozen-{FREEZE[:4]} decade, anchored projection on realized drivers:")
    print(frozen_decade(panels).round(2).to_string())
    print(f"\nRolling {HORIZON}m-ahead, annual refits, vs random walk "
          "(ratio < 1 beats no-change):")
    print(rolling_vs_random_walk(panels).round(2).to_string())
    print("\nResidual persistence: RMSE by how much of today's unexplained premium "
          "survives per year")
    g = residual_persistence_grid(panels)
    print(g.round(2).to_string())
    cols = [c for c in g.columns if c.startswith("rho=")]
    print("\nmean across cells:  " + "  ".join(f"{c} {g[c].mean():.3f}" for c in cols))
    print("worst cell:         " + "  ".join(f"{c} {g[c].max():.3f}" for c in cols))


if __name__ == "__main__":
    _main()
