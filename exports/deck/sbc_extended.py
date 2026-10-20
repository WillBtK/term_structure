"""Core inflation vs realised stock-bond correlation, four markets, on the LONGEST
month-end 10y yield histories available -- not the daily curves the deck uses, which
start 1971 (US 10y), 1979 (UK), 1986 (JP 10y) and 1997 (DE):

    US  FRED GS10, constant-maturity 10y, monthly, 1953+
    UK  BoE GLC nominal spot 10y, month-end, 1970+
    DE  Bundesbank Svensson 10y, month-end, 1972-09+ (BBSIS monthly series)
    JP  MoF JGB 10y, month-end; the 9y before 10y exists (to 1986-07), 1974-09+

The correlation is the trailing 120-month correlation of monthly equity returns and
bond returns (``model.stock_bond.realised_sbc``, duration 8), read each December; the
inflation level is that calendar year's average of year-on-year core CPI. No growth
data is required, so the panel runs to 2025.

    python -m exports.deck.sbc_extended
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "3_core_inflation_vs_stock_bond_corr_extended.csv"
BBK = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
       "M.I.ZST.ZI.EUR.S1311.B.A604.R10XX.R.A.A._Z._Z.A?format=csv")


def yields() -> pd.DataFrame:
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10", timeout=90)
    x = pd.read_csv(io.StringIO(r.text))
    us = pd.Series(pd.to_numeric(x.iloc[:, 1], errors="coerce").values,
                   index=pd.to_datetime(x.iloc[:, 0]))
    from uk.fetch_boe import load_curve
    uk = load_curve("nominal", freq="M")[10.0] * 100
    rows = [l.split(";") for l in requests.get(BBK, timeout=180).text.splitlines() if l[:4].isdigit()]
    de = pd.Series({pd.Period(a, "M").to_timestamp(): float(b.replace(",", "."))
                    for a, b, *_ in rows if b not in (".", "")})
    from japan.fetch_mof import load_yields
    jp = load_yields(refresh=True)
    jp = jp[10.0].combine_first(jp[9.0])
    out = {}
    for c, s in (("US", us), ("UK", uk), ("DE", de), ("JP", jp)):
        s = s.dropna()
        out[c] = s.groupby(s.index.to_period("M")).last()
    y = pd.DataFrame(out)
    y.index = y.index.to_timestamp()
    return y


def panel(last_year: int = 2025) -> pd.DataFrame:
    from macro.fetch_core_cpi import core_inflation_yoy
    from macro.fetch_macro import load_share_prices
    from model.stock_bond import realised_sbc
    y = yields()
    eq = load_share_prices()
    eq.index = pd.DatetimeIndex(eq.index).to_period("M").to_timestamp()
    lev = core_inflation_yoy().resample("YS").mean()
    rows = []
    for c in ("US", "UK", "DE", "JP"):
        sbc = realised_sbc(eq[c].dropna(), y[c].dropna(), window=120)
        for t, v in sbc[sbc.index.month == 12].items():
            yr = pd.Timestamp(t.year, 1, 1)
            if t.year <= last_year and yr in lev.index and np.isfinite(lev.at[yr, c]):
                rows.append({"country": c, "year": t.year, "pi": float(lev.at[yr, c]),
                             "sbc": float(v)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from validation.sbc_inflation_level import fit
    p = panel()
    f = fit(p)
    out = p.rename(columns={"pi": "core_inflation_pct", "sbc": "stock_bond_corr"}).round(3)
    out["fitted_corr"] = (f["slope"] * out["core_inflation_pct"] + f["intercept"]).round(3)
    out.to_csv(OUT, index=False)
    print({k: v for k, v in f.items() if k != "per_market"})
