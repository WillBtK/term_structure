"""Core inflation vs realised stock-bond correlation, four markets, on the LONGEST
month-end 10y yield histories available -- not the daily curves the deck uses, which
start 1971 (US 10y), 1979 (UK), 1986 (JP 10y) and 1997 (DE):

    US  FRED GS10, constant-maturity 10y, monthly, 1953+
    UK  BoE GLC nominal spot 10y, month-end, 1970+
    DE  Bundesbank Svensson 10y, month-end, 1972-09+ (BBSIS monthly series)
    JP  MoF JGB 10y, month-end; the 9y before 10y exists (to 1986-07), 1974-09+

The correlation is the trailing 120-month correlation of monthly equity returns and
bond returns (``model.stock_bond.realised_sbc``, duration 8), read each December. The
inflation level is the average of year-on-year core CPI over the SAME trailing 120
months (``model.okun.rolling_level_vol``), and its standard deviation is carried too:
a single year's inflation against a ten-year correlation halves the fit (R2 0.29 vs
0.56). No growth data is required, so the panel runs to 2025.

Two fits of the correlation on the ten-year average: a line, and an S-curve
(logistic: a hedging floor, a transition, a ceiling), which fits markedly better.

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
    from model.okun import rolling_level_vol
    from model.stock_bond import realised_sbc
    y = yields()
    eq = load_share_prices()
    eq.index = pd.DatetimeIndex(eq.index).to_period("M").to_timestamp()
    ci = core_inflation_yoy()
    rows = []
    for c in ("US", "UK", "DE", "JP"):
        lv = rolling_level_vol(ci[c], 120)
        sbc = realised_sbc(eq[c].dropna(), y[c].dropna(), window=120)
        for t, v in sbc[sbc.index.month == 12].items():
            m = pd.Timestamp(t.year, 12, 1)
            if t.year <= last_year and m in lv.index:
                rows.append({"country": c, "year": t.year, "pi": float(lv.at[m, "level"]),
                             "vol": float(lv.at[m, "vol"]), "sbc": float(v)})
    return pd.DataFrame(rows)


def s_curve(x, lo, hi, mid, scale):
    return lo + (hi - lo) / (1.0 + np.exp(-(np.asarray(x) - mid) / scale))


def fit_s_curve(p: pd.DataFrame, n_boot: int = 0, seed: int = 0) -> dict:
    """Logistic fit of the correlation on the ten-year average; with ``n_boot``, a 90%
    bootstrap band for where it crosses zero."""
    from scipy.optimize import brentq, curve_fit
    x, y = p["pi"].to_numpy(), p["sbc"].to_numpy()

    def one(xx, yy):
        q, _ = curve_fit(s_curve, xx, yy, p0=[-0.3, 0.35, 2.5, 0.7], maxfev=20000)
        return q, brentq(lambda z: s_curve(z, *q), xx.min(), xx.max())
    q, cross = one(x, y)
    r2 = 1 - ((y - s_curve(x, *q)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = {"lo": q[0], "hi": q[1], "mid": q[2], "scale": q[3], "r2": r2, "cross": cross}
    if n_boot:
        rng, draws = np.random.default_rng(seed), []
        for _ in range(n_boot):
            i = rng.integers(0, len(x), len(x))
            try:
                draws.append(one(x[i], y[i])[1])
            except Exception:                                       # noqa: BLE001
                continue
        out["cross_lo"], out["cross_hi"] = (float(v) for v in np.percentile(draws, [5, 95]))
    return out


def block_cv(p: pd.DataFrame, years: int = 5) -> dict:
    """Out-of-sample error, leaving out each block of ``years`` in turn: line vs S-curve."""
    x, y, b = p["pi"].to_numpy(), p["sbc"].to_numpy(), p["year"].to_numpy() // years
    el, es = [], []
    for k in np.unique(b):
        tr, te = b != k, b == k
        sl, ic = np.polyfit(x[tr], y[tr], 1)
        el += list(y[te] - (ic + sl * x[te]))
        f = fit_s_curve(p[tr])
        es += list(y[te] - s_curve(x[te], f["lo"], f["hi"], f["mid"], f["scale"]))
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    return {"linear": rm(el), "s_curve": rm(es)}


if __name__ == "__main__":
    from validation.sbc_inflation_level import fit
    p = panel()
    f, sc = fit(p), fit_s_curve(p)
    out = p.rename(columns={"pi": "core_inflation_10y_avg_pct", "vol": "core_inflation_10y_vol_pp",
                            "sbc": "stock_bond_corr_10y"}).round(3)
    out["fitted_linear"] = (f["slope"] * p["pi"] + f["intercept"]).round(3)
    out["fitted_s_curve"] = s_curve(p["pi"], sc["lo"], sc["hi"], sc["mid"], sc["scale"]).round(3)
    out.to_csv(OUT, index=False)
    print("linear", {k: v for k, v in f.items() if k != "per_market"})
    print("s-curve", sc)
