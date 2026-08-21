"""Test the stock-bond correlation channel, then run the full scenario chain.

Run: ``python -m validation.sbc_macro``

Two things happen here.

1. **A falsifiable test.** The stock-bond correlation is observable, so the macro ->
   correlation link can be checked directly rather than assumed. We build the realised
   correlation from equity prices and 10y yields, build inflation and growth volatility
   from core CPI and industrial production, and ask whether the AQR sign structure holds
   and how much it explains.

2. **The chain.** Scenario terminal inflation -> (Okun) inflation uncertainty ->
   (ratio) stock-bond correlation. That is the whole point of routing through an
   observable: each link is separately testable, so a miss can be attributed.

Findings in ``validation/FINDINGS_sbc.md``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.okun import DEFAULT_WINDOW, fit_hockey_stick, rolling_level_vol
from model.stock_bond import fit_sbc, realised_sbc

logger = logging.getLogger(__name__)

MARKETS = ["US", "DE", "JP", "UK", "CH", "CA"]


def build_panel(window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Annual panel of (realised sbc, sigma_pi, sigma_g, cov(g,pi)) per market."""
    from macro.fetch_core_cpi import core_inflation_yoy
    from macro.fetch_macro import growth_yoy_monthly, load_share_prices

    eq = load_share_prices()
    infl = core_inflation_yoy()
    grow = growth_yoy_monthly()

    # History-extended: Germany's 10y comes from the monthly Bundesbank Svensson
    # parameters before 1997, which is what lets its correlation reach back past the
    # regime change (see macro.tp_history.long_yield_monthly).
    from macro.tp_history import long_yield_monthly
    y10m = long_yield_monthly(10.0)

    frames = []
    for c in MARKETS:
        if c not in eq.columns or c not in y10m.columns:
            continue
        sbc = realised_sbc(eq[c], y10m[c], window=window).rename("sbc")
        j = pd.DataFrame({"g": grow[c], "pi": infl[c]}).dropna()
        if len(j) < window + 24:
            continue
        m = pd.DataFrame({
            "sigma_g": j["g"].rolling(window).std(),
            "sigma_pi": j["pi"].rolling(window).std(),
            "cov_gpi": j["g"].rolling(window).cov(j["pi"]),
        }).dropna()
        out = pd.concat([sbc, m], axis=1, join="inner").reset_index()
        out.columns = ["date"] + list(out.columns[1:])
        out["country"] = c
        frames.append(out)
    panel = pd.concat(frames, ignore_index=True)
    return panel[panel["date"].dt.month == 12].reset_index(drop=True)   # annual


def _ols(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, 1.0 - (r @ r) / ((y - y.mean()) ** 2).sum()


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    P = build_panel()
    print(f"panel: n={len(P)} annual obs, markets={sorted(P.country.unique())}, "
          f"{P.date.min().date()}..{P.date.max().date()}\n")

    print("=== 1. AQR STRUCTURAL FORM  sbc ~ sigma_pi^2 + sigma_g^2 + cov(g,pi) ===")
    print("    prediction: inflation-vol coefficient POSITIVE, growth-vol NEGATIVE")
    for lab, q in [("POOLED", P)] + [(c, P[P.country == c]) for c in sorted(P.country.unique())]:
        if len(q) < 15:
            continue
        b, r2 = _ols(q.sbc.values, [(q.sigma_pi ** 2).values, (q.sigma_g ** 2).values,
                                    q.cov_gpi.values])
        flag = "" if (b[1] > 0 and b[2] < 0) else "   <-- sign(s) against the model"
        print(f"  {lab:7} n={len(q):3}  pi_vol={b[1]:+.4f}  g_vol={b[2]:+.5f}  "
              f"cov={b[3]:+.4f}  R2={r2:.3f}{flag}")

    print("\n=== 2. RATIO FORM  sbc ~ sigma_pi / sigma_g  (one variable) ===")
    for lab, q in [("POOLED", P)] + [(c, P[P.country == c]) for c in sorted(P.country.unique())]:
        if len(q) < 15:
            continue
        f = fit_sbc(q.sbc.values, q.sigma_pi.values, q.sigma_g.values)
        print(f"  {lab:7} n={f.n:3}  intercept={f.intercept:+.3f}  slope={f.slope:+.3f}  R2={f.r2:.3f}")

    pooled = fit_sbc(P.sbc.values, P.sigma_pi.values, P.sigma_g.values)

    print("\n=== 3. ERA STABILITY (pooled ratio form) ===")
    for lab, q in (("<=2000", P[P.date.dt.year <= 2000]), ("2001+", P[P.date.dt.year > 2000])):
        f = fit_sbc(q.sbc.values, q.sigma_pi.values, q.sigma_g.values)
        print(f"  {lab:7}: intercept={f.intercept:+.3f}  slope={f.slope:+.3f}  R2={f.r2:.3f}  n={f.n}")

    print("\n=== 4. FULL CHAIN: terminal core inflation -> uncertainty -> correlation ===")
    from validation.okun_mirage import annual, build_panel as okun_panel
    A = annual(okun_panel())
    okun = fit_hockey_stick(A["level"], A["vol"])
    sigma_g_typ = float(P.sigma_g.median())
    print(f"    Okun: vol = {okun.floor:.2f} + {okun.slope:.2f}*max(level-{okun.kink:.2f},0)")
    print(f"    growth vol held at the panel median, {sigma_g_typ:.2f}pp\n")
    print("    terminal core   inflation      sigma_pi/     implied")
    print("    inflation %     uncertainty    sigma_g       stock-bond corr")
    for lv in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        sp = okun.sigma(lv)
        print(f"      {lv:4.1f}          {sp:5.2f}pp        {sp / sigma_g_typ:5.2f}        "
              f"{pooled.predict(sp, sigma_g_typ):+.2f}")


if __name__ == "__main__":
    _main()
