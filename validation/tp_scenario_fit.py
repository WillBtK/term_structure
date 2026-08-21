"""Fit the final link (correlation -> term premium) and demonstrate the whole chain.

Run: ``python -m validation.tp_scenario_fit``

Findings in ``validation/FINDINGS_tp_scenario.md``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.okun import fit_hockey_stick
from model.stock_bond import fit_sbc, realised_sbc
from model.tp_scenario import (FairValueParams, fit_error_correction, fit_fair_value,
                               scenario_term_premium)

logger = logging.getLogger(__name__)

# Markets whose samples span the 1990s-2000s regime change. DE and UK enter only from
# 2007 (Bund daily curve from 1997, UK industrial production from 1997), so their samples
# sit entirely inside one correlation regime and carry no information about the shift the
# model is built to capture. They are reported but excluded from the fitted parameters.
LONG_SAMPLE = ["US", "CA", "JP", "CH"]
WINDOW = 120


def build_panel() -> pd.DataFrame:
    from macro.fetch_core_cpi import core_inflation_yoy
    from macro.fetch_macro import growth_yoy_monthly, load_ind_prod, load_share_prices

    eq, infl, grow = load_share_prices(), core_inflation_yoy(), growth_yoy_monthly()
    d = pd.read_parquet("viz/data_daily.parquet")
    nom = d[d.instrument == "nominal"]
    y10 = (nom[nom.tenor == 10.0].pivot_table(index="date", columns="country",
                                              values="observed").resample("MS").last() * 100)
    tp = {t: (nom[nom.tenor == t].pivot_table(index="date", columns="country",
                                              values="term_premium").resample("MS").last() * 100)
          for t in (3.0, 10.0)}
    frames = []
    for c in ["US", "DE", "JP", "UK", "CH", "CA"]:
        sbc = realised_sbc(eq[c], y10[c], window=WINDOW).rename("sbc")
        j = pd.DataFrame({"g": grow[c], "pi": infl[c]}).dropna()
        m = pd.DataFrame({"sigma_g": j["g"].rolling(WINDOW).std(),
                          "sigma_pi": j["pi"].rolling(WINDOW).std()}).dropna()
        ip = np.log(load_ind_prod()[c].dropna()) * 100
        gap = ((ip - ip.rolling(40, min_periods=20).mean())
               .resample("MS").interpolate(limit_area="inside").rename("gap"))
        x = pd.concat([sbc, m, gap], axis=1, join="inner")
        for t in (3.0, 10.0):
            x[f"tp{int(t)}"] = tp[t][c]
        x = x.dropna()
        x["country"] = c
        frames.append(x.reset_index().rename(columns={"index": "date"}))
    P = pd.concat(frames, ignore_index=True)
    return P[P["date"].dt.month == 12].sort_values(["country", "date"]).reset_index(drop=True)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    P = build_panel()
    print(f"panel n={len(P)}  {P.date.min().date()}..{P.date.max().date()}")
    print("  sample spans: " + "  ".join(
        f"{c}:{P[P.country==c].date.min().year}-{P[P.country==c].date.max().year}"
        f"(n={len(P[P.country==c])})" for c in sorted(P.country.unique())) + "\n")

    print("=== CORRELATION OF TERM PREMIUM WITH STOCK-BOND CORRELATION ===")
    for tcol in ("tp3", "tp10"):
        per = "  ".join(f"{c}:{P[P.country==c][tcol].corr(P[P.country==c].sbc):+.2f}"
                        for c in sorted(P.country.unique()))
        print(f"  {tcol:5} pooled={P[tcol].corr(P.sbc):+.3f}   {per}")

    L = P[P.country.isin(LONG_SAMPLE)]
    print(f"\n=== FAIR VALUE (long-sample markets: {', '.join(LONG_SAMPLE)}, n={len(L)}) ===")
    fitted = {}
    for tcol in ("tp3", "tp10"):
        f = fit_fair_value(L[tcol].values, L.sbc.values, L.gap.values)
        print(f"  {tcol:5}: TP* = {f.intercept:+.2f} {f.b_sbc:+.2f}*SBC {f.b_gap:+.4f}*gap"
              f"    R2={f.r2:.3f}")
        fitted[tcol] = f

    print("\n=== ERROR CORRECTION (annual speed toward fair value) ===")
    for tcol in ("tp3", "tp10"):
        f = fitted[tcol]
        q = L.copy()
        q["fair"] = f.fair_value(q.sbc.values, q.gap.values)
        q["d"] = q.groupby("country")[tcol].diff()
        q["gapfv"] = (q.groupby("country")
                      .apply(lambda g: (g["fair"] - g[tcol]).shift(1))
                      .reset_index(level=0, drop=True))
        q = q.dropna(subset=["d", "gapfv"])
        lam = fit_error_correction(q["d"].values, q["gapfv"].values)
        fitted[tcol] = FairValueParams(f.intercept, f.b_sbc, f.b_gap, lam, f.r2, f.n)
        hl = np.log(0.5) / np.log(1 - lam)
        print(f"  {tcol:5}: lambda={lam:.3f}/yr  half-life={hl:.1f}y"
              f"   3y closes {100*(1-(1-lam)**3):.0f}%   10y closes {100*(1-(1-lam)**10):.0f}%")

    from validation.okun_mirage import annual, build_panel as okun_panel
    A = annual(okun_panel())
    okun = fit_hockey_stick(A["level"], A["vol"])
    sbc_model = fit_sbc(P.sbc.values, P.sigma_pi.values, P.sigma_g.values)
    sigma_g = float(P.sigma_g.median())

    print(f"\n=== FULL CHAIN (sigma_g held at panel median {sigma_g:.2f}pp) ===")
    for tcol, tenor in (("tp3", "3y"), ("tp10", "10y")):
        tp_today = float(P[(P.country == "US")].sort_values("date")[tcol].iloc[-1])
        print(f"\n  --- {tenor} term premium, starting from US latest {tp_today:.2f}% ---")
        print("   terminal   sigma_pi   stock-bond   fair    TP@3y    TP@10y")
        print("   inflation             corr         value   horizon  horizon")
        for lv in (1.0, 2.0, 2.5, 3.0, 4.0, 5.0):
            a = scenario_term_premium(lv, tp_today, 3, okun, sbc_model, fitted[tcol], sigma_g)
            b = scenario_term_premium(lv, tp_today, 10, okun, sbc_model, fitted[tcol], sigma_g)
            print(f"     {lv:4.1f}%     {a['sigma_pi']:5.2f}pp     {a['stock_bond_corr']:+.2f}"
                  f"       {a['tp_fair_value']:5.2f}%  {a['tp_at_horizon']:5.2f}%   {b['tp_at_horizon']:5.2f}%")


if __name__ == "__main__":
    _main()
