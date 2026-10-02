"""Spread markets (production relative model, refitted each quarter in real time): does
leaning against the residual pay, and does a CORRECT rate / inflation call pay through the
expected-rate and premium channels? Perfect foresight on the expected-inflation and
uncertainty drivers; the rate call is the oracle path.

    python -m validation.backtest_inflation_channel
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from model.tp_drivers import build_panel, fit_market
from model.tp_relative import build_pair, fit_relative
from validation.backtest_yield_forecasts import OUT
CH = ("expected_inflation", "inflation_uncertainty")
pu = build_panel("US")
def kw(row, cols): return {k: float(row[k]) for k in cols if pd.notna(row[k])}
usfit = {}
for c in ("UK", "DE", "JP"):
    rows = pd.read_pickle(OUT / f"{c.lower()}_rows.pkl")
    rows["E_oracle"] = rows["rwview|oracle|0.0"] - rows.y0 + rows.E_mkt
    pair = build_pair(c); po = build_panel(c)
    out = []
    for p0 in sorted(rows.origin.unique()):
        t = pd.Timestamp(p0.start_time) + pd.DateOffset(months=1, days=14)
        cut = t - pd.DateOffset(days=1)
        try:
            if p0 not in usfit: usfit[p0] = fit_market("US", pu.loc[:cut])
            mu = usfit[p0]; m = fit_relative(c, pair.loc[:cut], us_model=mu)
        except Exception as e:
            continue
        u0, o0 = pu.loc[:cut].iloc[-1], po.loc[:cut].iloc[-1]
        cu = [k for k in pu.columns if k != "tp"]; co = [k for k in po.columns if k != "tp"]
        U0, O0 = kw(u0, cu), kw(o0, co)
        tpus0, tp0 = float(u0.tp), float(o0.tp)
        resid = tp0 - m.premium(U0, O0, tpus0)
        for _, r in rows[rows.origin == p0].iterrows():
            if not np.isfinite(r.truth): continue
            tgt = (p0 + int(r.h)).start_time + pd.DateOffset(months=1)
            if tgt > min(pu.index.max(), po.index.max()): continue
            uT, oT = pu.loc[:tgt].iloc[-1], po.loc[:tgt].iloc[-1]
            UT = dict(U0, **{k: float(uT[k]) for k in CH}); OT = dict(O0, **{k: float(oT[k]) for k in CH})
            dus = mu.premium(**UT) - mu.premium(**U0)                  # US premium move, inflation calls
            dprem = m.beta_us * dus + (m.premium(UT, OT, tpus0) - m.premium(U0, O0, tpus0))
            out.append(dict(h=r.h, H=r.H, y0=r.y0, truth=r.truth, tilt=r.E_oracle - r.E_mkt,
                            resid=resid, dprem=dprem, beta=m.beta_us))
    X = pd.DataFrame(out)
    print(f"\n== {c} n_origins {X.shape[0]//3}  beta {X.beta.median():.2f}")
    for h in (4, 5):
        x = X[X.h == h].dropna()
        rm = lambda f: np.sqrt(((f - x.truth) ** 2).mean()); rw = rm(x.y0)
        lean = lambda rho: x.y0 - (1 - rho ** x.H) * x.resid
        print(f"h={h} n={len(x)} lean-vs-resid rho.7 {rm(lean(.7))/rw:.3f} rho.5 {rm(lean(.5))/rw:.3f} | "
              f"rates call {rm(x.y0+x.tilt)/rw:.3f} + inflation call {rm(x.y0+x.tilt+x.dprem)/rw:.3f} "
              f"| inflation only {rm(x.y0+x.dprem)/rw:.3f} | slope {np.polyfit(x.dprem, x.truth-x.y0-x.tilt,1)[0]:.2f}")
