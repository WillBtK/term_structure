"""Fair value + fading deviation, real time: y0 - (1-rho^H)*resid + (1-lam^H)*(E_view-E_mkt).

resid = the ten-year premium less the driver model's fair premium, the model refitted each
quarter on data available at the origin. Reads the rows written by
validation.backtest_yield_forecasts.

    python -m validation.backtest_fair_value_reversion
"""
import warnings, sys, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from model.tp_drivers import build_panel, fit_market
from validation.backtest_yield_forecasts import OUT

def realtime_resid(c, origins, ref_cache):
    p = build_panel(c); out = {}
    for p0 in origins:
        t = pd.Timestamp(p0.start_time) + pd.DateOffset(months=1, days=14)
        sub = p.loc[:t - pd.DateOffset(days=1)]
        if len(sub.dropna(subset=["tp"])) < 12 * 12:
            continue
        try:
            ref = None
            if c != "US":
                ref = ref_cache.get(p0)
                if ref is None: continue
            m = fit_market(c, sub, reference=ref)
            if c == "US": ref_cache[p0] = m
            last = sub.iloc[-1]
            kw = {k: float(last[k]) for k in sub.columns if k != "tp" and pd.notna(last[k])}
            out[p0] = float(last["tp"]) - m.premium(**kw)
        except Exception as e:
            continue
    return pd.Series(out, name="resid")

ref = {}
res = {}
for c in ("US", "UK", "DE", "JP"):
    rows = pd.read_pickle(OUT / f"{c.lower()}_rows.pkl")
    r = realtime_resid(c, sorted(rows.origin.unique()), ref)
    d = rows.merge(r, left_on="origin", right_index=True)
    if "E_none" not in d: d["E_none"] = d["E_mkt"]
    if "E_oracle" not in d and "rwview|oracle|0.0" in d: d["E_oracle"] = d["rwview|oracle|0.0"] - d.y0 + d.E_mkt
    res[c] = d
    print(f"\n== {c}: origins {d.origin.min()}..{d.origin.max()}, residual now {r.iloc[-1]:+.2f}")
    views = ["none"] + [v for v in ("spf", "oracle") if f"E_{v}" in d]
    for h in (1, 4, 5):
        x = d[d.h == h]
        if not len(x): continue
        line = f"h={h} n={len(x)}"
        for v in views:
            for rho in (1.0, 0.7, 0.5, 0.3):
                lam = 0.8
                xx = x.dropna(subset=["truth", "y0", "resid", f"E_{v}"])
                f = xx.y0 - (1 - rho ** xx.H) * xx.resid + (1 - lam ** xx.H) * (xx[f"E_{v}"] - xx.E_mkt)
                rw = np.sqrt(((xx.y0 - xx.truth) ** 2).mean())
                line += f" | {v[:3]} r{rho}: {np.sqrt(((f - xx.truth) ** 2).mean()) / rw:.3f}"
        print(line)
    # stability: how much of a quarter-on-quarter spot move passes into the h=4 forecast
    x = d[d.h == 4].sort_values("origin").dropna(subset=["resid"])
    dy = x.y0.diff()
    for rho in (1.0, 0.5):
        f = x.y0 - (1 - rho ** x.H) * x.resid + (1 - 0.8 ** x.H) * (x.E_none - x.E_mkt)
        df_ = f.diff(); ok = dy.notna()
        print(f"   rho {rho}: pass-through of spot to forecast {np.polyfit(dy[ok], df_[ok], 1)[0]:.2f}, "
              f"sd revisions/sd spot {df_.std() / dy.std():.2f}")
pd.to_pickle(res, OUT / "fair_value_rows.pkl")
