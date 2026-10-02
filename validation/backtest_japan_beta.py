"""What US-premium beta should a JGB yield forecast use? (production relative model: 0.48)

Two channels carry the US premium into a JGB forecast:
  scenario  : JP premium moves by beta x the US premium move the scenario implies --
              tested with perfect foresight on the realised US move, plus the oracle
              JGB rate path, so only beta is being judged
  spillover : today's US mispricing fades, and beta of that fade reaches JGBs
Scored on the 4- and 5-quarter JGB yield change, by regime (YCC Sep 2016 - Mar 2024).

    python -m validation.backtest_japan_beta
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from validation.backtest_yield_forecasts import OUT, curves, at, target_avg

BETAS = (0.0, 0.1, 0.15, 0.25, 0.35, 0.48)
REGIMES = {"all 2006-26": ("2006", "2026"), "pre-YCC": ("2006", "2016Q2"),
           "YCC": ("2016Q3", "2023Q4"), "post-YCC": ("2024Q1", "2026")}

tp_us = curves("US")["term_premium"][10.0].dropna()
r = pd.read_pickle(OUT / "jp_rows.pkl")
r["tilt"] = r["rwview|oracle|0.0"] - r.y0
res = pd.read_pickle(OUT / "fair_value_rows.pkl")["US"][["origin", "resid"]].drop_duplicates("origin")
r = r.merge(res, on="origin", how="left")
t0 = [pd.Timestamp(p.start_time) + pd.DateOffset(months=1, days=14) for p in r.origin]
r["dtp_us"] = [target_avg(tp_us, p + int(h)) - at(tp_us, t) for p, h, t in zip(r.origin, r.h, t0)]
for h in (4, 5):
    print(f"\n== {h} quarters ahead: error relative to no-change (lower is better)")
    print(f"{'':14}{'n':>4} | scenario channel, beta = " + " ".join(f"{b:>5}" for b in BETAS)
          + " | slope | spillover, beta = " + " ".join(f"{b:>5}" for b in BETAS))
    for name, (a, b) in REGIMES.items():
        x = r[(r.h == h) & (r.origin >= pd.Period(a, "Q")) & (r.origin <= pd.Period(b, "Q"))]
        x = x.dropna(subset=["truth", "tilt", "dtp_us"])
        if len(x) < 6: continue
        rm = lambda f: np.sqrt(((f - x.truth) ** 2).mean()); rw = rm(x.y0)
        sc = [rm(x.y0 + x.tilt + bb * x.dtp_us) / rw for bb in BETAS]
        slope = np.polyfit(x.dtp_us, x.truth - x.y0 - x.tilt, 1)[0]
        xs = x.dropna(subset=["resid"])
        rws = np.sqrt(((xs.y0 - xs.truth) ** 2).mean())
        sp = [np.sqrt(((xs.y0 - bb * (1 - 0.5 ** xs.H) * xs.resid - xs.truth) ** 2).mean()) / rws for bb in BETAS]
        print(f"{name:14}{len(x):>4} | {'':24}" + " ".join(f"{v:5.3f}" for v in sc)
              + f" | {slope:+.2f} | {'':17}" + " ".join(f"{v:5.3f}" for v in sp))
