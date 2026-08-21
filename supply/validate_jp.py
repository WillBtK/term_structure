"""Run: ``python -m supply.validate_jp``

Japan: stock and flow, same pre-registered test as the US.

Japan is where the prior says the channel should show up -- the Bank of Japan holds
roughly two-thirds of the JGB market by par against the Federal Reserve's one-seventh
of the Treasury market -- and where the variation is real rather than a narrow band:
the duration share runs from 5% to 44% inside the sample.
"""
import logging
import numpy as np, pandas as pd
logging.basicConfig(level=logging.WARNING)
from macro.tp_history import spliced_tp
from macro.fetch_core_cpi import core_inflation_yoy
from supply.jp_tye import build

tye = build()
tye.index = pd.PeriodIndex(tye.index, freq="M").to_timestamp()
infl = core_inflation_yoy()["JP"].ewm(halflife=54, min_periods=60).mean()

def ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T@A)) * (r@r)/dof, 1e-18))
    return b, b/se, 1 - (r@r)/((y-y.mean())**2).sum()

print(f"duration share AR(12) = {tye.tye_share.autocorr(12):+.3f}   "
      f"flow AR(12) = {tye.tye_share.diff(12).autocorr(12):+.3f}")

for t, k2 in ((10, 3.50), (3, 3.75)):
    D = pd.DataFrame({"tp": spliced_tp(float(t))["JP"], "pi": infl}).dropna()
    D = D.join(tye[["tye_share", "par_share"]].reindex(D.index).ffill(limit=3)).dropna()
    if len(D) < 40:
        print(f"\n{t}y: only {len(D)} obs after join -- skipped"); continue
    pi = D["pi"].values
    seg = [np.maximum(pi-2.0, 0), np.maximum(pi-k2, 0)]
    y = D["tp"].values
    _, _, r0 = ols(y, seg)
    print(f"\n=== {t}y LEVELS, n={len(D)} {D.index.min().date()}..{D.index.max().date()} ===")
    print(f"  inflation only            R2={r0:.3f}")
    for nm in ("tye_share", "par_share"):
        b, tt, r2 = ols(y, seg + [D[nm].values])
        print(f"  + {nm:10}           R2={r2:.3f}   coeff {b[3]:+.4f} (t={tt[3]:+.1f})")

    b0, _, _ = ols(y, seg)
    D["tp_star"] = b0[0] + b0[1]*seg[0] + b0[2]*seg[1]
    D["d_tp"] = D["tp"].diff(12)
    D["gap"] = D["tp_star"] - D["tp"].shift(12)
    D["d_stock"] = D["tye_share"].diff(12)
    D["d_par"] = D["par_share"].diff(12)
    E = D.dropna(subset=["d_tp", "gap", "d_stock"])
    print(f"  --- CHANGES (12m), n={len(E)} ---")
    yy = E["d_tp"].values
    for nm, cols in (("error-correction only     ", [E["gap"].values]),
                     ("+ FLOW (d duration share) ", [E["gap"].values, E["d_stock"].values]),
                     ("+ flow (d PAR share)      ", [E["gap"].values, E["d_par"].values]),
                     ("+ STOCK level             ", [E["gap"].values, E["tye_share"].values])):
        b, tt, r2 = ols(yy, cols)
        extra = f"  x={b[2]:+.4f}(t={tt[2]:+.1f})" if len(b) > 2 else ""
        print(f"    {nm} lambda={b[1]:+.3f}(t={tt[1]:+.1f}){extra}   R2={r2:.3f}")
