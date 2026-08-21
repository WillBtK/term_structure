"""Run: ``python -m supply.validate_us``

The pre-registered test, now on the DURATION share rather than the par share.

Registered before the run (unchanged from the earlier attempts): negative sign, an R2
gain over the piecewise-inflation baseline, and -- decisively -- it should hold up out
of sample. The sample starts 2003, so the pre/post-2000 level test is not available;
the analogous sharp test is fit on the PRE-QE years and predict the QE era.

The par share is carried alongside throughout. It is the control: if duration weighting
is doing real work, the two should behave differently.
"""
import logging
import numpy as np, pandas as pd
logging.basicConfig(level=logging.WARNING)
from macro.tp_history import spliced_tp
from macro.fetch_core_cpi import core_inflation_yoy

from supply.us_tye import build
tye = build()
# MSPD snapshots are month-END; spliced_tp is month-START. Align on the period, or the
# join silently produces zero rows.
tye.index = pd.PeriodIndex(tye.index, freq="M").to_timestamp()
infl = core_inflation_yoy()["US"].ewm(halflife=54, min_periods=60).mean()
TP = {t: spliced_tp(float(t))["US"] for t in (3, 10)}

def ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r)/dof, 1e-18))
    return b, b/se, 1 - (r@r)/((y-y.mean())**2).sum()

print(f"tye: {len(tye)} obs  duration share {tye.tye_share.min():.1f}-{tye.tye_share.max():.1f}%"
      f"  par share {tye.par_share.min():.1f}-{tye.par_share.max():.1f}%")
print(f"  corr(duration share, par share) = {tye.tye_share.corr(tye.par_share):+.3f}")
print(f"  duration share peaks {tye.tye_share.idxmax().date()}; "
      f"par share peaks {tye.par_share.idxmax().date()}")

for t, k2 in ((10, 3.50), (3, 3.75)):
    D = pd.DataFrame({"tp": TP[t], "pi": infl}).dropna()
    D = D.join(tye[["tye_share", "par_share"]].reindex(D.index).ffill(limit=40)).dropna()
    pi = D.pi.values
    seg = [np.maximum(pi-2.0, 0), np.maximum(pi-k2, 0)]
    y = D.tp.values
    _, _, r0 = ols(y, seg)
    print(f"\n=== {t}y, n={len(D)} monthly {D.index.min().date()}..{D.index.max().date()} ===")
    print(f"  inflation only                    R2={r0:.3f}")
    for nm in ("tye_share", "par_share"):
        b, tt, r2 = ols(y, seg + [D[nm].values])
        print(f"  + {nm:10}                   R2={r2:.3f}   coeff {b[3]:+.4f} (t={tt[3]:+.1f})")
    print("  --- fit PRE-QE (<=2008), predict QE era (2009+) ---")
    tr, te = D[:'2008'], D['2009':]
    if len(tr) > 24 and len(te) > 24:
        for nm in ([], ["tye_share"], ["par_share"]):
            ctr = [np.maximum(tr.pi.values-2.0,0), np.maximum(tr.pi.values-k2,0)] + [tr[c].values for c in nm]
            cte = [np.maximum(te.pi.values-2.0,0), np.maximum(te.pi.values-k2,0)] + [te[c].values for c in nm]
            bb, _, _ = ols(tr.tp.values, ctr)
            p = bb[0] + sum(bb[i+1]*c for i, c in enumerate(cte))
            lab = nm[0] if nm else "(none)"
            print(f"      {lab:10}: mean error {np.mean(p-te.tp.values):+.2f}pp  "
                  f"RMSE {np.sqrt(np.mean((p-te.tp.values)**2)):.2f}pp")

