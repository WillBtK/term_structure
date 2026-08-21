"""Run: ``python -m supply.validate_us_control``

Does the Fed's balance sheet earn its place as a CONTROL rather than a driver?

If the stated intent is a neutral balance sheet, the variable is worth roughly zero in
projection. The only reason to carry it is that the 2009-2026 period would otherwise
contaminate the inflation coefficients the tool actually uses. So the test is not "does
supply predict the premium" but "does controlling for supply give cleaner and more
stable inflation coefficients".

Three things are checked: the fit over the QE-to-QT window specifically, whether the
inflation slopes estimated before and after 2008 move closer together, and whether the
systematic residual over the QE era shrinks.
"""
import logging
import numpy as np, pandas as pd
logging.basicConfig(level=logging.WARNING)
from macro.tp_history import spliced_tp
from macro.fetch_core_cpi import core_inflation_yoy
from supply.us_tye import build

tye = build()
tye.index = pd.PeriodIndex(tye.index, freq="M").to_timestamp()
infl = core_inflation_yoy()["US"].ewm(halflife=54, min_periods=60).mean()

def ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T@A)) * (r@r)/dof, 1e-18))
    return b, b/se, 1 - (r@r)/((y-y.mean())**2).sum(), r

for t, k2 in ((10, 3.50), (3, 3.75)):
    D = pd.DataFrame({"tp": spliced_tp(float(t))["US"], "pi": infl}).dropna()
    D = D.join(tye[["tye_share"]].reindex(D.index).ffill(limit=3))
    # Pre-2003 the Fed's duration share sat in a narrow band with no programme behind
    # it, so the FLOW is set to zero there rather than dropping fifty years of the
    # inflation relationship the control exists to protect.
    D["flow"] = D["tye_share"].diff(12).fillna(0.0)
    D["stock"] = D["tye_share"].fillna(D["tye_share"].dropna().iloc[0] if D["tye_share"].notna().any() else 0.0)
    pi = D["pi"].values
    seg = [np.maximum(pi - 2.0, 0), np.maximum(pi - k2, 0)]
    y = D["tp"].values
    qe = (D.index >= "2009-01-01")

    print(f"\n{'='*76}\n=== {t}y  n={len(D)}  {D.index.min().date()}..{D.index.max().date()} ===")
    for nm, cols in (("inflation only        ", seg),
                     ("+ duration-share FLOW ", seg + [D["flow"].values]),
                     ("+ duration-share STOCK", seg + [D["stock"].values])):
        b, tt, r2, r = ols(y, cols)
        extra = f"  supply={b[3]:+.4f}(t={tt[3]:+.1f})" if len(b) > 3 else "                      "
        sse = ((r[qe]) ** 2).sum(); sst = ((y[qe] - y[qe].mean()) ** 2).sum()
        print(f"  {nm}  R2={r2:.3f}   slope(2-{k2})={b[1]:+.3f}{extra}"
              f"   QE-era: R2={1-sse/sst:+.3f}  mean resid={r[qe].mean():+.3f}pp  "
              f"RMSE={np.sqrt((r[qe]**2).mean()):.3f}")

    print("  --- inflation slope stability across the QE break ---")
    for nm, extra in (("without control", []), ("with FLOW      ", ["flow"])):
        out = []
        for lab, m in (("<=2008", D.index <= "2008-12-31"), ("2009+", qe)):
            sub = D[m]
            p = sub["pi"].values
            b, _, _, _ = ols(sub["tp"].values,
                             [np.maximum(p-2.0,0), np.maximum(p-k2,0)] +
                             [sub[e].values for e in extra])
            out.append(f"{lab} slope={b[1]:+.3f}")
        print(f"    {nm}: " + "   ".join(out))
