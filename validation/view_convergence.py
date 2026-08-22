"""Can the speed at which the market adopts OUR view be estimated?

``scenarios.engine.VIEW_CONVERGENCE`` (lambda) governs how fast a user's policy path
displaces the curve's own forward expectation. Unlike RESIDUAL_PERSISTENCE, which is
measured directly out of sample, lambda describes convergence to a forecast that has no
historical track record -- so it needs a proxy. Two are run here, and the honest verdict
is in ``validation/FINDINGS_view_convergence.md``.

  A. SPF long-run test. The Survey of Professional Forecasters asks annually for the
     average 3-month bill rate expected over the next TEN years -- the same object our
     expectations leg projects, forecast independently, and dated. When it disagrees
     with the curve, how much of the gap does the curve close?

  B. Perfect-foresight test. Much longer sample: when the curve is wrong about the
     ten-year average short rate, how much of that error is still outstanding h years
     later? An upper bound on lambda, since shocks landing inside the window sit in
     both errors and are never forecastable at any horizon.

Run: python -m validation.view_convergence
"""
from __future__ import annotations


def spf_test() -> None:
    import warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    import numpy as np, pandas as pd
    
    spf = pd.read_csv("us/data/anchors/spf_bill10.csv", parse_dates=["date"]).set_index("date")["value"]
    d = pd.read_parquet("viz/data_daily.parquet")
    n = d[(d["country"] == "US") & (d["instrument"] == "nominal")]
    exp10 = (n[n["tenor"] == 10.0].set_index("date")["expected"].sort_index() * 100).resample("MS").mean()
    
    # curve reading at each SPF vintage, and h years later
    rows = []
    for t, s in spf.items():
        t0 = pd.Timestamp(t.year, t.month, 1)
        if t0 not in exp10.index or pd.isna(exp10.loc[t0]):
            continue
        rows.append({"date": t0, "spf": float(s), "curve": float(exp10.loc[t0])})
    df = pd.DataFrame(rows).set_index("date")
    print(f"SPF vintages matched to the curve: n={len(df)}  {df.index[0].date()}..{df.index[-1].date()}")
    print(f"mean gap (SPF - curve) {(df['spf']-df['curve']).mean():+.2f}pp, "
          f"sd {(df['spf']-df['curve']).std():.2f}pp, "
          f"|gap|>0.25pp in {(abs(df['spf']-df['curve'])>0.25).mean():.0%} of years\n")
    
    print("Curve_(t+h) = a + b*Curve_t + c*SPF_t     [c = weight the curve ends up putting on the view]")
    print(f"{'h':>3} {'n':>4} {'b':>7} {'c':>7} {'se(c)':>7} {'t(c)':>6} {'share to view':>14} {'implied lambda':>15}")
    out = {}
    for h in (1, 2, 3, 5):
        a_, b_, y_ = [], [], []
        for t, r in df.iterrows():
            tf = t + pd.DateOffset(years=h)
            tf = pd.Timestamp(tf.year, tf.month, 1)
            if tf in exp10.index and pd.notna(exp10.loc[tf]):
                a_.append(r["curve"]); b_.append(r["spf"]); y_.append(float(exp10.loc[tf]))
        if len(a_) < 8:
            continue
        X = np.column_stack([np.ones(len(a_)), a_, b_]); Y = np.array(y_)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        s2 = (resid @ resid) / (len(Y) - 3)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * s2))
        b, c = beta[1], beta[2]
        share = c / (b + c) if (b + c) != 0 else np.nan          # weight on the view
        lam = (1 - share) ** (1 / h) if 0 < share < 1 else np.nan
        out[h] = (share, lam)
        print(f"{h:>3} {len(Y):>4} {b:>7.2f} {c:>7.2f} {se[2]:>7.2f} {c/se[2]:>6.1f} "
              f"{share:>13.0%} {lam:>15.2f}")
    
    print("\nDirect form:  (Curve_(t+h) - Curve_t) = a + beta*(SPF_t - Curve_t)")
    print(f"{'h':>3} {'n':>4} {'beta':>7} {'se':>6} {'t':>6} {'implied lambda':>15}")
    for h in (1, 2, 3, 5):
        g, m = [], []
        for t, r in df.iterrows():
            tf = t + pd.DateOffset(years=h); tf = pd.Timestamp(tf.year, tf.month, 1)
            if tf in exp10.index and pd.notna(exp10.loc[tf]):
                g.append(r["spf"] - r["curve"]); m.append(float(exp10.loc[tf]) - r["curve"])
        if len(g) < 8:
            continue
        X = np.column_stack([np.ones(len(g)), g]); Y = np.array(m)
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta; s2 = (resid @ resid) / (len(Y) - 2)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * s2))[1]
        lam = (1 - beta[1]) ** (1 / h) if 0 < beta[1] < 1 else np.nan
        print(f"{h:>3} {len(Y):>4} {beta[1]:>7.2f} {se:>6.2f} {beta[1]/se:>6.1f} {lam:>15.2f}")


def foresight_test() -> None:
    import warnings, logging; warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
    import numpy as np, pandas as pd
    
    d = pd.read_parquet("viz/data_daily.parquet")
    n = d[(d["country"] == "US") & (d["instrument"] == "nominal")]
    piv = n.pivot_table(index="date", columns="tenor", values="expected", aggfunc="mean") * 100
    piv = piv.resample("MS").mean()
    short = (n[n["tenor"] == 0.25].set_index("date")["observed"].sort_index() * 100).resample("MS").mean()
    print("expected-curve tenors:", [t for t in piv.columns][:12])
    print("20y expected available from:", piv[20.0].first_valid_index())
    print("short rate available from:", short.first_valid_index(), "to", short.last_valid_index())
    
    def fwd(t, h, span=10.0):
        """curve's expected average short rate over [t+h, t+h+span], from the reading at t"""
        if t not in piv.index: return np.nan
        row = piv.loc[t].dropna()
        if len(row) < 4: return np.nan
        ts, vs = row.index.values.astype(float), row.values.astype(float)
        cum = lambda x: x * np.interp(x, ts, vs)
        if h + span > ts.max(): return np.nan
        return (cum(h + span) - cum(h)) / span
    
    rows = []
    for T in pd.date_range("1975-01-01", "2016-01-01", freq="MS"):
        win = short.loc[T:T + pd.DateOffset(years=10)]
        if len(win) < 118 or win.isna().any(): continue
        R = float(win.mean())
        S = fwd(T, 0.0)                                   # market's spot view when W starts
        if not np.isfinite(S): continue
        rec = {"T": T, "R": R, "S": S}
        for h in (1, 2, 3, 5):
            t0 = T - pd.DateOffset(years=h)
            rec[f"F{h}"] = fwd(pd.Timestamp(t0.year, t0.month, 1), float(h))
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("T")
    print(f"\nwindows: n={len(df)}  {df.index[0].date()}..{df.index[-1].date()}")
    print(f"\n  e_after = beta * e_before   ->   beta = lambda^h  (share of error still outstanding)")
    print(f"{'h':>3} {'n':>5} {'beta':>7} {'se':>6} {'lambda':>8} {'sd(e_before)':>13}")
    for h in (1, 2, 3, 5):
        s = df[["R", "S", f"F{h}"]].dropna()
        if len(s) < 30: continue
        eb = (s["R"] - s[f"F{h}"]).values                 # error h years before the window
        ea = (s["R"] - s["S"]).values                     # error at the window start
        X = np.column_stack([np.ones(len(eb)), eb])
        beta, *_ = np.linalg.lstsq(X, ea, rcond=None)
        resid = ea - X @ beta; s2 = (resid @ resid) / (len(ea) - 2)
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * s2))[1]
        lam = beta[1] ** (1 / h) if beta[1] > 0 else np.nan
        print(f"{h:>3} {len(eb):>5} {beta[1]:>7.2f} {se:>6.2f} {lam:>8.2f} {eb.std():>13.2f}")


def _main() -> None:
    print("=" * 78)
    print("A. Does the curve converge to the SPF long-run forecast?")
    print("=" * 78)
    spf_test()
    print("\n" + "=" * 78)
    print("B. How fast does the curve correct its own ten-year errors?")
    print("=" * 78)
    foresight_test()


if __name__ == "__main__":
    _main()
