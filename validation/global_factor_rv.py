"""Portfolio diagnostic for the global term-premium factor.

Two questions:
  1. **Is the local residual a relative-value signal?** i.e. does "cheap vs the global
     cycle" (a positive local term-premium residual) predict that a market's bond
     *outperforms* the others next month? We build a duration-hedged, cross-sectionally
     demeaned (≈ currency-neutral) monthly excess return from the yield data alone and
     test the residual signal against a naive-carry benchmark.
  2. **Is the global factor economically real?** Correlate G_t with the US 10y term
     premium and with external risk proxies (VIX, the broad dollar) from FRED.

This is a **construct-validity diagnostic, not a tradeable strategy**: synthetic
yields-only excess returns (carry − duration·Δy), cross-sectional demean as a
first-order currency hedge, no transaction costs, no real hedging, a 5-name
cross-section. Read the IR / information coefficient as "does the residual carry
information?", not as a Sharpe ratio.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import global_factor as gf

logger = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
CANON = HERE.parent / "viz" / "data_daily.parquet"
ORDER = ["US", "UK", "DE", "JP", "CH"]
TENOR = 10.0


# ------------------------------------------------------------------ data assembly
def _month_end(s: pd.Series) -> pd.Series:
    return s.groupby([s.index.year, s.index.month]).tail(1)


def load_curve_panels():
    """Month-end (date × market) panels of the 10y observed yield, the short rate
    (shortest available tenor per market), and the 10y term premium — all decimal."""
    df = pd.read_parquet(CANON); df["date"] = pd.to_datetime(df["date"])
    nom = df[df.instrument == "nominal"]
    y10, short, tp10 = {}, {}, {}
    for cc in ORDER:
        g = nom[nom.country == cc]
        s10 = g[np.isclose(g.tenor, TENOR)].set_index("date").sort_index()
        short_ten = g.tenor.min()
        ssh = g[np.isclose(g.tenor, short_ten)].set_index("date").sort_index()
        y10[cc] = _month_end(s10["observed"]); short[cc] = _month_end(ssh["observed"])
        tp10[cc] = _month_end(s10["term_premium"])
    idx = None
    for d in (y10, short, tp10):
        for cc in ORDER:
            d[cc].index = d[cc].index.to_period("M").to_timestamp("M")
    Y10 = pd.DataFrame(y10).sort_index(); SH = pd.DataFrame(short).sort_index()
    TP = pd.DataFrame(tp10).sort_index()
    return Y10, SH, TP


def excess_returns(Y10: pd.DataFrame, SH: pd.DataFrame) -> pd.DataFrame:
    """Monthly duration-hedged, cross-sectionally demeaned relative excess returns.
    xret_{i,t→t+1} = (y10−short)/12  −  D·Δy10,   D = 10/(1+y10);  then demean across
    markets each month (absorbs the common global level + a first-order common
    currency/hedging wedge). Row t is the return realized over [t, t+1]."""
    D = 10.0 / (1.0 + Y10)
    carry = (Y10 - SH) / 12.0
    dprice = -D * Y10.diff().shift(-1)                  # −D·(y_{t+1}−y_t), aligned to t
    xret = carry + dprice
    rx = xret.sub(xret.mean(axis=1), axis=0)            # cross-sectional demean = relative
    return rx


# --------------------------------------------------------------------- backtest
def _xs_weights(sig_row: pd.Series) -> pd.Series:
    """Dollar-neutral, unit-gross cross-sectional weights from a signal row (rank-demeaned)."""
    s = sig_row.dropna()
    if len(s) < 3:
        return pd.Series(0.0, index=sig_row.index)
    r = s.rank() - (len(s) + 1) / 2.0                  # centered ranks (sum 0)
    w = r / np.abs(r).sum() if np.abs(r).sum() > 0 else r * 0.0
    return w.reindex(sig_row.index).fillna(0.0)


def backtest(signal: pd.DataFrame, rx: pd.DataFrame) -> dict:
    """Cross-sectional long-cheap/short-rich backtest of `signal` (known at t) against
    next-month relative returns `rx` (row t = [t,t+1]). Returns IR, hit-rate, IC, PnL."""
    idx = signal.index.intersection(rx.index)
    sig, r = signal.loc[idx], rx.loc[idx]
    port = pd.Series(index=idx, dtype=float)
    ics = []
    for t in idx:
        w = _xs_weights(sig.loc[t])
        port[t] = float((w * r.loc[t].fillna(0.0)).sum())
        both = pd.concat([sig.loc[t], r.loc[t]], axis=1).dropna()
        if len(both) >= 3:
            ics.append(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
    port = port.dropna()
    ir = float(port.mean() / port.std() * np.sqrt(12)) if port.std() > 0 else np.nan
    # pooled rank IC across all (market, month) pairs
    pooled = pd.DataFrame({"s": sig.stack(), "r": r.stack()}).dropna()
    pooled_ic = float(pooled["s"].corr(pooled["r"], method="spearman"))
    return {"IR": ir, "hit_rate": float((port > 0).mean()), "n_months": int(len(port)),
            "mean_xs_IC": float(np.nanmean(ics)) if ics else np.nan,
            "pooled_IC": pooled_ic, "cum_pnl": float(port.sum()), "port": port}


def _pooled_regression(resid: pd.DataFrame, carry: pd.DataFrame, rx: pd.DataFrame) -> dict:
    """Pooled OLS of next-month relative return on the residual and carry signals
    (all cross-sectionally demeaned so it tests the *relative* relationship)."""
    def dm(x):
        return x.sub(x.mean(axis=1), axis=0)
    idx = resid.index.intersection(carry.index).intersection(rx.index)
    d = pd.DataFrame({"resid": dm(resid.loc[idx]).stack(), "carry": dm(carry.loc[idx]).stack(),
                      "rx": rx.loc[idx].stack()}).dropna()
    X = np.column_stack([np.ones(len(d)), d["resid"].to_numpy(), d["carry"].to_numpy()])
    y = d["rx"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid_ols = y - X @ beta
    dof = len(y) - X.shape[1]
    sig2 = (resid_ols @ resid_ols) / dof
    cov = sig2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return {"n": int(len(d)), "b_resid": float(beta[1]), "t_resid": float(beta[1] / se[1]),
            "b_carry": float(beta[2]), "t_carry": float(beta[2] / se[2])}


# ------------------------------------------------------------ construct validity
def _fred_monthly(series_id: str):
    """Month-end FRED series (cached under validation/reports/), or None if unreachable."""
    from validation import config as cfg
    cache = cfg.DATA_DIR / f"fred_{series_id}.parquet"
    try:
        from validation.fred_api import fred_series
        s = fred_series(series_id)
        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        s.to_frame("v").to_parquet(cache)
    except Exception:                                              # noqa: BLE001
        if cache.exists():
            s = pd.read_parquet(cache)["v"]
        else:
            return None
    s.index = pd.to_datetime(s.index)
    return _month_end(s).pipe(lambda x: x.set_axis(x.index.to_period("M").to_timestamp("M")))


def construct_validity(G: pd.Series, TP: pd.DataFrame) -> dict:
    """Correlate the global factor with the US 10y TP and external risk proxies."""
    out = {"corr_US_TP": float(G.corr(TP["US"].reindex(G.index)))}
    for sid, label in [("VIXCLS", "VIX"), ("DTWEXBGS", "broad_dollar")]:
        px = _fred_monthly(sid)
        if px is None:
            out[f"corr_{label}"] = None; continue
        j = pd.concat([G.rename("G"), px.rename("x")], axis=1).dropna()
        out[f"corr_{label}"] = float(j["G"].corr(j["x"])) if len(j) > 12 else None
        out[f"corr_{label}_chg"] = (float(j["G"].diff().corr(j["x"].diff()))
                                    if len(j) > 12 else None)
    return out


# --------------------------------------------------------------------------- run
def run(start: str = "1997-08-01") -> dict:
    """Estimate the DFM on full history (ragged), but run the RV backtest only on the
    **balanced** cross-section (default 1997-08+, when all five markets report) so the
    cross-sectional demeaning is well-posed."""
    Y10, SH, TP = load_curve_panels()
    # estimate the DFM on the daily panel; use the FILTERED (no look-ahead) residual
    daily = gf.load_tp_panel(TENOR)
    Yd = daily.to_numpy(float)
    overlap = np.all(np.isfinite(Yd), axis=1)
    p, diag = gf.estimate_dfm(Yd, list(daily.columns), overlap_mask=overlap)
    dec_f = gf.filter_forward(p, Yd, smoothed=False)     # real-time residual for the signal
    dec_s = gf.filter_forward(p, Yd, smoothed=True)      # smoothed factor for description
    resid_daily = pd.DataFrame(dec_f["residual"], index=daily.index, columns=daily.columns)
    G_daily = pd.Series(dec_s["G"], index=daily.index)
    # month-end align to the return panel
    me = lambda s: s.pipe(_month_end).pipe(
        lambda x: x.set_axis(x.index.to_period("M").to_timestamp("M")))
    resid = pd.DataFrame({cc: me(resid_daily[cc]) for cc in ORDER}).sort_index()
    G = me(G_daily)
    rx = excess_returns(Y10, SH)
    carry = (Y10 - SH)                                    # naive-carry benchmark signal
    # backtest only on the balanced window (all 5 markets → well-posed cross-section)
    win = lambda d: d[d.index >= pd.Timestamp(start)]
    resid, carry, rx = win(resid), win(carry), win(rx)
    # signal at t predicts rx over [t,t+1]; both are month-end indexed already
    bt_resid = backtest(resid, rx)
    bt_carry = backtest(carry, rx)
    reg = _pooled_regression(resid, carry, rx)
    cv = construct_validity(G, TP)
    return {"diag": diag, "beta": dict(zip(ORDER, p.beta.round(4))),
            "global_share": dict(zip(ORDER, (p.communality * 100).round(1))),
            "rv_residual": {k: v for k, v in bt_resid.items() if k != "port"},
            "rv_carry": {k: v for k, v in bt_carry.items() if k != "port"},
            "regression": reg, "construct_validity": cv,
            "sample": f"{rx.dropna(how='all').index.min().date()}..{rx.dropna(how='all').index.max().date()}"}


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = run()
    d = r["diag"]
    print(f"\nGlobal TP factor — DFM on 10y (phi={d['phi']:.4f}, PCA corr {d['pca_corr']:.3f})")
    print("global variance share:", r["global_share"])
    print(f"\nRelative-value backtest ({r['sample']}, monthly, 5-name cross-section):")
    print(f"  {'signal':<18}{'IR':>7}{'hit':>7}{'poolIC':>9}{'cumPnL':>9}")
    for name, key in [("local residual", "rv_residual"), ("naive carry", "rv_carry")]:
        b = r[key]
        print(f"  {name:<18}{b['IR']:>7.2f}{b['hit_rate']*100:>6.0f}%{b['pooled_IC']:>9.3f}"
              f"{b['cum_pnl']*100:>8.1f}%")
    g = r["regression"]
    print(f"\nPooled OLS  rx(t+1) ~ residual + carry  (n={g['n']}):")
    print(f"  residual: b={g['b_resid']:+.4f}  t={g['t_resid']:+.2f}")
    print(f"  carry:    b={g['b_carry']:+.4f}  t={g['t_carry']:+.2f}")
    cv = r["construct_validity"]
    print("\nConstruct validity of G:")
    print(f"  corr(G, US 10y TP) = {cv['corr_US_TP']:+.2f}")
    for k in ("corr_VIX", "corr_broad_dollar"):
        if cv.get(k) is not None:
            print(f"  {k} = {cv[k]:+.2f}   d/d = {cv.get(k+'_chg')}")
        else:
            print(f"  {k} = (FRED unavailable)")


if __name__ == "__main__":
    _main()
