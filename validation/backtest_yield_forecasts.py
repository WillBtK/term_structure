"""Backtest: how to forecast the ten-year yield 1-5 quarters ahead from a house view.

The scenario engine projects a year-end ten-year yield as

    y(H) = Pbar(H) + TP(H)
    Pbar(H) = E_mkt(H) + (1 - lam^H) * (E_view(H) - E_mkt(H))     expected-rate leg
    TP(H)   = TP_fair + rho^H * (TP_today - TP_fair)               premium leg

E_mkt(H) is the market's expected average short rate over [H, H+10] (the model's
expected-rate curve, forward-started), E_view(H) the same average under a view of the
policy path, lam the speed at which a view is priced, rho the persistence of the part of
today's premium the fair value does not explain. This script picks lam and rho by
forecast accuracy and compares the result with simpler and published alternatives.

Origins: SPF survey dates (mid second month of each quarter), 1992Q1 onward. Targets:
the quarterly average of the ten-year zero-coupon yield 1 and 4 quarters after the
origin quarter (the SPF's own horizon definition), plus 5 quarters for the models.
Everything is scored on the CHANGE from the origin, so the SPF (which forecasts the par
CMT yield) is scored against CMT and the rest against the zero curve.

Methods
  rw         no change from the origin date
  rw20       no change from a 20-business-day average ending at the origin
  fwd        the market forward ten-year yield H years ahead
  spf        the SPF's mean ten-year Treasury forecast (TBOND), US only
  regress    the real-policy-rate regression: (y10 - i) on (i - pi - r*), expanding
             window, forecast with the SPF's T-bill and CPI paths (US only; see
             _regression)
  engine     the decomposition above, under three views of the policy path:
               spf   the SPF T-bill path (US): the realistic stand-in for a house view
               pf    perfect foresight: the short rate that actually followed
               none  the market's own path (lam irrelevant)

Look-ahead to keep in mind: the expected-rate curve and premium come from the Atlas's
current estimation (full-sample parameters; survey anchor smoothed with mild look-ahead
between releases). The fair premium is real-time: the trailing ten-year mean.

    python -m validation.backtest_yield_forecasts      # prints the tables, writes CSVs
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "reports" / "yield_forecast_backtest"
SPF = ("https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/"
       "survey-of-professional-forecasters/data-files/files/mean_{v}_level.xlsx")
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
BILL_TO_MODEL = 0.126            # validation/survey_basis.py: US curve short rate - bill
HORIZONS_Q = (1, 4, 5)
LAMS = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0)
RHOS = (0.5, 0.7, 0.8, 0.9, 1.0)
FAIR_YEARS = 10


# ------------------------------------------------------------------ data
def _get(url: str) -> bytes:
    """Download with retries; cached under OUT/cache so reruns are offline."""
    import hashlib
    import time
    import requests
    cache = OUT / "cache" / (hashlib.sha1(url.encode()).hexdigest()[:16])
    if cache.exists():
        return cache.read_bytes()
    last = None
    for wait in (0, 2, 5, 10):
        time.sleep(wait)
        try:
            r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(r.content)
            return r.content
        except Exception as exc:                                    # noqa: BLE001
            last = exc
    raise RuntimeError(f"download failed: {url} ({last})")


def fred(sid: str) -> pd.Series:
    x = pd.read_csv(io.BytesIO(_get(FRED.format(sid))))
    x.columns = ["d", "v"]
    return pd.to_numeric(x.set_index(pd.to_datetime(x["d"]))["v"], errors="coerce").dropna()


def spf(v: str) -> pd.DataFrame:
    d = pd.read_excel(io.BytesIO(_get(SPF.format(v=v))))
    d = d.dropna(subset=["YEAR", "QUARTER"])
    d.index = pd.PeriodIndex([pd.Period(f"{int(y)}Q{int(q)}", freq="Q")
                              for y, q in zip(d["YEAR"], d["QUARTER"])])
    return d.drop(columns=["YEAR", "QUARTER"]).apply(pd.to_numeric, errors="coerce")


def curves(country: str) -> dict:
    d = pd.read_parquet(ROOT / "viz" / "data_daily.parquet")
    d = d[(d["country"] == country) & (d["instrument"] == "nominal")]
    out = {}
    for f in ("observed", "expected", "term_premium"):
        w = d.pivot_table(index="date", columns="tenor", values=f) * 100.0
        out[f] = w.sort_index()
    return out


# ------------------------------------------------------------------ curve arithmetic
def _cum(row: pd.Series, tau: float) -> float:
    """tau * x(tau), x linearly interpolated across tenors."""
    t = np.asarray(row.index, float)
    x = np.asarray(row.values, float)
    ok = np.isfinite(x)
    return tau * float(np.interp(tau, t[ok], x[ok]))


def fwd_avg(row: pd.Series, h: float, span: float = 10.0) -> float:
    """Average of a zero/average-rate curve over [h, h+span] (forward-started)."""
    if h <= 0:
        return float(np.interp(span, np.asarray(row.index, float), row.values))
    return (_cum(row, h + span) - _cum(row, h)) / span


def at(df: pd.DataFrame | pd.Series, t: pd.Timestamp):
    sub = df.loc[:t]
    return sub.iloc[-1] if len(sub) else None


# ------------------------------------------------------------------ the forecasting set-up
def origins(start="1992-01-01", end="2026-06-30") -> list:
    q = pd.period_range(start, end, freq="Q")
    return [(p, pd.Timestamp(p.start_time) + pd.DateOffset(months=1, days=14)) for p in q]


def target_avg(series: pd.Series, p: pd.Period) -> float:
    s = series.loc[p.start_time:p.end_time]
    return float(s.mean()) if len(s) >= 20 else np.nan


def horizon_years(t: pd.Timestamp, p: pd.Period) -> float:
    mid = p.start_time + (p.end_time - p.start_time) / 2
    return (mid - t).days / 365.25


def spf_bill_path(row: pd.Series, p0: pd.Period) -> "callable":
    """The SPF mean T-bill path as a function of years from the origin, model basis.
    Quarters 0..4 ahead from TBILL2..6; then the annual TBILLB/C/D by calendar year;
    TBILLD held beyond."""
    q = {p0 + k: row.get(f"TBILL{k + 2}") for k in range(5)}
    ann = {p0.year + k: row.get(f"TBILL{c}") for k, c in enumerate("ABCD")}
    ann = {y: v for y, v in ann.items() if v is not None and np.isfinite(v)}
    # the furthest annual forecast the round has (TBILLC/D only from 2005/2009), held
    last = ann[max(ann)] if ann else np.nan

    def r(years_ahead: float) -> float:
        pq = (pd.Timestamp(p0.start_time) + pd.DateOffset(months=int(round(years_ahead * 12)) + 1)
              ).to_period("Q")
        v = q.get(pq)
        if v is None or not np.isfinite(v):
            v = ann.get(pq.year, last)
        if v is None or not np.isfinite(v):
            v = last
        return float(v) + BILL_TO_MODEL
    return r


def path_avg(fn, h: float, span: float = 10.0, step: float = 0.25) -> float:
    g = np.arange(h + step / 2, h + span, step)
    return float(np.mean([fn(x) for x in g]))


# ------------------------------------------------------------------ methods
def engine(E_mkt, E_view, TP0, TP_fair, H, lam, rho):
    pbar = E_mkt + (1 - lam ** H) * (E_view - E_mkt)
    return pbar + TP_fair + rho ** H * (TP0 - TP_fair)


def _regression(hist: pd.DataFrame, t: pd.Timestamp):
    """(y10 - i) = a + b * (i - pi - r*), fitted on quarterly data up to t."""
    h = hist.loc[:t].dropna()
    if len(h) < 24:
        return None
    X = np.c_[np.ones(len(h)), h["gap"]]
    coef, *_ = np.linalg.lstsq(X, h["spread"], rcond=None)
    return coef


# ------------------------------------------------------------------ backtest
def run_us() -> pd.DataFrame:
    C = curves("US")
    obs, exp, tp = C["observed"], C["expected"], C["term_premium"]
    y10 = obs[10.0].dropna()
    y10_20 = y10.rolling(20).mean()
    short = obs[0.25].dropna()
    cmt = fred("DGS10")
    bill = fred("DTB3")
    cpi = fred("CPIAUCSL")
    pi = ((cpi / cpi.shift(12) - 1) * 100).dropna()
    tb, tbond, cpis = spf("tbill"), spf("tbond"), spf("cpi")
    # the horizon real rate (r*): the SPF's ten-year average bill (asked in Q1) less its
    # ten-year average CPI, as of each date -- available from 1992; the SPF's year-3
    # bill forecast that would otherwise define the horizon only starts in 2009
    a = ROOT / "us" / "data" / "anchors"
    b10 = pd.read_csv(a / "spf_bill10.csv", parse_dates=["date"]).set_index("date")["value"]
    c10 = pd.read_csv(a / "spf_cpi10.csv", parse_dates=["date"]).set_index("date")["value"]

    def rstar_at(t):
        b, c = at(b10, t), at(c10, t)
        return (b - c) if (b is not None and c is not None) else np.nan

    # the regression's real-time history: quarterly, from 1982
    reg = []
    for p, t in origins("1982-01-01", "2026-06-30"):
        if p not in tb.index or p not in cpis.index:
            continue
        i, y, inf = at(bill, t), at(y10, t), at(pi, t)
        rstar = rstar_at(t)
        if None in (i, y, inf) or not np.isfinite(rstar):
            continue
        reg.append({"t": t, "spread": y - i, "gap": (i - inf) - rstar})
    reg = pd.DataFrame(reg).set_index("t")

    tp10 = tp[10.0].dropna()
    rows = []
    for p0, t in origins():
        if t > y10.index.max():
            break
        row_obs, row_exp = at(obs, t), at(exp, t)
        if row_obs is None:
            continue
        y0, y0s, TP0 = at(y10, t), at(y10_20, t), at(tp10, t)
        fair = tp10.loc[t - pd.DateOffset(years=FAIR_YEARS):t].mean()
        for hq in HORIZONS_Q:
            pt = p0 + hq
            truth = target_avg(y10, pt)
            if not np.isfinite(truth):
                continue
            H = horizon_years(t, pt)
            rec = {"origin": p0, "h": hq, "H": H, "y0": y0, "truth": truth,
                   "rw": y0, "rw20": y0s, "fwd": fwd_avg(row_obs, H)}
            E_mkt = fwd_avg(row_exp, H)
            rec["E_mkt"] = E_mkt
            # SPF ten-year forecast (CMT), scored on the CMT change
            if p0 in tbond.index and hq in (1, 4):
                f = tbond.loc[p0, f"TBOND{hq + 2}"]
                c0, ct = at(cmt, t), target_avg(cmt, pt)
                if np.isfinite(f) and c0 is not None and np.isfinite(ct):
                    rec["spf_err"] = (f - c0) - (ct - c0)
            # views
            views = {"none": E_mkt}
            if p0 in tb.index:
                views["spf"] = path_avg(spf_bill_path(tb.loc[p0], p0), H)
            # oracle: the expected-rate leg the market actually priced at the target
            # quarter (its ten-year average expected rate there) -- the ceiling on what
            # any view of the rate path can add to a yield forecast
            e10 = exp[10.0].dropna()
            tq = e10.loc[pt.start_time:pt.end_time]
            if len(tq) >= 20:
                views["oracle"] = float(tq.mean())
            end = t + pd.DateOffset(days=int((H + 10) * 365.25))
            if end <= short.index.max():
                s = short.loc[t + pd.DateOffset(days=int(H * 365.25)):end]
                views["pf"] = float(s.mean())
            for v, Ev in views.items():
                rec[f"E_{v}"] = Ev
                for lam in LAMS:
                    # spot, plus only the part of the view that differs from the market's
                    # priced path: no bet that the priced drift itself materialises
                    rec[f"rwview|{v}|{lam}"] = y0 + (1 - lam ** H) * (Ev - E_mkt)
                for lam in LAMS:
                    for rho in RHOS:
                        rec[f"eng|{v}|{lam}|{rho}"] = engine(E_mkt, Ev, TP0, fair, H, lam, rho)
            # the regression method
            if p0 in tb.index and p0 in cpis.index and hq in (1, 4):
                coef = _regression(reg, t)
                if coef is not None:
                    i_hat = tb.loc[p0, f"TBILL{hq + 2}"]
                    pi_hat = cpis.loc[p0, "CPIA" if pt.year == p0.year else "CPIB"]
                    rstar = rstar_at(t)
                    if np.isfinite(i_hat) and np.isfinite(pi_hat) and np.isfinite(rstar):
                        # fitted on the zero ten-year less the bill, so this is a
                        # zero-curve forecast, scored like the others
                        rec["regress"] = i_hat + coef[0] + coef[1] * ((i_hat - pi_hat) - rstar)
                        # the same method anchored to today: spot plus the CHANGE the
                        # regression predicts between now and the target
                        i0, pi0 = at(bill, t), at(pi, t)
                        now = i0 + coef[0] + coef[1] * ((i0 - pi0) - rstar)
                        rec["regress_anchored"] = y0 + (rec["regress"] - now)
            rows.append(rec)
    return pd.DataFrame(rows)


def run_other(country: str, start: str) -> pd.DataFrame:
    """UK / euro area (Bund) / Japan: no survey path in hand, so only the no-view and
    oracle cases -- which need no forecasts -- plus no-change and forwards."""
    C = curves(country)
    obs, exp, tp = C["observed"], C["expected"], C["term_premium"]
    y10, e10, tp10 = obs[10.0].dropna(), exp[10.0].dropna(), tp[10.0].dropna()
    rows = []
    for p0, t in origins(start):
        if t > y10.index.max():
            break
        row_obs, row_exp = at(obs, t), at(exp, t)
        if row_obs is None:
            continue
        y0, TP0 = at(y10, t), at(tp10, t)
        hist = tp10.loc[t - pd.DateOffset(years=FAIR_YEARS):t]
        if hist.index.min() > t - pd.DateOffset(years=5):
            continue
        fair = hist.mean()
        for hq in HORIZONS_Q:
            pt = p0 + hq
            truth = target_avg(y10, pt)
            if not np.isfinite(truth):
                continue
            H = horizon_years(t, pt)
            E_mkt = fwd_avg(row_exp, H)
            rec = {"origin": p0, "h": hq, "H": H, "y0": y0, "truth": truth, "rw": y0,
                   "fwd": fwd_avg(row_obs, H), "E_mkt": E_mkt}
            tq = e10.loc[pt.start_time:pt.end_time]
            views = {"none": E_mkt}
            if len(tq) >= 20:
                views["oracle"] = float(tq.mean())
            for v, Ev in views.items():
                for lam in LAMS:
                    rec[f"rwview|{v}|{lam}"] = y0 + (1 - lam ** H) * (Ev - E_mkt)
                    for rho in RHOS:
                        rec[f"eng|{v}|{lam}|{rho}"] = engine(E_mkt, Ev, TP0, fair, H, lam, rho)
            rows.append(rec)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, cols) -> pd.DataFrame:
    """RMSE / MAE / bias of each method, and its RMSE relative to no-change on the SAME
    forecast dates (methods have different coverage)."""
    out = []
    for c in cols:
        if c not in df:
            continue
        if c == "spf_err":
            e = df[c].dropna()
            idx = e.index
        else:
            d = df[[c, "truth", "y0"]].dropna()
            e = d[c] - d["truth"]
            idx = d.index
        if len(e) < 8:
            continue
        rw = (df.loc[idx, "rw"] - df.loc[idx, "truth"])
        rmse = float(np.sqrt((e ** 2).mean()))
        out.append({"method": c, "n": len(e), "rmse": rmse, "mae": float(e.abs().mean()),
                    "bias": float(e.mean()),
                    "rel_rw": rmse / float(np.sqrt((rw ** 2).mean()))})
    return pd.DataFrame(out).set_index("method")


def _main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_us()
    df.to_pickle(OUT / "us_rows.pkl")
    for hq in HORIZONS_Q:
        d = df[df["h"] == hq]
        base = ["rw", "rw20", "fwd", "spf_err", "regress", "regress_anchored"]
        eng = [c for c in d.columns if c.startswith("eng|") or c.startswith("rwview|")]
        s = score(d, base + eng)
        s.to_csv(OUT / f"us_h{hq}.csv")
        print(f"\n=== US, {hq} quarter(s) ahead (H ~ {d['H'].mean():.2f}y), origins {d['origin'].min()}..{d['origin'].max()}")
        print(s.loc[[c for c in base if c in s.index]].round(3).to_string())
        for fam in ("rwview", "eng"):
          for v in ("spf", "pf", "oracle", "none"):
            sv = s[s.index.str.startswith(f"{fam}|{v}|")]
            if len(sv):
                best = sv["rmse"].idxmin()
                cur = f"{fam}|{v}|0.8|0.9" if fam == "eng" else f"{fam}|{v}|0.8"
                print(f"  {fam}/{v}: best {best} rmse {sv.loc[best,'rmse']:.3f} (rel rw {sv.loc[best,'rel_rw']:.3f});"
                      f" at lam=0.8{' rho=0.9' if fam=='eng' else ''}: {sv.loc[cur,'rmse']:.3f} (rel {sv.loc[cur,'rel_rw']:.3f})")


def _main_other() -> None:
    for c, start in (("UK", "1995-01-01"), ("DE", "1998-01-01"), ("JP", "2000-01-01")):
        df = run_other(c, start)
        df.to_pickle(OUT / f"{c.lower()}_rows.pkl")
        for hq in (1, 4):
            d = df[df["h"] == hq]
            cols = ["rw", "fwd"] + [k for k in d.columns if k.startswith(("eng|", "rwview|"))]
            s = score(d, cols)
            s.to_csv(OUT / f"{c.lower()}_h{hq}.csv")
            line = f"{c} h={hq} n={len(d)} rw {s.loc['rw','rmse']:.3f} fwd {s.loc['fwd','rel_rw']:.3f}"
            for key in ("rwview|oracle|0.0", "rwview|oracle|0.8", "eng|none|0.8|0.9",
                        "eng|none|0.8|1.0", "eng|none|0.8|0.7"):
                if key in s.index:
                    line += f" | {key} {s.loc[key,'rel_rw']:.3f}"
            print(line)


if __name__ == "__main__":
    import sys
    _main_other() if "--other" in sys.argv else _main()
