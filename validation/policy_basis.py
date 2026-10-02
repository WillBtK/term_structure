"""Policy rate -> curve short-rate basis, per market.

House projections are POLICY rates; the ten-year yield is the expected average of the
GOVERNMENT curve's own short rate plus the premium. The gap is measured as the curve's
shortest fitted tenor less the policy rate, with the policy change realised over that
tenor taken out (a 6-month or 1-year rate prices the path, not today's setting):

    curve(tau) - policy = basis + b x (realised average policy over [t, t+tau] - policy)

    US  3m  vs fed funds target LOWER bound (the house convention): midpoint + 0.152
            (validation/survey_basis.py) + 0.125
    DE  3m  vs ECB deposit facility rate (FRED ECBDFR)
    UK  6m  vs Bank Rate (BoE IUDBEDR)
    JP  1y  vs uncollateralised overnight call rate, monthly (FRED IRSTCI01JPM156N)

    python -m validation.policy_basis
"""
import io
import numpy as np
import pandas as pd
import requests

WINDOWS = {"2017+": ("2017-01-01", None), "2023+": ("2023-01-01", None)}
BOE = ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
       "&Datefrom=01/Jan/2015&Dateto=now&SeriesCodes=IUDBEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N")


def fred(s):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}", timeout=90)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text)); x.columns = ["d", "v"]
    return pd.to_numeric(x.set_index(pd.to_datetime(x.d)).v, errors="coerce").dropna()


def bank_rate():
    r = requests.get(BOE, headers={"User-Agent": "Mozilla/5.0"}, timeout=90); r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    return pd.Series(x.iloc[:, 1].values, index=pd.to_datetime(x.iloc[:, 0], format="%d %b %Y"))


def measure(canon):
    pol = {"DE": (fred("ECBDFR"), 0.25), "UK": (bank_rate(), 0.5),
           "JP": (fred("IRSTCI01JPM156N"), 1.0)}
    out = {}
    for c, (p, ten) in pol.items():
        m = canon[(canon.country == c) & (canon.instrument == "nominal") & (canon.tenor == ten)]
        s = m.set_index("date")["observed"] * 100.0
        if c == "JP":
            s = s.resample("MS").mean()
        p = p.sort_index()
        pp = p.reindex(s.index.union(p.index)).ffill()
        step = pd.Timedelta(days=int(ten * 365.25))
        drift = pd.Series({t: pp.loc[t:t + step].mean() - pp.loc[t] for t in s.index
                           if t + step <= p.index.max()})
        j = pd.DataFrame({"gap": s - pp.reindex(s.index), "drift": drift}).dropna()
        out[c] = {}
        for k, (a, b) in WINDOWS.items():
            x = j.loc[a:b]
            X = np.c_[np.ones(len(x)), x.drift]
            coef, *_ = np.linalg.lstsq(X, x.gap, rcond=None)
            out[c][k] = {"basis": round(float(coef[0]), 3), "drift_coef": round(float(coef[1]), 2),
                         "raw_gap": round(float(x.gap.mean()), 3), "n": len(x)}
    return out


if __name__ == "__main__":
    from pathlib import Path
    canon = pd.read_parquet(Path(__file__).resolve().parent.parent / "viz" / "data_daily.parquet")
    for c, r in measure(canon).items():
        print(c, r)
