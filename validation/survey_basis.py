"""Survey-rate -> model short-rate basis, per market.

Long-horizon surveys forecast a MONEY-MARKET rate (Consensus: 3-month interbank, or the
3-month T-bill in Canada; SPF: the 3-month T-bill; SME: the fed funds target range). The
term premium is measured against the model's own short rate, which is the short end of the
fitted GOVERNMENT curve. The two differ systematically: government bills and short bonds
trade below interbank rates (collateral and safety value), and the US Treasury curve,
fitted to coupon securities, runs above bills. A survey read must be moved onto the
model's basis before it anchors the expected path, or that gap is booked as term premium.

The conversion is the average gap between the model's shortest fitted tenor and the
survey's instrument over the survey era. Interbank and T-bill rates are OECD monthly
3-month rates via FRED (IR3TIB01*M156N); the US also uses the daily 3-month bill (DTB3,
converted from the discount basis) and the fed funds target range (DFEDTARL/U).

    python -m validation.survey_basis        # prints the table the anchors use

Shortest fitted tenor: 3m (US, DE, CA), 6m (UK), 1y (JP, CH). A longer tenor carries some
expected drift over its life; over a full survey era this largely averages out.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
# market -> (FRED 3m rate, model tenor, window start, window end)
CONSENSUS = {"DE": ("IR3TIB01DEM156N", 0.25), "UK": ("IR3TIB01GBM156N", 0.5),
             "JP": ("IR3TIB01JPM156N", 1.0), "CA": ("IR3TIB01CAM156N", 0.25),
             "CH": ("IR3TIB01CHM156N", 1.0)}
SURVEY_ERA = ("2017-01-01", "2026-12-31")          # the Consensus 6-10y history
US_FRAMEWORK = [("2016-01-01", "2019-12-31"), ("2022-07-01", "2026-12-31")]  # ample reserves, off the ZLB


def _fred(sid: str) -> pd.Series:
    import requests
    r = requests.get(FRED.format(sid), timeout=60); r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text)); x.columns = ["date", "v"]
    return pd.to_numeric(x.set_index(pd.to_datetime(x["date"]))["v"], errors="coerce").dropna()


def _model_short(canon: pd.DataFrame, c: str, tenor: float) -> pd.Series:
    m = canon[(canon.country == c) & (canon.instrument == "nominal") & (canon.tenor == tenor)]
    return m.set_index("date")["observed"] * 100.0


def measure(canon: pd.DataFrame) -> dict:
    out = {}
    for c, (sid, ten) in CONSENSUS.items():
        mm = _model_short(canon, c, ten).resample("MS").mean()
        j = pd.DataFrame({"m": mm, "s": _fred(sid)}).dropna().loc[SURVEY_ERA[0]:SURVEY_ERA[1]]
        out[c] = {"vs": f"3m rate ({sid})", "tenor": ten, "gap": round(float((j.m - j.s).mean()), 3),
                  "n": len(j)}
    m3 = _model_short(canon, "US", 0.25)
    bill = -np.log(1 - _fred("DTB3") / 100 * 91 / 360) / (91 / 365) * 100
    mid = (_fred("DFEDTARL") + _fred("DFEDTARU")) / 2
    j = pd.DataFrame({"m": m3, "bill": bill, "mid": mid})
    j = pd.concat([j.loc[a:b] for a, b in US_FRAMEWORK])
    out["US"] = {"vs": "3m T-bill (SPF)", "tenor": 0.25,
                 "gap": round(float((j.m - j.bill).dropna().mean()), 3),
                 "gap_vs_ff_mid": round(float((j.m - j.mid).dropna().mean()), 3),
                 "n": int((j.m - j.bill).notna().sum())}
    return out


if __name__ == "__main__":
    from pathlib import Path
    canon = pd.read_parquet(Path(__file__).resolve().parent.parent / "viz" / "data_daily.parquet")
    for c, r in measure(canon).items():
        print(c, r)
