"""Monthly US term premia, exported for the fiscal-sensitivity repo.

``WillBtK/imf_fiscal_sensitivity`` regresses yields and term premia on CBO fiscal
projections. Its term-premium data is Adrian-Crump-Moench, and its own handoff records
two ceilings that ACM imposes rather than the analysis choosing:

    "No term premium past 10y. Anything you want at 20y or 30y has to come from a
     model you fit yourself, off the GSW curve."

    "No real/breakeven decomposition of the term premium. ACM is nominal only ...
     constructing one requires a real-rate term-structure model that does not exist
     in this repo."

Both exist here. This writes them out in the shape that repo already consumes.

    tp{2,3,5,7,10,20,30}    nominal term premium, 1961+   (ACM: 1-10y)
    rtp{2,3,5,7,10,20}      REAL term premium, 2004+      (ACM: none)
    btp{2,3,5,7,10,20}      tp - rtp: the inflation-risk premium, the breakeven
                            decomposition of the premium  (ACM: impossible)

Conventions chosen to make the swap a one-line change over there:

* **Percent**, like ``acm_tp5`` and like that repo's GSW yields. The parquet here is
  decimal.
* **Calendar-month average of daily observations**, because that repo averages market
  series to the vintage month before an as-of merge. Doing the averaging here means the
  merge logic is untouched.
* ``n_days`` is carried so a partial final month is visible rather than silent.

The premia are AFNS + Christensen-Rudebusch shadow-rate estimates on each market's
regime-clean sample, with Bauer-Rudebusch-Wu bias correction -- not ACM, and not
directly comparable to it. ``compare_with_acm`` reports the difference at the two
maturities they overlap on, so a re-estimation can be read against the old one rather
than as a fresh series of unknown provenance.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "viz" / "data_daily.parquet"
OUT = Path(__file__).resolve().parent / "us_term_premia_monthly.csv"
COUNTRY = "US"
MIN_DAYS = 5          # a month with fewer observations than this is dropped


def _wide(daily: pd.DataFrame, instrument: str, prefix: str) -> pd.DataFrame:
    d = daily[daily["instrument"] == instrument]
    if d.empty:
        return pd.DataFrame()
    d = d.assign(month=d["date"].values.astype("datetime64[M]"))
    g = d.groupby(["month", "tenor"])["term_premium"]
    out = (g.mean().unstack("tenor") * 100.0)
    days = g.size().unstack("tenor").max(axis=1)
    out = out[days >= MIN_DAYS]
    out.columns = [f"{prefix}{t:g}" for t in out.columns]
    return out


def build(country: str = COUNTRY) -> pd.DataFrame:
    """Monthly nominal, real and inflation-risk premia, in percent."""
    daily = pd.read_parquet(DAILY)
    daily = daily[daily["country"] == country]
    nom = _wide(daily, "nominal", "tp")
    real = _wide(daily, "real", "rtp")
    out = nom.join(real, how="outer")
    for c in real.columns:                       # btp = nominal - real, where both exist
        t = c[3:]
        if f"tp{t}" in out:
            out[f"btp{t}"] = out[f"tp{t}"] - out[c]
    n = (daily[daily["instrument"] == "nominal"]
         .assign(month=lambda x: x["date"].values.astype("datetime64[M]"))
         .groupby("month")["date"].nunique())
    out.insert(0, "n_days", n.reindex(out.index).astype("Int64"))
    out.index.name = "month"
    return out.sort_index().round(4)


def compare_with_acm(monthly: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ours against ACM at 5y and 10y -- the only maturities they share.

    Reported so a re-estimation over there can be read as a change from a known
    baseline. The two are different models on different samples; correlation matters
    more than level.
    """
    from validation.benchmarks import load_acm
    m = build() if monthly is None else monthly
    acm = load_acm()          # tidy: date, maturity, term_premium, ... in DECIMAL
    acm = acm.assign(month=pd.to_datetime(acm["date"]).values.astype("datetime64[M]"))
    rows = []
    for t in (5, 10):
        sub = acm[acm["maturity"] == float(t)]
        if sub.empty or f"tp{t}" not in m:
            continue
        a = sub.groupby("month")["term_premium"].mean() * 100.0
        j = pd.concat([m[f"tp{t}"].rename("ours"), a.rename("acm")], axis=1).dropna()
        rows.append({"tenor": t, "n_months": len(j),
                     "from": str(j.index.min())[:7], "to": str(j.index.max())[:7],
                     "corr": round(float(j["ours"].corr(j["acm"])), 3),
                     "mean_ours": round(float(j["ours"].mean()), 2),
                     "mean_acm": round(float(j["acm"].mean()), 2),
                     "mean_gap": round(float((j["ours"] - j["acm"]).mean()), 2)})
    return pd.DataFrame(rows)


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    m = build()
    OUT.write_text(m.to_csv())
    tp = [c for c in m.columns if c.startswith("tp")]
    rtp = [c for c in m.columns if c.startswith("rtp")]
    btp = [c for c in m.columns if c.startswith("btp")]
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(m)} months, "
          f"{str(m.index.min())[:7]}..{str(m.index.max())[:7]}, "
          f"{len(OUT.read_text()):,} bytes)")
    print(f"  nominal  {len(tp):>2} tenors: {' '.join(tp)}")
    print(f"  real     {len(rtp):>2} tenors: {' '.join(rtp)}")
    print(f"  infl-risk{len(btp):>2} tenors: {' '.join(btp)}")
    print(f"\nlatest:\n{m.tail(2).T.dropna().to_string()}")
    print("\nagainst ACM where they overlap:")
    print(compare_with_acm(m).to_string(index=False))


if __name__ == "__main__":
    _main()
