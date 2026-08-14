"""Thin the canonical full-daily dataset into the artifact payload (viz/data.json).

A published Claude Artifact is a single self-contained file under a 16 MB CSP-locked
budget — it cannot fetch an external store — so the full ~420k-row daily dataset
can't be embedded whole. This step derives a size-bounded *view* of the canonical
data (viz/data_daily.parquet): **daily** resolution for the recent window and
**weekly** before it. The schema is byte-identical to the monthly payload, so the
artifact JS is unchanged apart from day-resolution x.

Keeping the canonical daily dataset (export_daily.py) and this presentation view
separate is deliberate: a future hosted/Supabase dashboard reads the *canonical*
table directly at full daily resolution and never touches this thinning step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CANON = HERE / "data_daily.parquet"
META = HERE / "data_daily_meta.json"
OUT = HERE / "data.json"

DAILY_YEARS = 6           # recent window kept at daily resolution
# everything older is kept weekly (last observation per ISO week)


def _xdec(idx: pd.DatetimeIndex) -> list[float]:
    """Decimal year with day resolution: year + (dayofyear-1)/365.25."""
    return [round(d.year + (d.dayofyear - 1) / 365.25, 5) for d in idx]


def _keep_dates(dates: pd.DatetimeIndex, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(sorted(dates))
    recent = dates[dates >= cutoff]
    hist = dates[dates < cutoff]
    if len(hist):
        s = pd.Series(hist, index=hist)
        weekly = s.groupby(hist.to_period("W")).last().values
        hist = pd.DatetimeIndex(weekly)
    return hist.append(recent).sort_values()


def _block(sub: pd.DataFrame, meta: dict, cutoff: pd.Timestamp) -> dict:
    tenors = sorted(sub["tenor"].unique())
    all_dates = pd.DatetimeIndex(sub["date"].unique())
    keep = _keep_dates(all_dates, cutoff)
    out = {"tenors": ["%g" % float(t) for t in tenors],
           "x": _xdec(keep),
           "dates": [str(d.date()) for d in keep],
           "obs": {}, "exp": {}, "tp": {}, "breaks": [],
           "label": meta["label"], "sample": meta["sample"], "elb": meta["elb"]}
    for meas, key in (("observed", "obs"), ("expected", "exp"), ("term_premium", "tp")):
        wide = (sub.pivot_table(index="date", columns="tenor", values=meas)
                .reindex(keep))
        for t in tenors:
            tk = "%g" % float(t)
            col = wide[t] if t in wide.columns else pd.Series(index=keep, dtype=float)
            out[key][tk] = [None if pd.isna(v) else round(float(v) * 100, 4) for v in col]
    return out


def build() -> dict:
    canon = pd.read_parquet(CANON)
    canon["date"] = pd.to_datetime(canon["date"])
    meta = json.loads(META.read_text())
    asof = canon["date"].max()
    cutoff = asof - pd.DateOffset(years=DAILY_YEARS)
    countries: dict = {}
    for cc in meta["order"]:
        countries[cc] = {"name": meta["names"][cc], "instruments": {}}
    for key, m in meta["instruments"].items():
        cc, inst = key.split(":")
        sub = canon[(canon["country"] == cc) & (canon["instrument"] == inst)]
        block = _block(sub, m, cutoff)
        countries[cc]["instruments"][inst] = block
        print(f"{cc} {inst}: {len(block['x'])} points "
              f"({block['dates'][0]}..{block['dates'][-1]})")
    return {"meta": {"asof": str(asof.date()), "units": "percent",
                     "resolution": f"daily (last {DAILY_YEARS}y) + weekly history",
                     "note": "Term premium = observed yield - expected avg short rate (realised)."},
            "order": meta["order"], "countries": countries}


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nwrote {OUT}  {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
