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

import os

# Recent window kept at daily resolution; everything older is weekly (last obs per
# ISO week). Set DAILY_YEARS >= the full span (or VIZ_FULL_DAILY=1) to keep the
# entire history daily — the payload is still well under the 16 MB artifact cap.
DAILY_YEARS = float(os.environ.get("VIZ_DAILY_YEARS", "100"))
FULL_DAILY = os.environ.get("VIZ_FULL_DAILY", "1") == "1"


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
           "label": meta["label"], "sample": meta["sample"], "elb": meta["elb"],
           "est_start": meta.get("est_start")}
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
    full = cutoff <= canon["date"].min()
    res = "daily, full history" if full else f"daily (last {DAILY_YEARS:g}y) + weekly history"
    _attach_us_regime_overlay(countries)
    _attach_us_benchmarks(countries)
    _attach_us_tips_noise(countries)
    return {"meta": {"asof": str(asof.date()), "units": "percent",
                     "resolution": res,
                     "note": "Term premium = observed yield - expected avg short rate (realised)."},
            "order": meta["order"], "countries": countries}


def _attach_us_regime_overlay(countries: dict) -> None:
    """Attach the US 2-regime (per-era θ) 10y TP series to US:nominal as a
    comparison trace for the deep-past / Volcker window, if the run is present."""
    f = HERE.parent / "scratchpad" / "regime_us_tp.parquet"
    if not f.exists():
        return
    t = pd.read_parquet(f)
    if 10.0 not in t.columns:
        return
    s = (t[10.0] * 100).dropna()
    idx = pd.DatetimeIndex(s.index)
    countries["US"]["instruments"]["nominal"]["regime10"] = {
        "x": _xdec(idx), "dates": [str(d.date()) for d in idx],
        "tp": [round(float(v), 4) for v in s.values],
        "note": "2-regime Kim-filter model (per-era θ); illustrative in the extrapolated era."}


# External US TP benchmarks for side-by-side plotting. (label, loader) — C&R via
# sf_fed; ACM/Kim-Wright/DKW via benchmarks. Each is optional/guarded so a cold or
# degraded source (e.g. FRED) never breaks the build.
def _attach_us_benchmarks(countries: dict) -> None:
    """Attach external US term-premium benchmarks (ACM, Kim-Wright, DKW, C&R) to
    US:nominal as a `benchmarks` sub-block, each `{label, tp:{tenor:[pct]}, x, dates}`
    at month-end resolution (the panel is a slow-moving comparison)."""
    from validation import benchmarks as bm
    from validation import sf_fed
    from validation import config as vcfg
    # (key, label, cache_path, loader) — cache-ONLY: skip any source whose parquet is
    # absent so the build never triggers a live fetch (the daily job refreshes first).
    sources = [
        ("acm", "ACM (NY Fed)", vcfg.ACM_PARQUET_PATH, bm.load_acm),
        ("kim_wright", "Kim-Wright (FRED)", vcfg.KIM_WRIGHT_PARQUET_PATH, bm.load_kim_wright),
        ("dkw", "DKW (Fed Board)", vcfg.DKW_PARQUET_PATH, bm.load_dkw),
        ("cr", "Christensen-Rudebusch (SF Fed)", vcfg.SF_FED_PARQUET_PATH, sf_fed.load_sf_fed_series),
    ]
    out = {}
    for key, label, cache_path, load in sources:
        if not cache_path.exists():
            continue                                               # no cache -> skip (no fetch)
        try:
            df = load(refresh=False)
        except Exception:                                          # noqa: BLE001
            continue
        if df is None or "term_premium" not in df.columns or df.empty:
            continue
        d = df.dropna(subset=["term_premium"]).copy()
        d["date"] = pd.to_datetime(d["date"])
        d = d.groupby([d["date"].dt.year, d["date"].dt.month, "maturity"]).tail(1)  # month-end
        dates = pd.DatetimeIndex(sorted(d["date"].unique()))
        tp = {}
        for mat, g in d.groupby("maturity"):
            s = g.set_index("date")["term_premium"].reindex(dates)
            tp["%g" % float(mat)] = [None if pd.isna(v) else round(float(v) * 100, 4) for v in s]
        out[key] = {"label": label, "x": _xdec(dates),
                    "dates": [str(x.date()) for x in dates], "tp": tp}
        print(f"benchmark {key}: {len(dates)} months, tenors {sorted(tp)}")
    if out:
        countries["US"]["instruments"]["nominal"]["benchmarks"] = out


def _attach_us_tips_noise(countries: dict) -> None:
    """US TIPS market-liquidity gauge — the curve-fitting 'noise' (RMS deviation of
    individual TIPS from a smooth fit), month-end in bp. Optional/guarded."""
    from validation import config as vcfg
    if not vcfg.TIPS_NOISE_PARQUET.exists():
        return
    try:
        from validation.tips_liquidity import build_noise
        m = build_noise(refresh=False, freq="M").dropna(subset=["noise"])
    except Exception:                                              # noqa: BLE001
        return
    if m.empty:
        return
    dates = pd.DatetimeIndex(m["date"])
    countries["US"]["tips_noise"] = {
        "x": _xdec(dates), "dates": [str(d.date()) for d in dates],
        "noise": [round(float(v) * 1e4, 2) for v in m["noise"]]}       # decimal -> bp
    print(f"US TIPS noise gauge: {len(dates)} months, latest "
          f"{countries['US']['tips_noise']['noise'][-1]:.1f} bp")


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nwrote {OUT}  {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
