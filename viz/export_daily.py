"""Build the **canonical full-daily** term-premium dataset for all markets.

This is the source-of-truth dataset: every market/instrument decomposed at DAILY
frequency via the fast filter-forward path (cached ProductionParams, no
re-estimation). It is stored long-form in a Supabase-shaped schema so the exact
same table can later back a hosted dashboard:

    viz/data_daily.parquet   columns: country, instrument, date, tenor,
                                      observed, expected, term_premium   (DECIMAL)

The artifact payload is a *thinned view* of this dataset (see viz/thin_daily.py) —
keeping the canonical daily data and the presentation payload separate is what
makes a future move to Supabase a one-adapter change.

Daily source coverage (see the per-country fetchers):
    US    GSW daily 1987+          US real  GSW/TIPS daily 2004+
    DE    Bundesbank daily 1999+   UK       BoE GLC daily 1992+
    JP    MoF JGB daily 1986+      UK real  BoE GLC daily 1992+
    CH    SNB rendoblid 1988+ (to ~2025-07)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from us.production_tp import ProductionParams, filter_forward

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "params"
CANON_PARQUET = HERE / "data_daily.parquet"
META_JSON = HERE / "data_daily_meta.json"

DT_DAILY = 1.0 / 252


def _panel_daily(kind: str) -> pd.DataFrame:
    """Daily engine-ready panel for a market/instrument (no monthly downsample)."""
    if kind == "us_nominal":
        from us.fetch_gsw import build_yield_panel
        from us.config import COMMON_GRID
        p = build_yield_panel(grid=COMMON_GRID)
        return p[p.index >= pd.Timestamp("1987-01-01")].dropna(how="all")
    if kind == "us_real":
        from us.tips import build_real_panel
        return build_real_panel(freq="D").dropna(how="all")
    if kind == "de_nominal":
        from germany.fetch_bundesbank import build_yield_panel
        from germany import config as g
        p = build_yield_panel(grid=g.COMMON_GRID, freq="D")
        return p[p.index >= pd.Timestamp("1999-01-01")].dropna(how="all")
    if kind == "jp_nominal":
        from japan.fetch_mof import build_yield_panel
        from japan import config as j
        return build_yield_panel(grid=j.COMMON_GRID, freq="D",
                                 start=j.SAMPLE_START).dropna(how="all")
    if kind == "uk_nominal":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("nominal", grid=u.COMMON_GRID, freq="D",
                                 start=u.PROD_START).dropna(how="all")
    if kind == "uk_real":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("real", grid=u.REAL_GRID, freq="D",
                                 start="1992-10-01").dropna(how="all")
    if kind == "ch_nominal":
        from switzerland.fetch_snb import load_yields
        from switzerland import config as sw
        # The SNB *daily* cube (rendoblid) publishes 1-9,15,20,30y but omits 10y;
        # the fitted Svensson curve is smooth, so interpolate the COMMON_GRID
        # points (only 10y is missing) across the maturity axis per date (PCHIP).
        full = load_yields(freq="D")                     # all available daily maturities
        full = full[full.index >= pd.Timestamp(sw.SAMPLE_START)].dropna(how="all")
        return _interp_to_grid(full, sw.COMMON_GRID)
    raise ValueError(kind)


def _interp_to_grid(curve: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Per-date monotone-cubic (PCHIP) interpolation of a wide spot-curve frame
    onto ``grid`` maturities. Points outside a row's observed maturity span stay
    NaN (no extrapolation) so ragged coverage is preserved."""
    from scipy.interpolate import PchipInterpolator
    grid = np.asarray(grid, dtype=float)
    mats = np.asarray(curve.columns, dtype=float)
    order = np.argsort(mats)
    mats = mats[order]
    vals = curve.to_numpy(dtype=float)[:, order]
    out = np.full((len(curve), len(grid)), np.nan)
    for i in range(len(curve)):
        row = vals[i]
        ok = np.isfinite(row)
        if ok.sum() < 2:
            continue
        xm, ym = mats[ok], row[ok]
        inside = (grid >= xm[0]) & (grid <= xm[-1])
        if inside.any():
            out[i, inside] = PchipInterpolator(xm, ym)(grid[inside])
    return pd.DataFrame(out, index=curve.index, columns=grid).dropna(how="all")


# (country, instrument, params file, panel kind, var_mult_fn tag, meta)
SPECS = [
    ("US", "nominal", "us_prod.json", "us_nominal", None,
     {"label": "US Treasuries (nominal)", "sample": "1987+", "elb": -0.35}),
    ("US", "real", "us_real.json", "us_real", None,
     {"label": "US TIPS (real)", "sample": "2004+", "elb": None}),
    ("UK", "nominal", "uk_nom.json", "uk_nominal", None,
     {"label": "UK gilts (nominal)", "sample": "1992+", "elb": -0.35}),
    ("UK", "real", "uk_real.json", "uk_real", None,
     {"label": "UK index-linked (real)", "sample": "1992+", "elb": None}),
    ("DE", "nominal", "de_prod.json", "de_nominal", None,
     {"label": "German Bunds (nominal)", "sample": "1999+", "elb": -0.80}),
    ("JP", "nominal", "japan_prod.json", "jp_nominal", "ycc",
     {"label": "Japanese JGBs (nominal)", "sample": "1986+", "elb": -0.17}),
    ("CH", "nominal", "ch_prod.json", "ch_nominal", None,
     {"label": "Swiss Confederation (nominal)", "sample": "1988+", "elb": -0.75}),
]
NAMES = {"US": "United States", "UK": "United Kingdom", "DE": "Germany",
         "JP": "Japan", "CH": "Switzerland"}
ORDER = ["US", "UK", "DE", "JP", "CH"]


def build_canonical() -> pd.DataFrame:
    """Decompose every market/instrument daily; return the long-form canonical frame."""
    parts = []
    meta = {"names": NAMES, "order": ORDER, "instruments": {}}
    for cc, inst, pfile, kind, vm, m in SPECS:
        prod = ProductionParams.from_json(PARAMS / pfile)
        panel = _panel_daily(kind)
        var_mult_fn = None
        if vm == "ycc":
            from japan import config as j
            var_mult_fn = j.ycc_downweight
        frame = filter_forward(prod, panel, dt=DT_DAILY, var_mult_fn=var_mult_fn)
        frame = frame.rename(columns={"observed_yield": "observed",
                                      "expected_rate": "expected"})
        frame["country"] = cc
        frame["instrument"] = inst
        parts.append(frame[["country", "instrument", "date", "maturity",
                            "observed", "expected", "term_premium"]]
                     .rename(columns={"maturity": "tenor"}))
        meta["instruments"][f"{cc}:{inst}"] = dict(
            m, tenors=[float(t) for t in sorted(panel.columns.astype(float))],
            start=str(panel.index.min().date()), end=str(panel.index.max().date()),
            n_days=int(len(panel)))
        print(f"{cc} {inst}: {len(panel)} days "
              f"{panel.index.min().date()}..{panel.index.max().date()}")
    canon = pd.concat(parts, ignore_index=True)
    canon["date"] = pd.to_datetime(canon["date"])
    return canon.sort_values(["country", "instrument", "tenor", "date"]), meta


def main() -> None:
    canon, meta = build_canonical()
    canon.to_parquet(CANON_PARQUET, index=False)
    META_JSON.write_text(json.dumps(meta, indent=2))
    asof = str(canon["date"].max().date())
    print(f"\ncanonical daily rows: {len(canon):,}  asof {asof}  "
          f"-> {CANON_PARQUET} ({CANON_PARQUET.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
