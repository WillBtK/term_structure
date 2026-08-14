"""Build the consolidated JSON payload for the term-premium Artifact.

Regenerates each country/instrument's canonical term-premium frame from the
committed ProductionParams (viz/params/*.json) via the fast filter-forward path —
no re-estimation — then reshapes to a compact JSON the single-file Artifact embeds
in its #data-blob. Also folds in regime breakpoints from the regime-dating runs
(scratchpad/regime_*_summary.json) when present.

Schema (decimal yields; x = decimal year = year + (month-1)/12):
{
  "meta": {...},
  "order": ["US","UK","DE","JP"],
  "countries": {
    "US": {"name","instruments": {
       "nominal": {"label","sample","elb","tenors":[..],"x":[..],"dates":[..],
                   "obs":{tenor:[..]}, "exp":{...}, "tp":{...}, "breaks":[{x,label}]},
       "real": {...}   # only where an inflation-linked curve exists
    }}, ...
  }
}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from us.production_tp import ProductionParams, filter_forward

HERE = Path(__file__).resolve().parent
PARAMS = HERE / "params"
SCRATCH = HERE.parent / "scratchpad"


def _panel(kind: str) -> pd.DataFrame:
    if kind == "us_nominal":
        from us.fetch_gsw import build_yield_panel
        from us.config import COMMON_GRID
        p = build_yield_panel(grid=COMMON_GRID)
        p = p[p.index >= pd.Timestamp("1987-01-01")]
        return p.groupby([p.index.year, p.index.month]).tail(1).dropna(how="all")
    if kind == "us_real":
        from us.tips import build_real_panel
        return build_real_panel()
    if kind == "de_nominal":
        from germany.fetch_bundesbank import build_yield_panel
        from germany import config as g
        p = build_yield_panel(grid=g.COMMON_GRID, freq="M")
        p = p[p.index >= pd.Timestamp("1999-01-01")]
        return p.groupby([p.index.year, p.index.month]).tail(1).dropna(how="all")
    if kind == "jp_nominal":
        from japan.fetch_mof import build_yield_panel
        from japan import config as j
        return build_yield_panel(grid=j.COMMON_GRID, freq="M", start=j.SAMPLE_START)
    if kind == "uk_nominal":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("nominal", grid=u.COMMON_GRID, start=u.PROD_START)
    if kind == "uk_real":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("real", grid=u.REAL_GRID, start="1992-10-01")
    if kind == "ch_nominal":
        from switzerland.fetch_snb import build_yield_panel
        from switzerland import config as sw
        return build_yield_panel(grid=sw.COMMON_GRID, freq="M", start=sw.SAMPLE_START)
    raise ValueError(kind)


def _xdec(idx: pd.DatetimeIndex) -> list[float]:
    return [round(d.year + (d.month - 1) / 12.0, 4) for d in idx]


def _frame_to_block(frame: pd.DataFrame, tenors) -> dict:
    dates = sorted(frame["date"].unique())
    didx = pd.DatetimeIndex(dates)
    x = _xdec(didx)
    out = {"tenors": ["%g" % float(t) for t in tenors], "x": x,
           "dates": [str(pd.Timestamp(d).date()) for d in dates],
           "obs": {}, "exp": {}, "tp": {}}
    for t in tenors:
        key = "%g" % float(t)                       # 0.25, 0.5, 2, 10 (unique, no collision)
        sub = frame[frame["maturity"] == t].set_index("date").reindex(dates)
        def col(name):
            return [None if pd.isna(v) else round(float(v) * 100, 4) for v in sub[name]]
        out["obs"][key] = col("observed_yield")
        out["exp"][key] = col("expected_rate")
        out["tp"][key] = col("term_premium")
    return out


def _breaks(tag: str) -> list[dict]:
    """Regime-transition breakpoints from a regime-dating summary, if present."""
    f = SCRATCH / f"regime_{tag}_summary.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text())
    breaks = []
    for (a, b) in d.get("episodes_high_regime", []):
        for edge, lab in ((a, "regime start"), (b, "regime end")):
            y, m, _ = edge.split("-")
            breaks.append({"x": round(int(y) + (int(m) - 1) / 12.0, 4), "date": edge,
                           "label": d.get("tag", tag) + " " + lab})
    return breaks


# (country, instrument) -> (params file, panel kind, var_mult_fn, regime tag, meta)
SPECS = [
    ("US", "nominal", "us_prod.json", "us_nominal", None, None,
     {"label": "US Treasuries (nominal)", "sample": "1987+", "elb": -0.35}),
    ("US", "real", "us_real.json", "us_real", None, None,
     {"label": "US TIPS (real)", "sample": "2004+", "elb": None}),
    ("UK", "nominal", "uk_nom.json", "uk_nominal", None, "uk_nominal",
     {"label": "UK gilts (nominal)", "sample": "1992+", "elb": -0.35}),
    ("UK", "real", "uk_real.json", "uk_real", None, "uk_real",
     {"label": "UK index-linked (real)", "sample": "1992+", "elb": None}),
    ("DE", "nominal", "de_prod.json", "de_nominal", None, "germany",
     {"label": "German Bunds (nominal)", "sample": "1999+", "elb": -0.80}),
    ("JP", "nominal", "japan_prod.json", "jp_nominal", "ycc", "japan",
     {"label": "Japanese JGBs (nominal)", "sample": "1986+", "elb": -0.17}),
    ("CH", "nominal", "ch_prod.json", "ch_nominal", None, None,
     {"label": "Swiss Confederation (nominal)", "sample": "1988+", "elb": -0.75}),
]
NAMES = {"US": "United States", "UK": "United Kingdom", "DE": "Germany",
         "JP": "Japan", "CH": "Switzerland"}


def build() -> dict:
    countries: dict = {}
    for cc, inst, pfile, kind, vm, rtag, meta in SPECS:
        prod = ProductionParams.from_json(PARAMS / pfile)
        panel = _panel(kind)
        var_mult_fn = None
        if vm == "ycc":
            from japan import config as j
            var_mult_fn = j.ycc_downweight
        frame = filter_forward(prod, panel, dt=1 / 12, var_mult_fn=var_mult_fn)
        tenors = sorted(panel.columns.astype(float))
        block = _frame_to_block(frame, tenors)
        block.update(meta)
        block["breaks"] = _breaks(rtag) if rtag else []
        # Regime-derived breakpoints are kept OUT of the viz (Kim-filter transitions
        # are noisy, esp. for Japan); the artifact renders a curated policy-date set
        # instead. Regime datings live in the findings docs.
        block["breaks"] = []
        countries.setdefault(cc, {"name": NAMES[cc], "instruments": {}})
        countries[cc]["instruments"][inst] = block
        print(f"{cc} {inst}: {len(block['x'])} months, tenors {tenors}")
    asof = max(max(b["dates"]) for c in countries.values()
               for b in c["instruments"].values())
    return {"meta": {"asof": asof, "units": "percent",
                     "note": "Term premium = observed yield - expected avg short rate (realised)."},
            "order": ["US", "UK", "DE", "JP", "CH"], "countries": countries}


def main():
    data = build()
    out = HERE / "data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print("wrote", out, f"{out.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
