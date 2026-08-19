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
    """Daily engine-ready panel for a market/instrument at its FULL daily reach.

    Panels run back to each daily source's earliest data (US 1961, UK 1979, JP
    1974, DE 1997, CH 1988), not the modern estimation sample. The production
    parameters are still the modern-sample ones — filter_forward applies them over
    the whole span (today unaffected), and the pre-estimation-sample stretch is
    flagged downstream via each spec's ``est_start``.
    """
    if kind == "us_nominal":
        from us.fetch_gsw import build_yield_panel
        from us.config import COMMON_GRID
        return build_yield_panel(grid=COMMON_GRID).dropna(how="all")   # 1961+ (10y from 1971)
    if kind == "us_real":
        from us.tips import build_real_panel
        return build_real_panel(freq="D").dropna(how="all")            # 2004+ (= est sample)
    if kind == "de_nominal":
        from germany.fetch_bundesbank import build_yield_panel
        from germany import config as g
        return build_yield_panel(grid=g.COMMON_GRID, freq="D").dropna(how="all")   # 1997-08+
    if kind == "jp_nominal":
        from japan.fetch_mof import build_yield_panel
        from japan import config as j
        return build_yield_panel(grid=j.COMMON_GRID, freq="D",
                                 start="1974-01-01").dropna(how="all")  # 1974+ (10y from 1986)
    if kind == "uk_nominal":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("nominal", grid=u.COMMON_GRID, freq="D",
                                 start=None).dropna(how="all")          # 1979+
    if kind == "uk_real":
        from uk.fetch_boe import build_yield_panel
        from uk import config as u
        return build_yield_panel("real", grid=u.REAL_GRID, freq="D",
                                 start=None).dropna(how="all")          # 1985+
    if kind == "ch_nominal":
        from switzerland.fetch_snb import load_yields
        from switzerland import config as sw
        # The successor SNB daily cube (rendeiduebd) publishes 1-10,20,30y — every
        # COMMON_GRID tenor natively, so the PCHIP step below is now a no-op pass-through
        # (it interpolates exactly at the nodes). It is kept as a guard: the predecessor
        # cube omitted 10y and had to be interpolated, and this keeps the build robust if
        # SNB ever drops a tenor again.
        full = load_yields(freq="D")                     # all available daily maturities
        full = full[full.index >= pd.Timestamp(sw.SAMPLE_START)].dropna(how="all")
        return _interp_to_grid(full, sw.COMMON_GRID)
    if kind == "ca_nominal":
        from canada.fetch_boc import build_yield_panel
        from canada import config as ca
        # BoC ZC curve carries every grid tenor directly (0.25..30y) — no interp.
        return build_yield_panel(grid=ca.COMMON_GRID, freq="D",
                                 start=None).dropna(how="all")   # 1986+ (30y from 1991)
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
# ``est_start`` = start of the sample the production parameters were estimated on;
# data before it is decomposed with extrapolated (modern) parameters and is flagged
# in the viz. ``sample`` now states the full data reach.
SPECS = [
    ("US", "nominal", "us_prod.json", "us_nominal", None,
     {"label": "US Treasuries (nominal)", "sample": "1961+", "est_start": "1987-01-01", "elb": -0.35}),
    ("US", "real", "us_real.json", "us_real", None,
     {"label": "US TIPS (real)", "sample": "2004+", "est_start": "2004-01-01", "elb": None}),
    ("UK", "nominal", "uk_nom.json", "uk_nominal", None,
     {"label": "UK gilts (nominal)", "sample": "1979+", "est_start": "1995-11-01", "elb": -0.35}),
    ("UK", "real", "uk_real.json", "uk_real", None,
     {"label": "UK index-linked (real)", "sample": "1985+", "est_start": "1995-11-01", "elb": None}),
    ("DE", "nominal", "de_prod.json", "de_nominal", None,
     {"label": "German Bunds (nominal)", "sample": "1997+", "est_start": "1984-01-01", "elb": -0.80}),
    ("JP", "nominal", "japan_prod.json", "jp_nominal", "ycc",
     {"label": "Japanese JGBs (nominal)", "sample": "1974+", "est_start": "2000-05-01", "elb": -0.20}),
    ("CH", "nominal", "ch_prod.json", "ch_nominal", None,
     {"label": "Swiss Confederation (nominal)", "sample": "1988+", "est_start": "1988-01-01", "elb": -0.75}),
    ("CA", "nominal", "ca_prod.json", "ca_nominal", None,
     {"label": "Canada (nominal)", "sample": "1986+", "est_start": "1991-01-01", "elb": 0.0}),
]
NAMES = {"US": "United States", "UK": "United Kingdom", "DE": "Germany",
         "JP": "Japan", "CH": "Switzerland", "CA": "Canada"}
ORDER = ["US", "UK", "DE", "JP", "CH", "CA"]


def build_canonical() -> pd.DataFrame:
    """Decompose every market/instrument daily; return the long-form canonical frame."""
    parts = []
    meta = {"names": NAMES, "order": ORDER, "instruments": {}}
    for cc, inst, pfile, kind, vm, m in SPECS:
        prod = ProductionParams.from_json(PARAMS / pfile)
        panel = _panel_daily(kind)
        # Timely nowcast: extend the US curves past the ~1-week-lagged GSW batch with the
        # next-business-day H.15 CMT *change* (GSW-anchored increment splice). Provisional
        # + self-healing; see us.fetch_h15. Recorded in meta so the viz can flag it.
        nc_dates, nc_src = [], None
        if cc == "US":
            from us.fetch_h15 import nowcast_extend
            panel, nc_dates = nowcast_extend(panel, inst)
            nc_src = "H.15 CMT (FRED DGS/DFII)"
        elif cc == "CA" and inst == "nominal":
            # BoC's fitted zero curve lags ~3 weeks; its benchmark bond yields post
            # next-business-day. Same increment splice, plus a roll guard (these are
            # on-the-run bonds, so the series steps when the benchmark changes).
            from canada.fetch_benchmark import nowcast_extend as ca_nowcast
            panel, nc_dates = ca_nowcast(panel)
            nc_src = "BoC benchmark bond yields (Valet)"
        if nc_dates:
            meta.setdefault("nowcast", {})[f"{cc}:{inst}"] = {
                "source": nc_src, "from": nc_dates[0],
                "to": nc_dates[-1], "n_days": len(nc_dates)}
        var_mult_fn = None
        if vm == "ycc":
            from japan import config as j
            var_mult_fn = j.ycc_downweight
        endpoint = None
        if cc == "DE" and inst == "nominal":
            # Neutral-rate shifting endpoint for Germany. The regime re-cut to 1984+ fixed
            # a break INSIDE the sample; it cannot fix r* drifting down secularly WITHIN
            # that regime, which leaves the sample mean implying a 2.84% long-run short
            # rate against a euro-area neutral nearer 2.5%. Inert until the Consensus CPI
            # file lands (see germany.neutral_anchor.is_enabled — the public rungs alone
            # only reach back to 1999 and would seam the sample at that date).
            from germany.neutral_anchor import build_nominal_neutral as de_neutral, is_enabled
            if is_enabled():
                endpoint, ameta = de_neutral(panel.index)
                meta.setdefault("anchor", {})[f"{cc}:{inst}"] = {
                    k: v for k, v in ameta.items() if k != "src_by_date"}
        if cc == "JP" and inst == "nominal":
            # Neutral-rate shifting endpoint for Japan's expected-rate path (fixes the
            # deflation-era theta anchor). Inert (all-NaN -> constant-theta) until the
            # survey/r* CSVs are committed under japan/data/anchors/. See japan.neutral_anchor.
            from japan.neutral_anchor import build_nominal_neutral
            endpoint, ameta = build_nominal_neutral(panel.index)
            meta.setdefault("anchors", {})["JP"] = {k: v for k, v in ameta.items()
                                                    if k != "src_by_date"}
        frame = filter_forward(prod, panel, dt=DT_DAILY, var_mult_fn=var_mult_fn,
                               endpoint=endpoint)
        frame = frame.rename(columns={"observed_yield": "observed",
                                      "expected_rate": "expected"})
        frame["country"] = cc
        frame["instrument"] = inst
        parts.append(frame[["country", "instrument", "date", "maturity",
                            "observed", "expected", "term_premium"]]
                     .rename(columns={"maturity": "tenor"}))
        # Report the lower bound the model was actually FIT with, not a hand-written
        # copy of it: for the markets where the ELB is ML-estimated (DE, JP, CA) the
        # fitted value drifts from the calibration each re-estimation, and the Method
        # section discusses those estimates by name — two places quoting the bound must
        # not be able to disagree. Real curves keep None (their bound is unreachable).
        elb_pct = m.get("elb")
        if elb_pct is not None and getattr(prod, "elb", None) is not None:
            elb_pct = round(float(prod.elb) * 100.0, 2)
        meta["instruments"][f"{cc}:{inst}"] = dict(
            m, elb=elb_pct,
            tenors=[float(t) for t in sorted(panel.columns.astype(float))],
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
