"""Thin the canonical full-daily dataset into the artifact payload (viz/data.json).

The Atlas is published to Supabase Storage and served through a Cloudflare Worker, so
the old 16 MB Claude-Artifact cap no longer applies. What remains is a *practical* page
budget — the payload is embedded in one self-contained HTML, so every megabyte is parse
time and memory in the reader's browser (and mobile is the binding case, not the CDN).
This step therefore still derives a *view* of the canonical data
(viz/data_daily.parquet) rather than shipping it whole: **daily** resolution for the
recent window and **weekly** before it, with the same schema either way.

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
# entire history daily — currently the default, since page weight (not a hard cap) is
# the only limit now that delivery is Supabase Storage + Cloudflare.
DAILY_YEARS = float(os.environ.get("VIZ_DAILY_YEARS", "100"))
# Days behind the freshest market before a series is badged "stale" in the Atlas.
STALE_DAYS = int(os.environ.get("VIZ_STALE_DAYS", "14"))
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
    # Consistency: the model gives an expected-rate value at every grid tenor, but a
    # maturity that was never observed (e.g. the JGB 20y/30y before those bonds existed)
    # should not show an extrapolated expected-rate line where it has no observed yield.
    # Mask exp to the tenor's observed availability so obs/exp/term-premium line up.
    for tk in out["tenors"]:
        ob, ex = out["obs"].get(tk), out["exp"].get(tk)
        if ob and ex:
            out["exp"][tk] = [None if ob[i] is None else ex[i] for i in range(len(ex))]
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
    # expected-rate anchor provenance (e.g. Japan's survey-anchored endpoint) -> per-market note
    for cc, am in (meta.get("anchors") or {}).items():
        b = countries.get(cc, {}).get("instruments", {}).get("nominal")
        if b is not None and am.get("source") not in (None, "none"):
            b["anchor"] = {"source": am.get("source"), "pi_e": am.get("pi_e"),
                           "r_star": am.get("r_star"), "asof": am.get("asof")}
    # staleness: flag any market/instrument lagging the freshest one by > STALE_DAYS, so a
    # dead upstream source (which keeps returning HTTP 200) is visible rather than silent.
    _ends = {k: m.get("end") for k, m in (meta.get("instruments") or {}).items() if m.get("end")}
    if _ends:
        _newest = max(_ends.values())
        for key, end in _ends.items():
            cc, inst = key.split(":")
            behind = (pd.Timestamp(_newest) - pd.Timestamp(end)).days
            b = countries.get(cc, {}).get("instruments", {}).get(inst)
            if b is not None and behind > STALE_DAYS:
                b["stale"] = {"last": end, "days_behind": int(behind)}
    # timely H.15 nowcast span (provisional tail past the lagged GSW batch) -> per-block flag
    for key, nc in (meta.get("nowcast") or {}).items():
        cc, inst = key.split(":")
        b = countries.get(cc, {}).get("instruments", {}).get(inst)
        if b is not None and nc.get("to"):
            b["nowcast"] = {"from": nc.get("from"), "to": nc.get("to"),
                            "source": nc.get("source"), "n_days": nc.get("n_days")}
    _extend_nominal_tp_history(countries, canon)
    _attach_us_regime_overlay(countries)
    _attach_japan_regime_overlay(countries)
    _attach_japan_regime_monitor(countries)
    _attach_germany_history(countries)
    _attach_us_benchmarks(countries)
    _attach_us_tips_noise(countries)
    _attach_germany_bund_noise(countries)
    result = {"meta": {"asof": str(asof.date()), "units": "percent",
                       "resolution": res,
                       "note": "Term premium = observed yield - expected avg short rate (realised)."},
              "order": meta["order"], "countries": countries}
    _attach_global_factor(result)
    return result


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


def _attach_japan_regime_overlay(countries: dict) -> None:
    """Attach the Japan 2-regime (per-era θ) 10y TP series to JP:nominal as a
    deep-past cross-check for the 1974+ high-inflation era, if the run is present.
    Mirrors the US regime overlay: a separate 2-regime Kim-filter fit whose
    regime-mixed 10y premium is compared against the modern single-regime line
    extrapolated back to 1974 (see the Atlas 'Great Inflation' panel)."""
    f = HERE.parent / "scratchpad" / "regime_jp_tp.parquet"
    if "JP" not in countries or not f.exists():
        return
    t = pd.read_parquet(f)
    col = 10.0 if 10.0 in t.columns else (9.0 if 9.0 in t.columns else None)
    if col is None:
        return
    s = (t[col] * 100).dropna()
    idx = pd.DatetimeIndex(s.index)
    countries["JP"]["instruments"]["nominal"]["regime10"] = {
        "x": _xdec(idx), "dates": [str(d.date()) for d in idx],
        "tp": [round(float(v), 4) for v in s.values],
        "note": "2-regime Kim-filter model (per-era θ); illustrative in the extrapolated era."}


def _attach_japan_regime_monitor(countries: dict) -> None:
    """Attach Japan's MODERN real-time regime monitor to JP:nominal, if present.

    A separate 2-regime Kim-filter fit on the modern (2000+) sample only — its two
    regimes separate the deep deflation/ZIRP world from a higher-level 'normalization'
    world. We ship the FILTERED (real-time, non-revising) probability of the higher
    regime as the headline signal, plus the smoothed (full-sample) probability as an
    optional view. Produced offline by scratchpad/run_jp_modern_regime.py."""
    f = HERE.parent / "scratchpad" / "jp_modern_regime.json"
    if "JP" not in countries or not f.exists():
        return
    try:
        d = json.loads(f.read_text())
    except Exception:                                              # noqa: BLE001
        return
    dates = d.get("dates") or []
    pf = d.get("p_hi_filtered") or []
    if not dates or len(pf) != len(dates):
        return
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    theta = d.get("theta_L_pct") or []
    hi = int(d.get("hi", 0))
    countries["JP"]["instruments"]["nominal"]["regime_monitor"] = {
        "x": _xdec(idx), "dates": dates,
        "p_filtered": [round(float(v), 4) for v in pf],
        "p_smoothed": [round(float(v), 4) for v in (d.get("p_hi_smoothed") or pf)],
        "theta_hi": (round(float(theta[hi]), 2) if theta else None),
        "theta_lo": (round(float(theta[1 - hi]), 2) if len(theta) == 2 else None),
        "start": dates[0],
        "note": "2-regime Kim filter, Japan modern sample (2000+); filtered = real-time."}


def _extend_nominal_tp_history(countries: dict, canon: pd.DataFrame) -> None:
    """Extend the MAIN term-structure explorer's nominal term-premium line back into the
    deep past for the markets with a historical splice, so selecting (e.g.) Japan 10y
    shows 1974+ rather than 1986+. Uses the SAME one-time level-match splice as the
    global factor (`model.global_factor.splice_backfill`): DE 10y/2y ← 1972+ history,
    JP 10y ← the 1974+ 9y overlay. Only the term-premium measure is extended (there is
    no observed/expected path before the modern curve), and only the backfilled tenors;
    the pre-seam rows are monthly and carry None for other tenors and for obs/exp."""
    import json as _json
    from model.global_factor import splice_backfill, _load_hist_tp, _month_end

    def _load_hist_meas(fname, key, ht):
        """A monthly historical-overlay measure (obs/exp) as a month-end DECIMAL series."""
        f = HERE / "history" / fname
        if not f.exists():
            return None
        try:
            dj = _json.loads(f.read_text()); vals = (dj.get(key) or {}).get(ht)
            if vals is None:
                return None
            s = pd.Series([None if v is None else float(v) / 100.0 for v in vals],
                          index=pd.to_datetime(dj["dates"]), dtype=float).dropna()
            return _month_end(s)
        except Exception:                                         # noqa: BLE001
            return None

    # (market, overlay json, {block_tenor: hist_tenor}). "all" = map every tenor in the
    # overlay JSON to itself (DE: full term structure back to 1972). JP: only the 10y
    # node (≤9y term premia already reach 1974 in the canonical set).
    specs = [("DE", "germany_hist_tp.json", "all"),
             ("JP", "japan_hist_tp.json", {"10": "9"})]
    for cc, fname, tmap in specs:
        block = countries.get(cc, {}).get("instruments", {}).get("nominal")
        if not block:
            continue
        if tmap == "all":
            f = HERE / "history" / fname
            if not f.exists():
                continue
            try:
                tmap = {t: t for t in (_json.loads(f.read_text()).get("tenors") or {})}
            except Exception:                                     # noqa: BLE001
                continue
        cc_tp = canon[(canon["country"] == cc) & (canon["instrument"] == "nominal")]
        block_first = pd.Timestamp(block["dates"][0])
        # block_tenor -> {Period: {'tp','obs','exp'} in pct or None}. Where the overlay
        # carries a fitted yield (DE), obs/exp are backfilled too (obs = exp + tp), so the
        # explorer shows Actual/Contribution back to 1972; otherwise (JP) only tp is set.
        pre = {}
        for bt, ht in tmap.items():
            if bt not in block.get("tenors", []):
                continue
            hist_tp = _load_hist_tp(fname, ht)                    # decimal, month-end
            if hist_tp is None:
                continue
            prod = _month_end(cc_tp[np.isclose(cc_tp["tenor"], float(bt))]
                              .set_index("date")["term_premium"])
            if prod.empty:
                continue
            spliced = splice_backfill(prod, hist_tp)              # decimal, continuous
            pre_tp = spliced[spliced.index < prod.index.min()]    # deep-past only (< prod start)
            if not len(pre_tp):
                continue
            hist_obs = _load_hist_meas(fname, "obs", ht)          # fitted yield, decimal (or None)
            m = {}
            for d, tpv in pre_tp.items():
                e = {"tp": round(float(tpv) * 100, 4), "obs": None, "exp": None}
                if hist_obs is not None and d in hist_obs.index:
                    ov = float(hist_obs.loc[d]); e["obs"] = round(ov * 100, 4)
                    e["exp"] = round((ov - float(tpv)) * 100, 4)  # keep obs = exp + tp
                m[d.to_period("M")] = e
            pre[bt] = m
        if not pre:
            continue
        # (A) FILL existing pre-production rows already in the block (JP: ≤9y observed back
        # to 1974, so 1974-1986 rows exist with a None 10y term premium).
        filled = 0
        for i, ds in enumerate(block["dates"]):
            per = pd.Timestamp(ds).to_period("M")
            for bt, mp in pre.items():
                e = mp.get(per)
                if not e:
                    continue
                if block["tp"][bt][i] is None and e["tp"] is not None:
                    block["tp"][bt][i] = e["tp"]; filled += 1
                if block["obs"][bt][i] is None and e["obs"] is not None:
                    block["obs"][bt][i] = e["obs"]
                if block["exp"][bt][i] is None and e["exp"] is not None:
                    block["exp"][bt][i] = e["exp"]
        # (B) PREPEND months entirely before the block's first date (DE: daily curve starts
        # only 1997-08, so 1972-1997 are new monthly rows).
        pre_months = sorted({p for mp in pre.values() for p in mp
                             if p.to_timestamp("M") < block_first})
        if pre_months:
            idx = pd.DatetimeIndex([p.to_timestamp("M") for p in pre_months])
            block["x"] = _xdec(idx) + block["x"]
            block["dates"] = [str(d.date()) for d in idx] + block["dates"]
            for tk in block["tenors"]:
                mp = pre.get(tk, {})
                for meas in ("tp", "obs", "exp"):
                    block[meas][tk] = [(mp.get(p) or {}).get(meas) for p in pre_months] + block[meas][tk]
        if filled or pre_months:
            print(f"{cc} nominal: history extended for {sorted(pre)} "
                  f"(filled {filled} rows, prepended {len(pre_months)} months)")


def _attach_germany_history(countries: dict) -> None:
    """Attach the Germany 1972+ MONTHLY historical term premium (a separate,
    full-sample historical exhibit — not the daily production series) to DE:nominal
    as a `history` sub-block, if the committed series is present.

    Produced offline by `scratchpad/make_de_history.py` (same shadow-AFNS +
    mean-reversion-floor decomposition as the daily line, fit over the full monthly
    Bundesbank Svensson history) and committed as JSON so the deployed build can
    read it without re-estimating."""
    f = HERE / "history" / "germany_hist_tp.json"
    if "DE" not in countries or not f.exists():
        return
    try:
        d = json.loads(f.read_text())
    except Exception:                                              # noqa: BLE001
        return
    dates = d.get("dates") or []
    tenors = d.get("tenors") or {}
    if not dates or not tenors:
        return
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    # Level-align splice (single-point, month-end): the historical uses a full-sample
    # theta (1972+) that the 1970s-80s high rates drag up, biasing its expected-rate
    # path high and its TP low vs the modern-sample production series. Rather than an
    # overlap-mean shift, join the two AT THE EARLIEST OVERLAP MONTH with a one-time
    # per-tenor level adjustment so the spliced line is continuous at the seam. The
    # historical monthly series follows a month-END convention (verified: 10y month-end
    # first-difference corr 0.75 vs 0.56 for month-average), so we match production's
    # month-END value at that month. Preserves the 1970s SHAPE; removes the level bias.
    prod = countries["DE"]["instruments"].get("nominal") or {}
    prod_dates = pd.to_datetime(prod.get("dates") or [])
    prod_tp = prod.get("tp") or {}
    hist_per = idx.to_period("M")
    offsets: dict = {}
    aligned: dict = {}
    seam_month = None
    for t, hvals in tenors.items():
        hs = pd.Series([np.nan if v is None else float(v) for v in hvals], index=hist_per)
        pv = prod_tp.get(t)
        if pv is None or len(pv) != len(prod_dates) or not len(prod_dates):
            continue
        ps = pd.Series([np.nan if v is None else float(v) for v in pv],
                       index=prod_dates).dropna()
        if ps.empty:
            continue
        # production month-END per calendar month (last observation)
        pend = ps.groupby(ps.index.to_period("M")).tail(1)
        pend.index = pend.index.to_period("M")
        common = hs.dropna().index.intersection(pend.index)
        if len(common) < 1:
            continue
        first = common.min()                                       # earliest overlap month
        seam_month = first if seam_month is None else min(seam_month, first)
        off = float(pend[first] - hs[first])                       # one-time level adjustment
        offsets[t] = round(off, 4)
        aligned[t] = [None if v is None else round(float(v) + off, 4) for v in hvals]
    seam = str(prod_dates.min().date()) if len(prod_dates) else None
    hist_block = {
        "x": _xdec(idx), "dates": [str(x.date()) for x in idx],
        "tp": {t: tenors[t] for t in tenors},
        "note": d.get("method", "monthly historical exhibit (full-sample fit)")}
    if offsets:
        hist_block["tp_aligned"] = aligned          # historical + one-time per-tenor offset
        hist_block["align_offset"] = offsets        # percent, prod(month-end) - hist at seam
        hist_block["align_month"] = str(seam_month) if seam_month is not None else None
        hist_block["seam"] = seam                   # production start = splice join date
    countries["DE"]["instruments"]["nominal"]["history"] = hist_block
    if offsets:
        print(f"DE history single-point level-align at {seam_month} (month-end), offsets (pct): "
              + ", ".join(f"{t}y +{offsets[t]:.2f}" for t in sorted(offsets, key=float)))


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


def _attach_global_factor(result: dict) -> None:
    """Global cycle block: for EVERY (measure x curve point) combination, the common
    factor G_t plus each market's observed value, the cycle-implied fitted and the local
    residual. Month-end (the cycle is slow, and one combo at daily resolution cost more
    payload than all fifteen at month-end). Guarded on the derived parquet + meta from
    `viz.global_factor`; older single-combo files are still read (back-compat)."""
    import json as _json
    f = HERE / "global_factor.parquet"
    mf = HERE / "global_factor_meta.json"
    if not f.exists() or not mf.exists():
        return
    # STALENESS GUARD. viz.global_factor is a SEPARATE step: this function only reads
    # whatever parquet is on disk, it does not fit anything. The daily job runs
    # export_daily -> global_factor -> thin_daily in order, but a manual rebuild that
    # skips the middle step publishes new country series beside a global cycle fitted on
    # the old ones — and the result looks entirely plausible (it was, until the DFM's
    # INPUTS were diffed and came back bit-identical). So refuse to be quiet about it.
    canon = HERE / "data_daily.parquet"
    if canon.exists() and f.stat().st_mtime < canon.stat().st_mtime:
        import datetime as _dt
        age = (canon.stat().st_mtime - f.stat().st_mtime) / 3600.0
        print(f"WARNING: global_factor.parquet is {age:.1f}h OLDER than the canonical "
              f"dataset (factor {_dt.datetime.fromtimestamp(f.stat().st_mtime):%Y-%m-%d %H:%M}, "
              f"canonical {_dt.datetime.fromtimestamp(canon.stat().st_mtime):%Y-%m-%d %H:%M}). "
              f"The global cycle was fitted on SUPERSEDED term premia — "
              f"run `python -m viz.global_factor` before this step.")
    d = pd.read_parquet(f); d["date"] = pd.to_datetime(d["date"])
    meta = _json.loads(mf.read_text())
    if "combos" not in meta or "combo" not in d.columns:
        return                                          # pre-multi-combo file: skip rather than mis-render
    combos = {}
    for key, cm in meta["combos"].items():
        sub = d[d["combo"] == key]
        if sub.empty:
            continue
        dates = pd.DatetimeIndex(sorted(sub[sub.country == "GLOBAL"]["date"].unique()))
        if len(dates) == 0:
            continue

        def series(country, kind, scale, _sub=sub, _dates=dates):
            v = _sub[(_sub.country == country) & (_sub.kind == kind)]
            v = v.set_index("date")["value"].reindex(_dates)
            return [None if pd.isna(x) else round(float(x) * scale, 4) for x in v]

        names = [cc for cc in cm["names"] if cc in set(sub.country)]
        combos[key] = {
            "measure": cm["measure"], "measure_label": cm["measure_label"],
            "point": cm["point"], "order": names,
            "x": _xdec(dates), "dates": [str(x.date()) for x in dates],
            "G": series("GLOBAL", "factor", 1),         # standardized (unitless)
            "markets": {cc: {"observed": series(cc, "observed", 100),
                             "fitted": series(cc, "fitted", 100),
                             "residual": series(cc, "residual", 100),
                             "beta": cm["markets"][cc]["beta"],
                             "share": cm["markets"][cc]["global_share"]} for cc in names},
            "pca_ladder": cm.get("pca_ladder", []), "phi": cm.get("phi"),
            "pca_corr": cm.get("pca_corr"), "balanced_start": cm.get("balanced_start")}
    if not combos:
        return
    default = meta.get("default") if meta.get("default") in combos else sorted(combos)[0]
    result["global_factor"] = {
        "freq": meta.get("freq", "M"), "default": default, "combos": combos,
        "measure_labels": meta.get("measure_labels", {}),
        "points": sorted({c["point"] for c in combos.values()},
                         key=lambda p: (len(p), p)),
        "measures": list(dict.fromkeys(c["measure"] for c in combos.values()))}
    n_pts = len(next(iter(combos.values()))["dates"])
    print(f"global cycle: {len(combos)} combos x ~{n_pts} month-end points "
          f"(default {default})")


def _attach_us_tips_noise(countries: dict) -> None:
    """US TIPS market-liquidity gauge — the curve-fitting 'noise' (RMS deviation of
    individual TIPS from a smooth fit), DAILY in bp. Optional/guarded. The Hu-Pan-Wang
    fit is per trading day, so a daily series is the natural resolution; each spike (a
    dislocation) is a single day rather than a smoothed monthly average."""
    from validation import config as vcfg
    if not vcfg.TIPS_NOISE_PARQUET.exists():
        return
    try:
        from validation.tips_liquidity import build_noise
        m = build_noise(refresh=False, freq="D").dropna(subset=["noise"]).sort_values("date")
    except Exception:                                              # noqa: BLE001
        return
    if m.empty:
        return
    dates = pd.DatetimeIndex(m["date"])
    countries["US"]["tips_noise"] = {
        "freq": "daily",
        "x": _xdec(dates), "dates": [str(d.date()) for d in dates],
        "noise": [round(float(v) * 1e4, 2) for v in m["noise"]]}       # decimal -> bp
    print(f"US TIPS noise gauge: {len(dates)} daily obs {dates[0].date()}..{dates[-1].date()}, "
          f"latest {countries['US']['tips_noise']['noise'][-1]:.1f} bp")


def _attach_germany_bund_noise(countries: dict) -> None:
    """German Bund market-liquidity gauge — the curve-fitting 'noise' (RMS deviation of
    individual Bund yields from a smooth fit), DAILY in bp. Optional/guarded.

    Daily (not month-end): the Bundesbank workbook is one cross-section per trading day
    and the available history is short (the index exposes only the most recent months),
    so a daily series is both truer to the data and far more legible than ~5 monthly
    dots. It lengthens day-by-day as the daily job accumulates workbooks."""
    from validation import config as vcfg
    if "DE" not in countries or not vcfg.BUND_NOISE_PARQUET.exists():
        return
    try:
        from validation.bund_liquidity import build_noise
        m = build_noise(refresh=False, freq="D").dropna(subset=["noise"])
    except Exception:                                              # noqa: BLE001
        return
    if m.empty:
        return
    m = m.sort_values("date")
    dates = pd.DatetimeIndex(m["date"])
    countries["DE"]["bund_noise"] = {
        "freq": "daily",
        "x": _xdec(dates), "dates": [str(d.date()) for d in dates],
        "noise": [round(float(v) * 1e4, 2) for v in m["noise"]]}       # decimal -> bp
    print(f"DE Bund noise gauge: {len(dates)} daily obs "
          f"{dates[0].date()}..{dates[-1].date()}, latest "
          f"{countries['DE']['bund_noise']['noise'][-1]:.1f} bp")


def main() -> None:
    data = build()
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    print(f"\nwrote {OUT}  {OUT.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
