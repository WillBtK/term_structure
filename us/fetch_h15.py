"""Timely H.15 nowcast extension for the US curves.

The Fed Board's GSW zero-coupon files (``feds200628`` nominal, ``feds200805`` TIPS) are
the model's native input, but they publish on a ~1-week lag. The **constant-maturity
(CMT) par curve** closes that gap. It reaches us by two routes carrying identical
numbers at different speeds: **Treasury's own XML feed** posts same-day (~15:30-16:00
ET), while the Fed H.15 release mirrored on FRED as ``DGS*`` (nominal) / ``DFII*``
(real) lands one business day later. We read Treasury for the tail and FRED for the
1962+ history (the XML serves one month per request), cross-checking the two on their
overlap. Treasury is an accelerator, never a dependency: if it is unreachable the panel
degrades to exactly the FRED-only behaviour. H.15 is a *constant-maturity par* curve built from the
most-recently-auctioned (on-the-run) securities, so its LEVEL differs from GSW's
off-the-run zero curve (par-vs-zero + on-the-run richness: empirically +7bp at 10y,
+18bp at 30y, −7bp at 2y, and growing with maturity). We therefore never substitute
H.15 levels; we use only its day-to-day CHANGE to nowcast the days GSW has not yet
published, anchored to GSW's last level:

    nowcast_zero(tau, t) = GSW_zero(tau, t0) + [ H15(tau, t) - H15(tau, t0) ]   (t > t0)

where ``t0`` is GSW's last date. The GSW anchor absorbs the par/zero/on-the-run wedge;
only the H.15 increment carries information. Over the ~1-3 day gap this is validated —
GSW and H.15 daily changes agree to <1.5bp (std) at every tenor — so the nowcast adds
~1bp of tenor-level noise, negligible for term premia. The nowcast is **provisional**:
it is flagged in the viz and is overwritten (self-heals) the moment the real GSW batch
lands. H.15 covers every nominal grid tenor exactly (3M..30Y); the real curve (DFII)
covers 5-30y, so 2y/3y real stay ragged (handled by the EKF's per-date masking) until
the TIPS GSW batch updates.

FRED access reuses ``validation.fred_api.fred_series`` (keyed API or keyless fallback).
Cached to ``us/data`` with graceful fallback so a fetch outage keeps the last curve.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from us.config import COMMON_GRID

logger = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parent / "data"

# grid tenor (years) -> FRED H.15 series id. Nominal covers the whole COMMON_GRID;
# real (DFII) starts at 5y.
H15_NOMINAL = {0.25: "DGS3MO", 0.5: "DGS6MO", 1.0: "DGS1", 2.0: "DGS2", 3.0: "DGS3",
               5.0: "DGS5", 7.0: "DGS7", 10.0: "DGS10", 20.0: "DGS20", 30.0: "DGS30"}
H15_REAL = {5.0: "DFII5", 7.0: "DFII7", 10.0: "DFII10", 20.0: "DFII20", 30.0: "DFII30"}
_MAPS = {"nominal": H15_NOMINAL, "real": H15_REAL}


# Treasury's own XML feed carries the SAME constant-maturity numbers a full business
# day before the FRED/H.15 mirror (verified: on 18 Aug 2026 every tenor matched to the
# published 0.01% precision). Field names differ per curve: BC_* nominal, TC_* real.
TREASURY_NOMINAL = {0.25: "BC_3MONTH", 0.5: "BC_6MONTH", 1.0: "BC_1YEAR", 2.0: "BC_2YEAR",
                    3.0: "BC_3YEAR", 5.0: "BC_5YEAR", 7.0: "BC_7YEAR", 10.0: "BC_10YEAR",
                    20.0: "BC_20YEAR", 30.0: "BC_30YEAR"}
TREASURY_REAL = {5.0: "TC_5YEAR", 7.0: "TC_7YEAR", 10.0: "TC_10YEAR",
                 20.0: "TC_20YEAR", 30.0: "TC_30YEAR"}
_TREASURY = {"nominal": ("daily_treasury_yield_curve", TREASURY_NOMINAL),
             "real": ("daily_treasury_real_yield_curve", TREASURY_REAL)}
_ODATA = "{http://schemas.microsoft.com/ado/2007/08/dataservices}"
# the <properties> wrapper sits in the METADATA namespace; only the fields inside
# it are in the dataservices one — mixing the two silently yields zero rows.
_ODATA_M = "{http://schemas.microsoft.com/ado/2007/08/dataservices/metadata}"

# Treasury quotes par yields to 0.01%; agreement with FRED is exact at that precision,
# so anything above half a basis point means the two sources genuinely disagree.
_XCHECK_TOL = 0.00005


def _treasury_month(instrument: str, month: str, timeout: int = 60) -> pd.DataFrame:
    """One month of Treasury par yields as a DECIMAL panel (index=date, cols=tenor).

    ``month`` is ``YYYYMM``. Raises on transport/parse failure — callers decide whether
    a Treasury outage is fatal (it is not; FRED remains the fallback)."""
    import requests
    import xml.etree.ElementTree as ET
    from us.config import TREASURY_XML_URL

    data_kind, tenmap = _TREASURY[instrument]
    resp = requests.get(TREASURY_XML_URL, timeout=timeout, params={
        "data": data_kind, "field_tdr_date_value_month": month})
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    rows = {}
    for props in root.iter(f"{_ODATA_M}properties"):
        date_el = props.find(f"{_ODATA}NEW_DATE")
        if date_el is None or not (date_el.text or "").strip():
            continue
        row = {}
        for ten, field in tenmap.items():
            el = props.find(f"{_ODATA}{field}")
            txt = (el.text or "").strip() if el is not None else ""
            if txt:                                   # Treasury blanks a tenor it did not quote
                row[ten] = float(txt) / 100.0         # percent -> decimal
        if row:
            rows[pd.Timestamp(date_el.text[:10])] = row
    if not rows:
        raise ValueError(f"Treasury {instrument} {month}: no dated observations")
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _treasury_tail(instrument: str, months: list[str] | None = None) -> pd.DataFrame:
    """Recent Treasury par yields (current + previous month by default), or an empty
    frame if the feed is unreachable — Treasury is an accelerator, never a dependency."""
    if months is None:
        today = pd.Timestamp.today().normalize()
        months = [(today - pd.DateOffset(months=1)).strftime("%Y%m"), today.strftime("%Y%m")]
    frames = []
    for m in months:
        try:
            frames.append(_treasury_month(instrument, m))
        except Exception as exc:                       # noqa: BLE001
            logger.warning("Treasury %s %s fetch failed: %s", instrument, m, exc)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _splice_treasury(panel: pd.DataFrame, tail: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """Extend ``panel`` (FRED history) with Treasury dates that FRED has not published.

    Treasury only ever ADDS dates here. Where the two overlap we cross-check rather than
    overwrite: they are the same series, so a disagreement is a signal that one of the
    two feeds has changed meaning — worth a loud warning, not a silent revision."""
    if tail.empty or panel.empty:
        return panel
    shared = [c for c in tail.columns if c in panel.columns]
    overlap = panel.index.intersection(tail.index)
    if len(overlap) and shared:
        diff = (panel.loc[overlap, shared] - tail.loc[overlap, shared]).abs().max().max()
        if pd.notna(diff) and diff > _XCHECK_TOL:
            logger.warning("Treasury/FRED %s disagree by up to %.1f bp on %d shared day(s) "
                           "— using FRED for the overlap; check whether a feed changed",
                           instrument, diff * 1e4, len(overlap))
        else:
            logger.info("Treasury/FRED %s agree on %d shared day(s) (max %.2f bp)",
                        instrument, len(overlap), (diff or 0.0) * 1e4)
    fresh = tail.loc[tail.index > panel.index.max()]
    if fresh.empty:
        return panel
    fresh = fresh.reindex(columns=panel.columns)       # ragged for tenors Treasury omits
    logger.info("Treasury %s: +%d day(s) ahead of FRED (%s..%s)", instrument, len(fresh),
                fresh.index.min().date(), fresh.index.max().date())
    return pd.concat([panel, fresh]).sort_index()


def _cache(instrument: str) -> Path:
    return DATA / f"h15_{instrument}.parquet"


def load_h15_panel(instrument: str = "nominal", refresh: bool = False) -> pd.DataFrame:
    """H.15 CMT curve as a DECIMAL panel on the grid tenors (columns = years, float),
    business-daily. Cached; on a fetch failure falls back to the committed cache."""
    cache = _cache(instrument)
    if not refresh and cache.exists():
        return pd.read_parquet(cache)
    tenmap = _MAPS[instrument]
    from validation.fred_api import fred_series
    cols = {}
    for ten, sid in tenmap.items():
        try:
            cols[ten] = fred_series(sid) / 100.0          # percent -> decimal
        except Exception as exc:                          # noqa: BLE001
            logger.warning("H.15: %s (%.2gy) fetch failed: %s", sid, ten, exc)
    if not cols:
        if cache.exists():
            logger.warning("H.15 %s: all fetches failed -> using cache", instrument)
            return pd.read_parquet(cache)
        raise RuntimeError(f"H.15 {instrument}: no series fetched and no cache")
    panel = pd.DataFrame(cols).sort_index()
    panel = panel[sorted(panel.columns)]
    panel.index = pd.to_datetime(panel.index)
    panel = _splice_treasury(panel, _treasury_tail(instrument), instrument)
    DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache)
    logger.info("H.15 %s: %d tenors, %s..%s (%d days)", instrument, panel.shape[1],
                panel.index.min().date(), panel.index.max().date(), len(panel))
    return panel


def nowcast_extend(panel: pd.DataFrame, instrument: str = "nominal",
                   refresh: bool = False, max_gap_days: int = 20):
    """Append H.15-increment nowcast rows to ``panel`` for business days after its last
    date, anchored to the panel's last GSW level. Returns ``(extended_panel, dates)``
    where ``dates`` is the list of nowcast dates (ISO strings; empty if none / on error).

    Only tenors present in BOTH ``panel`` and the H.15 map are nowcast; other tenors stay
    NaN on the new rows (ragged — handled downstream). ``max_gap_days`` guards against
    splicing across a stale GSW cache (if GSW is >this far behind H.15, skip and log)."""
    try:
        base = panel.dropna(how="all")
        if base.empty:
            return panel, []
        t0 = base.index.max()
        h15 = load_h15_panel(instrument, refresh=refresh)
        future = h15.index[h15.index > t0]
        if len(future) == 0:
            return panel, []                                # nothing newer than GSW
        # anchor at the last H.15 date on/before t0 (usually == t0)
        prior = h15.index[h15.index <= t0]
        if len(prior) == 0:
            logger.warning("H.15 %s nowcast: no H.15 obs on/before GSW's last date", instrument)
            return panel, []
        a0 = prior.max()
        if (future.max() - t0).days > max_gap_days:
            logger.warning("H.15 %s nowcast: GSW (%s) lags latest H.15 (%s) by >%dd -> skip "
                           "(too long a window to splice across an on-the-run roll)",
                           instrument, t0.date(), future.max().date(), max_gap_days)
            return panel, []
        tens = [t for t in base.columns if t in h15.columns]
        if not tens:
            return panel, []
        offset = base.loc[t0, tens].astype(float) - h15.loc[a0, tens].astype(float)
        rows = h15.loc[future, tens].astype(float) + offset                 # broadcast per tenor
        rows = rows.reindex(columns=base.columns)                           # ragged for missing tenors
        rows = rows.dropna(how="all")
        if rows.empty:
            return panel, []
        out = pd.concat([panel, rows]).sort_index()
        out = out[~out.index.duplicated(keep="first")]                      # keep GSW where it overlaps
        dates = [str(d.date()) for d in rows.index]
        logger.info("H.15 %s nowcast: +%d day(s) %s..%s over %d tenors (anchor %s)",
                    instrument, len(dates), dates[0], dates[-1], len(tens), t0.date())
        return out, dates
    except Exception as exc:                                                # noqa: BLE001
        logger.warning("H.15 %s nowcast failed (%s) -> GSW only", instrument, exc)
        return panel, []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for inst in ("nominal", "real"):
        p = load_h15_panel(inst, refresh=True)
        print(f"H.15 {inst}: {p.index.min().date()}..{p.index.max().date()} "
              f"tenors={list(p.columns)} latest={p.iloc[-1].round(4).to_dict()}")


if __name__ == "__main__":
    main()
