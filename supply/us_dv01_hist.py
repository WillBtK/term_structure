"""US$ DV01 outstanding and held by SOMA, 1994 onward: the months before the CUSIP-level
sources ``supply.us_dv01`` uses (MSPD API from 2001-01, SOMA by CUSIP from 2003-07).

Outstanding (1994-01 .. 2000-12): the security-level ladder from ``supply.fetch_us_hist``
(auction rebuild to 1998-03, the statements themselves from 1998-04), priced exactly as
``supply.us_dv01`` prices the API ladder -- every security off the GSW nominal curve at
its own maturity, TIPS off the real curve from 2004 (so excluded before, as in the
existing series), FRNs at a stipulated near-zero duration.

SOMA held:

* **1996-06 .. 2003-06: H.4.1 maturity buckets.** The Fed's Treasury holdings by
  remaining maturity (within 15 days, 16-90 days, 91 days-1 year, 1-5, 5-10, over 10
  years), each bucket priced at the DV01 per dollar of par of the OUTSTANDING ladder's
  securities in that bucket that month:  ``SOMA DV01 = sum_b H_b * (DV01_b / par_b)``.
  The assumption is that within a bucket SOMA holds the market's average security --
  close to true for a portfolio built by rolling at auction and buying across the curve.
  Checked against the CUSIP-level SOMA DV01 over 2003-07 .. 2007-11 (``validate``): the
  bucket estimate's error is reported there and in validation/FINDINGS_dv01_1994.md.
  TIPS inside the buckets (the release footnotes their amount) are taken out of the
  over-1-year buckets pro rata before 2004, matching the CUSIP series, which carries no
  TIPS DV01 before the real curve is reliable.
* **1994-01 .. 1996-05: the Federal Reserve Bulletin.** The same maturity distribution,
  month-end, from Bulletin table 1.19 (FRASER scans with an OCR text layer;
  ``supply.fetch_us_hist.frb_buckets``). Each month-end is printed in three issues and a
  reading is kept only if its buckets add up to the printed total; every month from
  1994-01 has one. In 1996-06, where both exist, the coupon buckets agree to the dollar.
  The Bulletin's within-15-days bucket includes repo, which carries no DV01 to speak of.

The 3-month SOMA net-purchase flow needs CUSIP-level holdings to separate purchases from
the portfolio's own ageing across buckets, so it starts with them, in 2003-07+3m.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply.fetch_us import DATA
from supply.fetch_us_hist import (H41_EDGES, H41_KEYS, TEXT_START, auction_ladder,
                                  frb_buckets, h41_buckets, mspd_text_ladder)
from supply.us_dv01 import _curves_by_month, _dv01_ladder, flow_dv01

logger = logging.getLogger(__name__)

HIST_START = pd.Period("1994-01", "M")
API_START = pd.Period("2001-01", "M")
SOMA_CUSIP_START = pd.Period("2003-07", "M")
# CUSIP / bucket SOMA DV01, mean over 2003-07 .. 2007-11 (validate); the build recomputes
# it from the cached ladders and falls back to this
SOMA_BUCKET_CALIBRATION = 0.948
HIST_LADDER_PARQUET = DATA / "us_dv01_ladder_outstanding_hist.parquet"
HIST_DV01_PARQUET = DATA / "us_dv01_hist.parquet"


def outstanding_ladder_hist() -> pd.DataFrame:
    """Security-level month-end ladder, 1994-01 .. 2000-12 (auction rebuild, then the
    statements), in the MSPD-shaped columns ``us_dv01`` prices."""
    txt = mspd_text_ladder()
    early = auction_ladder(pd.period_range(HIST_START, TEXT_START - 1, freq="M"))
    cols = ["record_date", "cusip", "asset_class", "maturity_date", "coupon_pct", "amount"]
    return pd.concat([early[cols], txt[cols]], ignore_index=True)


def _price(ladder: pd.DataFrame, date_col: str = "record_date") -> pd.DataFrame:
    from us.fetch_gsw import load_gsw_params
    from us.tips import load_tips_params
    gsw, tips, lookup = _curves_by_month(load_gsw_params(), load_tips_params())
    return _dv01_ladder(ladder, date_col, gsw, tips, lookup)


def bucket_ratios(priced: pd.DataFrame, maturity: pd.Series) -> pd.DataFrame:
    """Per month, per H.4.1 bucket: DV01 per $1 of par ($mn/bp per $mn) of the outstanding
    securities whose remaining maturity falls in that bucket. TIPS (unpriced before 2004)
    and FRNs are left out; SOMA's TIPS are taken out of its buckets to match."""
    p = priced.copy()
    p["maturity_date"] = maturity.reindex(p.index).to_numpy()
    p = p[p["asset_class"].isin(["bill", "nominal"]) & p["dv01_mn"].notna()]
    yrs = (pd.to_datetime(p["maturity_date"]) - pd.to_datetime(p["date"])).dt.days / 365.25
    p["bucket"] = pd.cut(yrs, [H41_EDGES[k][0] for k in H41_KEYS] + [100.0],
                         labels=H41_KEYS, right=True)
    g = p.groupby([p["date"].dt.to_period("M"), "bucket"], observed=False)
    r = (g["dv01_mn"].sum() / g["amount"].sum()).unstack()
    r.index = r.index.to_timestamp("M")
    return r[H41_KEYS]


def _soma_buckets_ex_tips(h41: pd.DataFrame) -> pd.DataFrame:
    """H.4.1 buckets with SOMA's TIPS taken out of the over-1-year buckets pro rata
    (before 2004 only -- from 2004 the CUSIP series prices TIPS on the real curve)."""
    b = h41[H41_KEYS].copy()
    tips = h41.get("tips", pd.Series(np.nan, index=h41.index)).copy()
    # the release stops footnoting SOMA's TIPS at 2002-12; bridge to the CUSIP file's
    # first reading (2003-07) so the last H.4.1 months do not price ~$12bn of TIPS as
    # nominal coupons
    anchor = _soma_tips_first()
    if anchor is not None and anchor[0] in tips.index:
        tips.loc[anchor[0]] = anchor[1]
    tips = tips.interpolate(limit_area="inside").fillna(0.0)
    tips[h41.index >= pd.Timestamp("2004-01-01")] = 0.0
    long = ["b5y", "b10y", "b10p"]
    share = b[long].div(b[long].sum(axis=1), axis=0)
    b[long] = b[long].sub(share.mul(tips, axis=0))
    return b


def _soma_tips_first():
    """(month-end, $mn) of SOMA's TIPS in the first CUSIP-level month, if cached."""
    try:
        from supply.fetch_us import SOMA_RAW, fetch_soma_all
        if not SOMA_RAW.exists():
            return None
        s = fetch_soma_all()
        t = s[s["asset_class"] == "tips"].groupby("asof")["amount"].sum().sort_index()
        return (t.index[0].to_period("M").to_timestamp("M"), float(t.iloc[0])) if len(t) else None
    except Exception:                                               # noqa: BLE001
        return None


def soma_from_buckets(ratios: pd.DataFrame, h41: pd.DataFrame) -> pd.Series:
    b = _soma_buckets_ex_tips(h41)
    idx = b.index.intersection(ratios.index)
    return (b.loc[idx] * ratios.loc[idx]).sum(axis=1, min_count=6).rename("soma_dv01_mn")


def build_hist(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(monthly frame 1994-01 .. 2003-06, priced outstanding ladder 1994-2000)."""
    if not refresh and HIST_DV01_PARQUET.exists() and HIST_LADDER_PARQUET.exists():
        return pd.read_parquet(HIST_DV01_PARQUET), pd.read_parquet(HIST_LADDER_PARQUET)
    raw = outstanding_ladder_hist()
    logger.info("Pricing the 1994-2000 outstanding ladder (%d security-months)...", len(raw))
    priced = _price(raw)
    priced.to_parquet(HIST_LADDER_PARQUET)
    out = priced.groupby("date")["dv01_mn"].sum(min_count=1)
    out.index = out.index.to_period("M").to_timestamp("M")
    supply = flow_dv01(priced, months=3)
    supply.index = pd.DatetimeIndex(supply.index).to_period("M").to_timestamp("M")
    # SOMA: bucket ratios need the outstanding ladder for every SOMA month to 2003-06 --
    # 1994-2000 from the historical ladder, 2001-2003 from the API ladder
    mats = raw[["record_date", "cusip", "maturity_date"]].drop_duplicates(["record_date", "cusip"])
    m = priced.merge(mats, left_on=["date", "cusip"], right_on=["record_date", "cusip"],
                     how="left")
    ratios_hist = bucket_ratios(priced, m["maturity_date"])
    ratios_api = api_bucket_ratios(end=SOMA_CUSIP_START - 1)
    ratios = pd.concat([ratios_hist, ratios_api]).sort_index()
    ratios = ratios[~ratios.index.duplicated()]
    h41 = h41_buckets()
    frb = frb_buckets()
    frb = frb[(frb.index >= HIST_START.to_timestamp()) & (frb.index < h41.index.min())]
    buckets = pd.concat([frb[H41_KEYS], h41[H41_KEYS + ["tips"]]]).sort_index()
    soma_b = soma_from_buckets(ratios, buckets)
    soma_b = soma_b[soma_b.index < SOMA_CUSIP_START.to_timestamp()]
    # Within each maturity bucket SOMA held lower-duration issues than the market average
    # (more seasoned, higher-coupon paper), so the bucket estimate reads high. Scale by the
    # overlap's average CUSIP/bucket ratio (``validate``).
    try:
        v = validate()
        cal = float((v["cusip"] / v["buckets"]).mean())
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("SOMA bucket calibration unavailable (%s); using %.3f",
                       exc, SOMA_BUCKET_CALIBRATION)
        cal = SOMA_BUCKET_CALIBRATION
    logger.info("SOMA bucket calibration: x%.3f", cal)
    soma_b = soma_b * cal
    idx = pd.date_range(HIST_START.to_timestamp("M"),
                        (SOMA_CUSIP_START - 1).to_timestamp("M"), freq="M")
    df = pd.DataFrame(index=idx)
    df["outstanding_dv01_mn"] = out.reindex(idx)
    df["soma_dv01_mn"] = soma_b.reindex(idx)
    df["supply_3m_dv01_mn"] = supply.reindex(idx)
    df["soma_net_3m_dv01_mn"] = np.nan
    df["outstanding_method"] = np.where(idx < TEXT_START.to_timestamp(), "auction_rebuild",
                                        np.where(idx < API_START.to_timestamp(),
                                                 "mspd_statement", "mspd_api"))
    df["soma_method"] = np.where(idx < h41.index.min(), "fed_bulletin", "h41_buckets")
    df["soma_calibration"] = cal
    df.index.name = "date"
    # the API months' outstanding/supply come from supply.us_dv01 itself
    df.loc[df.index >= API_START.to_timestamp(), ["outstanding_dv01_mn", "supply_3m_dv01_mn"]] = np.nan
    df.to_parquet(HIST_DV01_PARQUET)
    return df, priced


def api_bucket_ratios(start=API_START, end=pd.Period("2007-11", "M")) -> pd.DataFrame:
    """Bucket ratios from the API-era priced ladder (``supply.us_dv01``'s cache)."""
    from supply.fetch_us import fetch_mspd_all
    from supply.us_dv01 import OUT_LADDER_PARQUET
    priced = pd.read_parquet(OUT_LADDER_PARQUET)
    priced = priced[(priced["date"] >= start.to_timestamp())
                    & (priced["date"] <= end.to_timestamp("M"))].reset_index(drop=True)
    mats = fetch_mspd_all()[["record_date", "cusip", "maturity_date"]]
    mats = mats.drop_duplicates(["record_date", "cusip"])
    m = priced.merge(mats, left_on=["date", "cusip"], right_on=["record_date", "cusip"],
                     how="left")
    return bucket_ratios(priced, m["maturity_date"])


def validate(start: str = "2003-07", end: str = "2007-11") -> pd.DataFrame:
    """The bucket method against the CUSIP-level SOMA DV01 where both exist, like for
    like: bills and nominal coupons only on both sides (SOMA's TIPS, by bucket from the
    CUSIP file, taken out of the H.4.1 buckets), callables priced to call throughout."""
    from supply.fetch_us import fetch_soma_all
    from supply.us_dv01 import HELD_LADDER_PARQUET
    held = pd.read_parquet(HELD_LADDER_PARQUET)
    held = held[(held["date"] >= pd.Period(start, "M").to_timestamp())
                & (held["date"] <= pd.Period(end, "M").to_timestamp("M"))].reset_index(drop=True)
    mats = fetch_soma_all()[["asof", "cusip", "maturity_date"]].drop_duplicates(["asof", "cusip"])
    held["maturity_date"] = held.merge(mats, left_on=["date", "cusip"],
                                       right_on=["asof", "cusip"], how="left")["maturity_date"].to_numpy()
    held["month"] = held["date"].dt.to_period("M").dt.to_timestamp("M")
    yrs = (pd.to_datetime(held["maturity_date"]) - held["date"]).dt.days / 365.25
    held["bucket"] = pd.cut(yrs, [H41_EDGES[k][0] for k in H41_KEYS] + [100.0], labels=H41_KEYS)
    nominal = held["asset_class"].isin(["bill", "nominal"])
    truth = held[nominal].groupby("month")["dv01_mn"].sum()
    tips = (held[held["asset_class"] == "tips"].groupby(["month", "bucket"], observed=False)
            ["amount"].sum().unstack().reindex(columns=H41_KEYS).fillna(0.0))
    ratios = api_bucket_ratios(start=pd.Period(start, "M"), end=pd.Period(end, "M"))
    h41 = h41_buckets()[H41_KEYS]
    idx = truth.index.intersection(ratios.index).intersection(h41.index)
    H = (h41.loc[idx] - tips.reindex(idx).fillna(0.0)).clip(lower=0.0)
    est = (H * ratios.loc[idx]).sum(axis=1)
    j = pd.DataFrame({"cusip": truth.loc[idx], "buckets": est})
    j["err_pct"] = 100 * (j["buckets"] / j["cusip"] - 1)
    return j


def splice_hist(df: pd.DataFrame, refresh: bool = False) -> pd.DataFrame:
    """Extend ``supply.us_dv01.aggregate``'s monthly frame (2001+ outstanding, 2003-07+
    SOMA) back to 1994 with the historical estimates, and tag every month's source."""
    from supply.us_dv01 import OUT_LADDER_PARQUET
    hist, priced_hist = build_hist(refresh=refresh)
    out = df.copy()
    out["outstanding_method"] = np.where(out["outstanding_dv01_mn"].notna(), "mspd_api", None)
    out["soma_method"] = np.where(out["soma_dv01_mn"].notna(), "cusip", None)
    idx = hist.index.union(out.index)
    out = out.reindex(idx)
    early_out = idx < API_START.to_timestamp()
    early_soma = idx < SOMA_CUSIP_START.to_timestamp()
    for c in ("outstanding_dv01_mn", "supply_3m_dv01_mn", "outstanding_method"):
        out.loc[early_out, c] = hist.loc[idx[early_out], c].to_numpy()
    out["soma_calibration"] = np.nan
    for c in ("soma_dv01_mn", "soma_method", "soma_calibration"):
        out.loc[early_soma, c] = hist.loc[idx[early_soma], c].to_numpy()
    # the 3m supply flow across the 2000-12 / 2001-01 seam: statement and API ladders
    # carry identical CUSIPs and amounts (validated), so one flow over both is exact
    api = pd.read_parquet(OUT_LADDER_PARQUET)
    seam = pd.concat([priced_hist[priced_hist["date"] >= pd.Timestamp("2000-10-01")]
                      [api.columns.intersection(priced_hist.columns)],
                      api[api["date"] < pd.Timestamp("2001-04-01")]], ignore_index=True)
    f = flow_dv01(seam, months=3)
    f.index = pd.DatetimeIndex(f.index).to_period("M").to_timestamp("M")
    for d, v in f.items():
        if pd.Timestamp("2001-01-01") <= d < pd.Timestamp("2001-04-01"):
            out.loc[d, "supply_3m_dv01_mn"] = v
    out.index.name = "date"
    return out
