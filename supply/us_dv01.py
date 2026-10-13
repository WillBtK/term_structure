"""US$ DV01: interest-rate risk outstanding, held by the Federal Reserve, and the
FLOW that adds to each -- monthly, from the same MSPD + SOMA CUSIP-level ladders
``supply.us_tye`` uses.

DV01 (dollar value of a 1bp move) is the right unit for "how much duration risk must
the market absorb" -- unlike par amount, it prices a bill and a thirty-year bond as
what they actually are. Four series:

1. **Outstanding** -- DV01 of the full marketable ladder: bills, nominal notes/bonds,
   TIPS, FRNs.
2. **Held by SOMA** -- DV01 of the Federal Reserve's own portfolio, same ladder.
3. **Rolling 3-month net supply** -- the DV01 the market absorbs from PRIMARY ISSUANCE
   net of redemptions/buybacks over the trailing 3 months.
4. **Rolling 3-month net SOMA purchases** -- the DV01 the Fed adds to or removes from
   its own book over the trailing 3 months.

**(3) and (4) are NOT the change in (1)/(2).** The stock (1)/(2) moves every month even
with zero primary-market activity, because every outstanding bond's own duration drifts
with rolldown and with the level of rates. (3)/(4) isolate the QUANTITY change only:
for each CUSIP, ``[amount(t) - amount(t-3mo)] * (DV01 per unit of that CUSIP)`` --
priced at the CUSIP's OWN duration at whichever end of the window it exists (see
``flow_dv01``). A continuing bond with an unchanged par amount contributes exactly zero to
the flow, however much its DV01-per-unit has moved since; a newly issued or reopened
CUSIP contributes its full incremental par at TODAY's duration; a CUSIP that vanished
from the ladder (matured, or bought back to zero) contributes negatively at the
duration it had 3 months ago -- which, for anything that fully matures out of the
snapshot, is necessarily small (it cannot have had more than ~3 months left to run and
still be gone by the second date).

Two deliberate approximations, both because MSPD/SOMA are par-amount snapshots, not
priced trade tickets:

* **Curve-implied yield, not a market price.** Every security is priced/duration-ed off
  the GSW nominal (or TIPS real) zero curve AT ITS OWN MATURITY, the same simplification
  ``supply.tenyear`` already uses for the TYE share -- not its own quoted market yield,
  which this dataset does not carry.
* **FRNs get a stipulated ~0 duration, not a curve-priced one.** A floating-rate note
  reprices to the market every quarter, so pricing it as a fixed-coupon bond at today's
  rate to its stated maturity would hand it several years of spurious duration. It is
  priced instead as ``amount * FRN_EFFECTIVE_DURATION * 1bp`` -- a placeholder for "the
  time until the next reset," which MSPD/SOMA do not publish per-CUSIP. At <3% of the
  ladder and near-zero duration either way, this cannot move the totals materially, but
  it is a stated simplification, not a measurement.

TIPS are priced off the REAL curve (``us.tips``), which starts existing in 1999 and is
flagged reliable only from 2004-01 (``REAL_CURVE_START``, matching ``us.config
.REAL_SAMPLE_START``); TIPS duration is NaN and excluded from every sum before that.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.bond_math import modified_duration, price_from_ytm
from supply.fetch_us import DATA, fetch_mspd_all, fetch_soma_all

logger = logging.getLogger(__name__)
DV01_PARQUET = DATA / "us_dv01.parquet"
OUT_LADDER_PARQUET = DATA / "us_dv01_ladder_outstanding.parquet"  # per-CUSIP, for the flow calc
HELD_LADDER_PARQUET = DATA / "us_dv01_ladder_soma.parquet"

FRN_EFFECTIVE_DURATION = 0.125     # years -- see module docstring
REAL_CURVE_START = pd.Timestamp("2004-01-01")


def _nominal_curve_for(params_row) -> callable:
    """Zero curve in DECIMAL from a row of GSW nominal Svensson parameters."""
    from us.reconstruct import svensson_yield
    b0, b1, b2, b3 = (float(params_row[c]) for c in ("BETA0", "BETA1", "BETA2", "BETA3"))
    t1, t2 = float(params_row["TAU1"]), float(params_row["TAU2"])

    def curve(t):
        t = np.maximum(np.asarray(t, dtype=float), 1.0 / 365.25)
        return np.asarray(svensson_yield(t, b0, b1, b2, b3, t1, t2), dtype=float) / 100.0
    return curve


def _real_curve_for(params_row) -> callable:
    """Zero curve in DECIMAL from a row of TIPS (real) Svensson parameters.

    Mirrors ``us.tips.reconstruct_real`` for a single date/row rather than a panel --
    same NS-fallback (BETA3/TAU2 blank before the Svensson term was fitted).
    """
    from us.reconstruct import svensson_yield
    b0, b1, b2 = (float(params_row[c]) for c in ("BETA0", "BETA1", "BETA2"))
    has_sv = pd.notna(params_row.get("BETA3")) and pd.notna(params_row.get("TAU2")) \
        and float(params_row["TAU2"]) > 0
    b3 = float(params_row["BETA3"]) if has_sv else 0.0
    t1 = float(params_row["TAU1"])
    t2 = float(params_row["TAU2"]) if has_sv else 1.0

    def curve(t):
        t = np.maximum(np.asarray(t, dtype=float), 1.0 / 365.25)
        return np.asarray(svensson_yield(t, b0, b1, b2, b3, t1, t2), dtype=float) / 100.0
    return curve


def security_dv01(ladder: pd.DataFrame, asof, nominal_curve,
                  real_curve: "callable | None") -> pd.Series:
    """Per-security DV01 in $mn per 1bp, aligned to ``ladder``'s index.

    ``ladder`` needs ``maturity_date``, ``coupon_pct``, ``amount`` ($mn, par) and
    ``asset_class`` in {"nominal", "bill", "tips", "frn"}. ``real_curve=None`` (no
    reliable real curve for this date) makes every TIPS row NaN, not zero -- so it is
    excluded from a sum rather than silently misstated as riskless.
    """
    asof = pd.Timestamp(asof)
    yrs = (pd.to_datetime(ladder["maturity_date"]) - asof).dt.days / 365.25
    amt = ladder["amount"].to_numpy(dtype=float)
    coupons = ladder["coupon_pct"].to_numpy(dtype=float)
    ac = ladder["asset_class"].to_numpy()
    out = np.full(len(ladder), np.nan)
    for i, (t, c, mat, a, cls) in enumerate(zip(
            yrs.to_numpy(), coupons, pd.to_datetime(ladder["maturity_date"]), amt, ac)):
        if not np.isfinite(t) or t <= 0 or not np.isfinite(a):
            continue
        if cls == "frn":
            out[i] = a * FRN_EFFECTIVE_DURATION * 0.0001
            continue
        curve = real_curve if cls == "tips" else nominal_curve
        if curve is None:
            continue
        z = float(curve(t))
        ytm = 2.0 * (np.exp(z / 2.0) - 1.0)          # continuous zero -> semiannual BEY
        moddur = modified_duration(asof, mat, float(c), ytm)
        price = price_from_ytm(asof, mat, float(c), ytm)
        if not np.isfinite(moddur) or not np.isfinite(price):
            continue
        out[i] = a * (price / 100.0) * moddur * 0.0001
    return pd.Series(out, index=ladder.index, name="dv01_mn")


def to_call(ladder: pd.DataFrame) -> pd.DataFrame:
    """Callable bonds priced to their first call date, the date Treasury redeemed every
    one of them (``supply.fetch_us.CALLABLE_FIRST_CALL``). A no-op for the SOMA ladder,
    whose file already carries the call date."""
    from supply.fetch_us import CALLABLE_FIRST_CALL
    calls = ladder["cusip"].map(CALLABLE_FIRST_CALL)
    if calls.isna().all():
        return ladder
    out = ladder.copy()
    call = pd.to_datetime(calls)
    out["maturity_date"] = call.where(call.notna() & (call < pd.to_datetime(out["maturity_date"])),
                                      pd.to_datetime(out["maturity_date"]))
    return out


def _curves_by_month(gsw: pd.DataFrame, tips: pd.DataFrame):
    """{period('M') -> (nominal_curve, real_curve_or_None)}, most-recent-as-of-month-end."""
    gsw = gsw.dropna(subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
    gsw = gsw[gsw["TAU2"] > 0]
    gsw.index = pd.DatetimeIndex(gsw.index)
    tips = tips.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])
    tips = tips[tips["TAU1"] > 0]
    tips.index = pd.DatetimeIndex(tips.index)

    def _lookup(asof, params):
        prior = params.index[params.index <= asof]
        return params.loc[prior.max()] if len(prior) else None
    return gsw, tips, _lookup


def _dv01_ladder(ladder_all: pd.DataFrame, date_col: str, gsw, tips, _lookup) -> pd.DataFrame:
    """Per-(date, cusip) DV01, long-form, across every month in ``ladder_all``."""
    ladder_all = ladder_all.copy()
    ladder_all["ym"] = ladder_all[date_col].dt.to_period("M")
    frames = []
    for ym, sub in ladder_all.groupby("ym"):
        asof = sub[date_col].max()
        gsw_row = _lookup(asof, gsw)
        if gsw_row is None:
            continue
        tips_row = _lookup(asof, tips) if asof >= REAL_CURVE_START else None
        nom_curve = _nominal_curve_for(gsw_row)
        real_curve = _real_curve_for(tips_row) if tips_row is not None else None
        dv01 = security_dv01(to_call(sub), asof, nom_curve, real_curve)
        frames.append(pd.DataFrame({
            "date": asof, "cusip": sub["cusip"].to_numpy(), "asset_class": sub["asset_class"].to_numpy(),
            "amount": sub["amount"].to_numpy(), "dv01_mn": dv01.to_numpy(),
        }))
        if len(frames) % 36 == 0:
            logger.info("  %s  n=%d", asof.date(), len(sub))
    return pd.concat(frames, ignore_index=True)


def flow_dv01(ladder: pd.DataFrame, months: int = 3) -> pd.Series:
    """Rolling ``months``-month net DV01 flow: for each date t with a match at t-months,
    ``sum_i [A_i(t) - A_i(t-months)] * ratio_i``, ``ratio_i`` = DV01-per-unit-par at t if
    the CUSIP exists then, else at (t-months) if it existed there but not at t. See the
    module docstring for why this -- not the change in the DV01 stock -- is "supply."
    """
    piv_a = ladder.pivot_table(index="cusip", columns="date", values="amount", aggfunc="sum")
    piv_d = ladder.pivot_table(index="cusip", columns="date", values="dv01_mn", aggfunc="sum")
    ratio = (piv_d / piv_a.replace(0, np.nan))
    dates = sorted(piv_a.columns)
    per = pd.PeriodIndex(dates, freq="M")
    lag_of = dict(zip(per, dates))
    out = {}
    for d, p in zip(dates, per):
        p_lag = p - months
        if p_lag not in lag_of:
            continue
        d_lag = lag_of[p_lag]
        a_t = piv_a[d].fillna(0.0)
        a_l = piv_a[d_lag].fillna(0.0)
        r_t = ratio[d]
        r_l = ratio[d_lag]
        r_used = r_t.where(a_t > 0, r_l)                 # exists now -> today's ratio,
        flow = (a_t - a_l) * r_used                       # else -> its ratio 3mo ago
        out[d] = float(flow.sum(skipna=True))
    return pd.Series(out).sort_index()


def _monthly_last(s: pd.Series) -> pd.Series:
    """Collapse a date-indexed series to one value per calendar month (its last obs)."""
    out = s.copy()
    out.index = pd.PeriodIndex(pd.DatetimeIndex(out.index), freq="M")
    return out.groupby(level=0).last()


def aggregate(out_ladder: pd.DataFrame, held_ladder: pd.DataFrame) -> pd.DataFrame:
    """The two priced ladders -> one monthly 4-column frame.

    MSPD and SOMA snapshot on DIFFERENT calendar days within a month -- MSPD's
    ``record_date`` is Treasury's own month-end, SOMA's ``asof`` is whichever
    Wednesday happened to close out that reporting month. Joining by exact date would
    union the two into separate rows, each missing the OTHER book's columns entirely
    (an all-NaN "latest" row is the symptom). Collapse each series to one
    calendar-month value first, then join on that.
    """
    out_tot = _monthly_last(out_ladder.groupby("date")["dv01_mn"].sum(min_count=1))
    held_tot = _monthly_last(held_ladder.groupby("date")["dv01_mn"].sum(min_count=1))
    supply_flow = _monthly_last(flow_dv01(out_ladder, months=3))
    soma_flow = _monthly_last(flow_dv01(held_ladder, months=3))
    df = pd.DataFrame({
        "outstanding_dv01_mn": out_tot,
        "soma_dv01_mn": held_tot,
        "supply_3m_dv01_mn": supply_flow,
        "soma_net_3m_dv01_mn": soma_flow,
        "supply_1m_dv01_mn": _monthly_last(flow_dv01(out_ladder, months=1)),
        "soma_net_1m_dv01_mn": _monthly_last(flow_dv01(held_ladder, months=1)),
    }).sort_index()
    df.index = df.index.to_timestamp("M")     # month-end, matching MSPD's own convention
    df.index.name = "date"
    return df


def build(refresh: bool = False) -> pd.DataFrame:
    if not refresh and DV01_PARQUET.exists() and OUT_LADDER_PARQUET.exists() \
            and HELD_LADDER_PARQUET.exists():
        return pd.read_parquet(DV01_PARQUET)

    if not refresh and OUT_LADDER_PARQUET.exists() and HELD_LADDER_PARQUET.exists():
        # The (expensive) per-security pricing is already cached -- only the cheap
        # aggregation below needs re-running, e.g. after fixing that step itself.
        out_ladder = pd.read_parquet(OUT_LADDER_PARQUET)
        held_ladder = pd.read_parquet(HELD_LADDER_PARQUET)
    else:
        from us.fetch_gsw import load_gsw_params
        from us.tips import load_tips_params

        out_all = fetch_mspd_all(refresh=refresh)
        held_all = fetch_soma_all(refresh=refresh)
        gsw, tips, _lookup = _curves_by_month(load_gsw_params(), load_tips_params())

        logger.info("Pricing the outstanding ladder...")
        out_ladder = _dv01_ladder(out_all, "record_date", gsw, tips, _lookup)
        logger.info("Pricing the SOMA ladder...")
        held_ladder = _dv01_ladder(held_all, "asof", gsw, tips, _lookup)

        # Written SEPARATELY, not concatenated into one frame: pandas 2.1.3 hits a
        # block-concatenation bug (a numpy axis mismatch inside
        # DatetimeArray._concat_same_type) when stacking two large, independently-built
        # ladders of different lengths -- not reproducible on small synthetic frames,
        # so a library issue rather than a data one. Nothing downstream needs them
        # combined; each is used on its own in ``aggregate``.
        DATA.mkdir(parents=True, exist_ok=True)
        out_ladder.to_parquet(OUT_LADDER_PARQUET)
        held_ladder.to_parquet(HELD_LADDER_PARQUET)

    df = aggregate(out_ladder, held_ladder)
    # back to 1994: statement PDFs, auction records and H.4.1 maturity buckets
    from supply.us_dv01_hist import splice_hist
    df = splice_hist(df, refresh=refresh)
    df.to_parquet(DV01_PARQUET)
    return df


def _main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = build(refresh="--refresh" in sys.argv)
    print(f"\n{len(d)} monthly obs {d.index.min().date()}..{d.index.max().date()}\n")
    print("           outstanding    SOMA held    3m net supply   3m net SOMA buy   ($mn/bp)")
    for yr in range(d.index.year.min(), d.index.year.max() + 1):
        w = d[d.index.year == yr]
        if w.empty:
            continue
        print(f"  {yr}    {w.outstanding_dv01_mn.mean():>10,.0f}   {w.soma_dv01_mn.mean():>10,.0f}"
              f"      {w.supply_3m_dv01_mn.mean():>+10,.0f}      {w.soma_net_3m_dv01_mn.mean():>+10,.0f}")
    # MSPD and SOMA publish on different lags -- SOMA's own latest month is often a
    # calendar month or more ahead of MSPD's, so "the latest row" and "the latest
    # outstanding figure" are not the same date. Report each off its own last valid
    # observation rather than a single (possibly NaN) last row.
    o = d["outstanding_dv01_mn"].dropna()
    h = d["soma_dv01_mn"].dropna()
    sup = d["supply_3m_dv01_mn"].dropna()
    buy = d["soma_net_3m_dv01_mn"].dropna()
    print(f"\nlatest outstanding ({o.index[-1].date()}): ${o.iloc[-1]:,.0f}mn/bp"
          + (f"  ({100*h.reindex([o.index[-1]]).iloc[0]/o.iloc[-1]:.1f}% held by SOMA)"
             if o.index[-1] in h.index else ""))
    print(f"latest SOMA held    ({h.index[-1].date()}): ${h.iloc[-1]:,.0f}mn/bp")
    print(f"latest 3m net supply ({sup.index[-1].date()}): ${sup.iloc[-1]:+,.0f}mn/bp")
    print(f"latest 3m net SOMA buy ({buy.index[-1].date()}): ${buy.iloc[-1]:+,.0f}mn/bp")


if __name__ == "__main__":
    _main()
