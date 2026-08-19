"""Timely nowcast extension for the Canadian curve, from BoC benchmark bond yields.

The Bank of Canada's fitted **zero-coupon** curve (``canada.fetch_boc``) is the model's
native input, but it publishes on a ~3-week lag. BoC's *benchmark bond yields* (Valet
group ``bond_yields_benchmark``: 2/3/5/7/10-year and Long-term) post next-business-day,
so they can carry the curve forward over the gap.

Levels are NOT interchangeable — a benchmark yield is a single on-the-run coupon bond's
YTM, while the model consumes zero-coupon spot rates — so, exactly as for the US H.15
splice, we transfer only the day-to-day CHANGE, anchored to the zero curve's last level:

    nowcast_zero(tau, t) = ZC(tau, t0) + sum of daily benchmark changes over (t0, t]

**Benchmark rolls.** Because these are on-the-run bonds, the series steps when the
benchmark bond changes: the quoted yield jumps to a different bond's yield without the
market having moved. Measured over 2015+ (2,866 days) this happens on ~0.30% of
tenor-days — about 0.76 roll events per tenor per year — and it is idiosyncratic: one
tenor breaks away while the rest of the curve moves normally (e.g. 2023-05-18, the 3y
moved -13bp while 2/5/7/10y moved +4/+7/+8/+9). A naive cumulative splice would bake
that spurious step into every later nowcast day, so :func:`_deroll` neutralises it by
replacing a robust cross-sectional outlier with the curve-median change for that day.

Away from rolls the two sources agree closely: the std of (benchmark change - zero-curve
change) is 0.62-0.97bp across tenors, comparable to the US H.15 splice. Notably the
*largest* discrepancies come from the BoC zero curve's own short-end fitting instability
(e.g. -24bp then +27bp at 2y on 2025-02-20/21 while the benchmark moved +2/+3bp), not
from the benchmark series.

Coverage: 2,3,5,7,10y and Long-term (mapped to 30y). The 0.25-1y and 20y grid points have
no benchmark counterpart and stay NaN on nowcast rows (ragged — the EKF masks per date;
with six well-spread tenors the three AFNS factors remain identified). The nowcast is
provisional, flagged in the viz, and self-heals when the real zero-curve batch lands.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parent / "data"
CACHE = DATA / "boc_benchmark.parquet"

VALET_URL = os.environ.get(
    "BOC_VALET_URL",
    "https://www.bankofcanada.ca/valet/observations/group/bond_yields_benchmark/json")
VALET_START = os.environ.get("BOC_VALET_START", "2000-01-01")
# Valet series suffix -> grid tenor (years). "LONG" is the long-term benchmark (~30y).
BENCH_TENORS = {"2YR": 2.0, "3YR": 3.0, "5YR": 5.0, "7YR": 7.0, "10YR": 10.0, "LONG": 30.0}
# Roll detection: a tenor's daily change is treated as a roll when it is both a robust
# cross-sectional outlier (modified z on the curve's own changes) and materially large.
ROLL_Z = float(os.environ.get("BOC_ROLL_Z", "4.0"))
ROLL_MIN_BP = float(os.environ.get("BOC_ROLL_MIN_BP", "4.0"))


def load_benchmark_panel(refresh: bool = False) -> pd.DataFrame:
    """BoC benchmark bond yields as a DECIMAL panel (columns = grid tenors, float),
    business-daily. Cached; falls back to the cache when the fetch fails."""
    if not refresh and CACHE.exists():
        return pd.read_parquet(CACHE)
    try:
        import requests
        r = requests.get(f"{VALET_URL}?start_date={VALET_START}",
                         headers={"User-Agent": "term-premium-atlas/1.0"}, timeout=60)
        r.raise_for_status()
        obs = r.json()["observations"]
    except Exception as exc:                                        # noqa: BLE001
        if CACHE.exists():
            logger.warning("BoC Valet fetch failed (%s) -> using cache", exc)
            return pd.read_parquet(CACHE)
        raise
    rows = {}
    for o in obs:
        d = pd.Timestamp(o["d"])
        rec = {}
        for k, v in o.items():
            if k == "d":
                continue
            key = k.split(".")[2] if k.count(".") >= 2 else k       # BD.CDN.10YR.DQ.YLD
            ten = BENCH_TENORS.get(key)
            if ten is None:
                continue                                            # e.g. RRB (real) — not ours
            val = v.get("v") if isinstance(v, dict) else v
            rec[ten] = float(val) / 100.0 if val not in (None, "") else np.nan
        if rec:
            rows[d] = rec
    panel = pd.DataFrame(rows).T.sort_index()
    panel = panel[sorted(panel.columns)]
    DATA.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE)
    logger.info("BoC benchmark yields: %d tenors, %s..%s (%d days)", panel.shape[1],
                panel.index.min().date(), panel.index.max().date(), len(panel))
    return panel


def _deroll(chg: pd.DataFrame, z: float = ROLL_Z, min_bp: float = ROLL_MIN_BP):
    """Neutralise benchmark-roll steps in a frame of daily CHANGES (decimal).

    A roll moves one tenor without the curve moving, so we flag a change that is both a
    modified-z outlier against the same day's other tenors and larger than ``min_bp``
    away from the curve median, and replace it with that median. Returns
    ``(cleaned, n_flagged)``."""
    if chg.shape[1] < 3:
        return chg, 0                                               # too few tenors to judge
    med = chg.median(axis=1)
    dev = chg.sub(med, axis=0)
    mad = dev.abs().median(axis=1).replace(0.0, np.nan)
    zs = 0.6745 * dev.div(mad, axis=0)
    bad = (zs.abs() > z) & (dev.abs() > min_bp / 10000.0)
    bad = bad.fillna(False)
    out = chg.mask(bad, med, axis=0)
    return out, int(bad.to_numpy().sum())


def nowcast_extend(panel: pd.DataFrame, refresh: bool = False, max_gap_days: int = 40):
    """Append benchmark-increment nowcast rows to a BoC zero-curve ``panel`` for business
    days after its last date. Returns ``(extended_panel, dates)``; ``dates`` is empty when
    there is nothing newer, the gap is implausibly long, or anything fails (inert)."""
    try:
        base = panel.dropna(how="all")
        if base.empty:
            return panel, []
        t0 = base.index.max()
        bm = load_benchmark_panel(refresh=refresh)
        future = bm.index[bm.index > t0]
        if len(future) == 0:
            return panel, []
        prior = bm.index[bm.index <= t0]
        if len(prior) == 0:
            logger.warning("BoC nowcast: no benchmark obs on/before the zero curve's last date")
            return panel, []
        a0 = prior.max()
        if (future.max() - t0).days > max_gap_days:
            logger.warning("BoC nowcast: zero curve (%s) lags benchmarks (%s) by >%dd -> skip",
                           t0.date(), future.max().date(), max_gap_days)
            return panel, []
        tens = [t for t in base.columns if t in bm.columns]
        if not tens:
            return panel, []
        span = bm.loc[(bm.index >= a0) & (bm.index <= future.max()), tens]
        chg = span.diff().iloc[1:]                                  # changes after the anchor
        chg, n_roll = _deroll(chg)
        cum = chg.cumsum()                                          # cumulative move since a0
        rows = cum.add(base.loc[t0, tens].astype(float), axis=1)
        rows = rows.reindex(columns=base.columns).loc[rows.index > t0].dropna(how="all")
        if rows.empty:
            return panel, []
        out = pd.concat([panel, rows]).sort_index()
        out = out[~out.index.duplicated(keep="first")]              # keep the real curve on overlap
        dates = [str(d.date()) for d in rows.index]
        logger.info("BoC nowcast: +%d day(s) %s..%s over %d tenors (anchor %s, %d roll step(s) "
                    "neutralised)", len(dates), dates[0], dates[-1], len(tens), t0.date(), n_roll)
        return out, dates
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("BoC nowcast failed (%s) -> zero curve only", exc)
        return panel, []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = load_benchmark_panel(refresh=True)
    print(f"BoC benchmarks {p.index.min().date()}..{p.index.max().date()} "
          f"tenors={list(p.columns)}\nlatest={(p.iloc[-1] * 100).round(3).to_dict()}")


if __name__ == "__main__":
    main()
