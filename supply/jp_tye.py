"""Japan ten-year-equivalent share held by the Bank of Japan, monthly.

Joins the BoJ's issue-by-issue holdings to the MoF auction reference (which supplies
the maturity and coupon the BoJ file omits), and prices both ladders off the MoF JGB
par curve for the same date.

Two approximations, both deliberate. The par yields are used as if they were zeros --
in a market that has spent this entire sample between -0.1% and 2% the coupon effect on
discounting is tiny, and it applies identically to the numerator and the denominator, so
it largely cancels in the ratio. And the outstanding amount per issue is total auction
allotments, which ignores buybacks; those are small in Japan and, again, common to both
sides for the issues the BoJ holds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply.fetch_jp import DATA, fetch_boj, fetch_mof_issues
from supply.tenyear import tye_share

logger = logging.getLogger(__name__)
TYE_PARQUET = DATA / "jp_tye.parquet"


def _curve_for(row: pd.Series) -> callable:
    """Log-linear interpolation across the MoF par curve, in DECIMAL."""
    tenors, vals = [], []
    for c in row.index:
        try:
            t = float(str(c).replace("Y", "").strip())
        except ValueError:
            continue
        v = row[c]
        if pd.notna(v):
            tenors.append(t)
            vals.append(float(v))
    if len(tenors) < 3:
        return None
    t = np.array(tenors); v = np.array(vals)
    order = np.argsort(t); t, v = t[order], v[order]

    def curve(x):
        x = np.asarray(x, dtype=float)
        return np.interp(np.clip(x, t.min(), t.max()), t, v)
    return curve


def build(refresh: bool = False) -> pd.DataFrame:
    if not refresh and TYE_PARQUET.exists():
        return pd.read_parquet(TYE_PARQUET)
    from japan.fetch_mof import load_yields

    ref = fetch_mof_issues()
    boj = fetch_boj()
    y = load_yields(refresh=False)
    y.index = pd.DatetimeIndex(y.index)

    key = ["bond_type", "issue"]
    held_all = boj.merge(ref[key + ["maturity_date", "coupon_pct"]], on=key, how="left")
    miss = held_all["maturity_date"].isna()
    if miss.any():
        logger.warning("%d of %d BoJ issue-months (%.1f%% of par) did not match the MoF "
                       "reference and are dropped", int(miss.sum()), len(held_all),
                       100 * held_all.loc[miss, "amount"].sum() / held_all["amount"].sum())
    held_all = held_all[~miss]

    rows = []
    for asof, held in held_all.groupby("asof"):
        prior = y.index[y.index <= asof]
        if len(prior) == 0:
            continue
        curve = _curve_for(y.loc[prior.max()])
        if curve is None:
            continue
        out = ref[ref["maturity_date"] > asof]        # still outstanding at this date
        if out.empty:
            continue
        r = tye_share(out, held, asof, curve)
        r["n_outstanding"], r["n_held"] = len(out), len(held)
        rows.append(r)
    df = pd.DataFrame(rows).set_index("asof").sort_index()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TYE_PARQUET)
    return df


def _main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = build(refresh="--refresh" in sys.argv)
    print(f"\n{len(d)} monthly obs {d.index.min().date()}..{d.index.max().date()}\n")
    print("           duration share   par share   ref duration   issues held")
    for yr in sorted(set(d.index.year)):
        w = d[d.index.year == yr]
        print(f"  {yr}      {w.tye_share.mean():5.1f}%         {w.par_share.mean():5.1f}%"
              f"        {w.ref_duration.mean():5.2f}y        {int(w.n_held.mean()):4}")


if __name__ == "__main__":
    _main()
