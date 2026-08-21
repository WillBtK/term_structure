"""US ten-year-equivalent share held by the Federal Reserve, monthly.

Combines the marketable Treasury ladder (MSPD) with the Fed's own holdings (SOMA),
duration-weighting both off the GSW zero curve for the same date. The output carries
BOTH the duration share and the par share, because the divergence between them is the
measurement point: the par share cannot distinguish a 1970s bills portfolio from a
QE-era long-bond portfolio, and the duration share can.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply.fetch_us import DATA, fetch_mspd, fetch_soma
from supply.tenyear import tye_share

logger = logging.getLogger(__name__)
TYE_PARQUET = DATA / "us_tye.parquet"


def _curve_for(params_row) -> callable:
    """Zero curve in DECIMAL from a row of GSW Svensson parameters.

    ``svensson_yield`` returns PERCENT, so divide -- the duration math works in decimal.
    """
    from us.reconstruct import svensson_yield
    b0, b1, b2, b3 = (float(params_row[c]) for c in ("BETA0", "BETA1", "BETA2", "BETA3"))
    t1, t2 = float(params_row["TAU1"]), float(params_row["TAU2"])

    def curve(t):
        t = np.maximum(np.asarray(t, dtype=float), 1.0 / 365.25)
        return np.asarray(svensson_yield(t, b0, b1, b2, b3, t1, t2), dtype=float) / 100.0
    return curve


def build(refresh: bool = False) -> pd.DataFrame:
    if not refresh and TYE_PARQUET.exists():
        return pd.read_parquet(TYE_PARQUET)
    from us.fetch_gsw import load_gsw_params

    out_all, held_all = fetch_mspd(), fetch_soma()
    gsw = load_gsw_params().dropna(subset=["BETA0", "BETA1", "BETA2", "BETA3",
                                           "TAU1", "TAU2"])
    gsw = gsw[gsw["TAU2"] > 0]
    gsw.index = pd.DatetimeIndex(gsw.index)

    out_all["ym"] = out_all["record_date"].dt.to_period("M")
    held_all["ym"] = held_all["asof"].dt.to_period("M")
    rows = []
    for ym, held in held_all.groupby("ym"):
        out = out_all[out_all["ym"] == ym]
        if out.empty:
            continue
        asof = out["record_date"].max()
        prior = gsw.index[gsw.index <= asof]
        if len(prior) == 0:
            continue
        r = tye_share(out, held, asof, _curve_for(gsw.loc[prior.max()]))
        r["n_outstanding"], r["n_held"] = len(out), len(held)
        rows.append(r)
        if len(rows) % 36 == 0:
            logger.info("  %s  tye %.1f%%  par %.1f%%", asof.date(),
                        r["tye_share"], r["par_share"])
    df = pd.DataFrame(rows).set_index("asof").sort_index()
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TYE_PARQUET)
    return df


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = build(refresh=True)
    print(f"\n{len(d)} monthly obs {d.index.min().date()}..{d.index.max().date()}\n")
    print("           duration share   par share   ref duration")
    for yr in range(2003, 2027):
        w = d[d.index.year == yr]
        if w.empty:
            continue
        print(f"  {yr}      {w.tye_share.mean():5.1f}%         {w.par_share.mean():5.1f}%"
              f"        {w.ref_duration.mean():.2f}y")


if __name__ == "__main__":
    _main()
