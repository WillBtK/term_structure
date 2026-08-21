"""Ten-year equivalents: how much duration risk a bond portfolio actually carries.

Par value is the wrong unit for the supply channel. A bill and a thirty-year bond are
the same number of dollars and completely different quantities of interest-rate risk,
and the whole point of quantitative easing was to buy the second kind. Measured by par
share, the Federal Reserve held MORE of the public debt through the 1970s -- when its
portfolio was mostly bills held for reserve management -- than at the peak of QE. So a
par-share measure cannot tell 1975 from 2014, and empirically it does not.

The fix, following Li and Wei (2013) and Ihrig et al. (2018), is to express a portfolio
as the amount of a benchmark ten-year note carrying the same duration risk:

    TYE = sum_i (D_i * A_i) / D_ref

with ``D_i`` the duration of security i, ``A_i`` its outstanding par amount, and
``D_ref`` the duration of a current ten-year note. The ratio of the central bank's TYE
to the market's TYE is then the share of duration risk removed from private hands --
which is the quantity preferred-habitat theory (Vayanos and Vila) says should matter.

This module is pure arithmetic over a maturity ladder; fetching the ladders is each
market's own problem and lives in the per-market fetchers.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.bond_math import macaulay_duration

logger = logging.getLogger(__name__)

# Duration of the benchmark the ladder is expressed in. Recomputed per date from the
# prevailing yield rather than fixed, because a 10y note's duration is ~8.9y at a 2%
# yield and ~7.8y at 6% -- holding it constant would put a spurious yield-driven trend
# into a series meant to measure quantities.
REF_TENOR_YEARS = 10.0


def ladder_durations(ladder: pd.DataFrame, asof, zero_curve) -> pd.Series:
    """Macaulay duration per security, priced off the market's own zero curve.

    ``ladder`` needs ``maturity_date`` and ``coupon_pct``. ``zero_curve(t)`` returns the
    continuously compounded zero yield in DECIMAL at ``t`` years, which is what every
    Svensson-parameterised curve in this repo already provides.
    """
    asof = pd.Timestamp(asof)
    yrs = (pd.to_datetime(ladder["maturity_date"]) - asof).dt.days / 365.25
    out = np.full(len(ladder), np.nan)
    coupons = ladder["coupon_pct"].to_numpy(dtype=float)
    for i, (t, c, mat) in enumerate(zip(yrs.to_numpy(), coupons,
                                        pd.to_datetime(ladder["maturity_date"]))):
        if not np.isfinite(t) or t <= 0:
            continue
        # Continuously compounded zero -> the semiannual bond-equivalent yield the
        # duration formula is written in.
        z = float(zero_curve(t))
        ytm = 2.0 * (np.exp(z / 2.0) - 1.0)
        out[i] = macaulay_duration(asof, mat, float(c), ytm)
    return pd.Series(out, index=ladder.index, name="duration")


def reference_duration(asof, zero_curve, tenor: float = REF_TENOR_YEARS) -> float:
    """Duration of a par benchmark note at ``tenor``, priced off the same curve.

    Par means its coupon equals its yield, so the coupon is read off the curve too.
    """
    asof = pd.Timestamp(asof)
    z = float(zero_curve(tenor))
    ytm = 2.0 * (np.exp(z / 2.0) - 1.0)
    mat = asof + pd.DateOffset(days=int(round(tenor * 365.25)))
    return macaulay_duration(asof, mat, ytm * 100.0, ytm)


def ten_year_equivalents(amounts, durations, ref_duration: float) -> float:
    """Par amount of the benchmark note carrying the same total duration risk."""
    a = np.asarray(amounts, dtype=float)
    d = np.asarray(durations, dtype=float)
    ok = np.isfinite(a) & np.isfinite(d)
    if not ok.any() or not np.isfinite(ref_duration) or ref_duration <= 0:
        return np.nan
    return float((a[ok] * d[ok]).sum() / ref_duration)


def tye_share(outstanding: pd.DataFrame, held: pd.DataFrame, asof,
              zero_curve) -> dict:
    """Share of the market's DURATION RISK held by the central bank, at one date.

    Both frames carry ``maturity_date``, ``coupon_pct`` and ``amount`` (par, any
    consistent unit). Returns the two ten-year-equivalent totals, the share, and the
    par share alongside it -- the gap between the two is the whole point of the measure,
    so it is worth carrying rather than recomputing later.
    """
    ref = reference_duration(asof, zero_curve)
    o_tye = ten_year_equivalents(outstanding["amount"],
                                 ladder_durations(outstanding, asof, zero_curve), ref)
    h_tye = ten_year_equivalents(held["amount"],
                                 ladder_durations(held, asof, zero_curve), ref)
    o_par = float(pd.to_numeric(outstanding["amount"], errors="coerce").sum())
    h_par = float(pd.to_numeric(held["amount"], errors="coerce").sum())
    return {
        "asof": pd.Timestamp(asof),
        "ref_duration": ref,
        "outstanding_tye": o_tye,
        "held_tye": h_tye,
        "tye_share": np.nan if not o_tye else 100.0 * h_tye / o_tye,
        "par_share": np.nan if not o_par else 100.0 * h_par / o_par,
    }
