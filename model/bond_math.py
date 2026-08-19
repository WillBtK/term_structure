"""Coupon-bond math: cashflow schedule, accrued interest, price from a discount curve,
and price -> yield inversion.

The other liquidity gauges consume yields their source already publishes. The US
nominal Treasury cross-section (TreasuryDirect FedInvest) publishes **prices**, so
putting each bond on a comparable yield basis is our job — that is what this module is
for. It is pure numpy/scipy with no I/O so it can be unit-tested against textbook
cases offline.

Conventions follow US Treasury notes and bonds:

* semiannual coupons, dated back from the maturity date (so a bond maturing 15 Aug
  pays every 15 Feb / 15 Aug);
* **actual/actual** accrual within the current coupon period (ICMA), which is the
  Treasury convention for notes and bonds — not 30/360;
* quoted (clean) prices per 100 face; the invoice (dirty) price adds accrued interest;
* the yield to maturity is the semiannually-compounded bond-equivalent yield, the
  basis every published Treasury yield uses.

Discount curves enter as a callable ``disc(t)`` giving the price today of 1 unit paid
``t`` years hence. For GSW/Bundesbank/BoE Svensson parameters the zero yields are
CONTINUOUSLY compounded, so ``disc(t) = exp(-y(t) * t)`` — see ``disc_from_zero``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

FREQ = 2                       # semiannual — US Treasury notes/bonds
_DAY = np.timedelta64(1, "D")


def coupon_dates(settle, maturity, freq: int = FREQ) -> pd.DatetimeIndex:
    """Remaining coupon dates strictly after ``settle``, up to and including maturity.

    Generated BACKWARDS from maturity so the payment day-of-month is the maturity's,
    which is how Treasury schedules actually work (a short first period at issue does
    not affect a seasoned bond's remaining schedule).
    """
    settle = pd.Timestamp(settle)
    maturity = pd.Timestamp(maturity)
    if maturity <= settle:
        return pd.DatetimeIndex([])
    step = pd.DateOffset(months=12 // freq)
    dates, d = [], maturity
    while d > settle:
        dates.append(d)
        d = d - step
    return pd.DatetimeIndex(sorted(dates))


def _period_bounds(settle, maturity, freq: int = FREQ):
    """(previous coupon date, next coupon date) bracketing ``settle``."""
    nxt = coupon_dates(settle, maturity, freq)
    if len(nxt) == 0:
        return None, None
    return nxt[0] - pd.DateOffset(months=12 // freq), nxt[0]


def accrued_interest(settle, maturity, coupon_pct: float, freq: int = FREQ) -> float:
    """Accrued interest per 100 face, actual/actual within the coupon period."""
    prev, nxt = _period_bounds(settle, maturity, freq)
    if prev is None:
        return 0.0
    settle = pd.Timestamp(settle)
    days_in_period = (nxt - prev).days
    if days_in_period <= 0:
        return 0.0
    frac = (settle - prev).days / days_in_period
    return coupon_pct / freq * frac


def year_fractions(settle, maturity, freq: int = FREQ) -> np.ndarray:
    """Time in years from ``settle`` to each remaining coupon date (actual/365.25).

    A calendar-day measure is used rather than a coupon-count measure so the times line
    up with the continuously-compounded zero curve the discount factors come from.
    """
    dates = coupon_dates(settle, maturity, freq)
    if len(dates) == 0:
        return np.empty(0)
    return (dates - pd.Timestamp(settle)).days.to_numpy() / 365.25


def cashflows(settle, maturity, coupon_pct: float, freq: int = FREQ):
    """``(times_in_years, amounts_per_100_face)`` for all remaining payments."""
    t = year_fractions(settle, maturity, freq)
    if t.size == 0:
        return t, t
    cf = np.full(t.size, coupon_pct / freq, dtype=float)
    cf[-1] += 100.0
    return t, cf


def disc_from_zero(zero_fn):
    """Discount function from a CONTINUOUSLY-compounded zero-yield function.

    ``zero_fn(t)`` returns the zero yield in DECIMAL at horizon ``t`` years; the result
    is ``exp(-y(t) t)``. GSW (US), Bundesbank and BoE all publish continuously
    compounded Svensson zeros, so their parameters drop straight in.
    """
    def disc(t):
        t = np.asarray(t, dtype=float)
        return np.exp(-np.asarray(zero_fn(t), dtype=float) * t)
    return disc


def price_from_curve(settle, maturity, coupon_pct: float, disc,
                     freq: int = FREQ) -> float:
    """CLEAN price per 100 face implied by a discount curve (dirty minus accrued)."""
    t, cf = cashflows(settle, maturity, coupon_pct, freq)
    if t.size == 0:
        return np.nan
    dirty = float(np.sum(cf * np.asarray(disc(t), dtype=float)))
    return dirty - accrued_interest(settle, maturity, coupon_pct, freq)


def price_from_ytm(settle, maturity, coupon_pct: float, ytm: float,
                   freq: int = FREQ) -> float:
    """CLEAN price per 100 face at a semiannually-compounded yield ``ytm`` (decimal)."""
    t, cf = cashflows(settle, maturity, coupon_pct, freq)
    if t.size == 0:
        return np.nan
    dirty = float(np.sum(cf * (1.0 + ytm / freq) ** (-freq * t)))
    return dirty - accrued_interest(settle, maturity, coupon_pct, freq)


def ytm_from_price(clean_price: float, settle, maturity, coupon_pct: float,
                   freq: int = FREQ, lo: float = -0.20, hi: float = 1.0) -> float:
    """Semiannually-compounded yield to maturity (decimal) from a CLEAN price.

    Price is strictly decreasing in yield, so a bracketed root-find is both safe and
    fast. Returns NaN when the price is outside the bracket (a data error, or a
    stripped/defaulted quote) rather than raising, so one bad row cannot take a whole
    day's cross-section down. The lower bound admits the negative yields that Bunds
    and short Treasuries have printed.
    """
    if not np.isfinite(clean_price) or clean_price <= 0:
        return np.nan
    t, _ = cashflows(settle, maturity, coupon_pct, freq)
    if t.size == 0:
        return np.nan

    def f(y):
        return price_from_ytm(settle, maturity, coupon_pct, y, freq) - clean_price

    try:
        flo, fhi = f(lo), f(hi)
    except (OverflowError, FloatingPointError):
        return np.nan
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


def yields_from_prices(panel: pd.DataFrame, freq: int = FREQ) -> pd.DataFrame:
    """Add a ``yield`` column (DECIMAL YTM) to a ``(date, maturity, coupon, price)``
    per-security panel. ``coupon`` is the annual rate in PERCENT, ``price`` the clean
    price per 100 face — the units TreasuryDirect publishes."""
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["maturity"] = pd.to_datetime(df["maturity"])
    df["yield"] = [
        ytm_from_price(p, d, m, c, freq)
        for p, d, m, c in zip(df["price"], df["date"], df["maturity"], df["coupon"])
    ]
    return df
