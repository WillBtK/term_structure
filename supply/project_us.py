"""Project the Fed's duration share forward under a NEUTRAL balance-sheet policy.

The policy modelled here: SOMA's par amount grows in line with nominal GDP, and its
purchases are distributed to match the MARKET's maturity profile, so the portfolio's
maturity composition converges on the market's. That is a "neutral" balance sheet -- one
that neither adds nor withdraws duration risk relative to the stock outstanding.

It is not where the portfolio starts. SOMA today runs LONG of the market: it holds about
14% of the marketable Treasury market by par but about 21% of its duration risk, the
legacy of Operation Twist and of QE purchases concentrated at the long end. Convergence
therefore means shedding duration relative to the market for as long as the legacy long
holdings take to roll off -- a persistent negative flow, which is the variable the
premium responds to.

Two modelling choices, both stated rather than buried:

* The MARKET's maturity profile is treated as stationary -- the Treasury is assumed to
  keep issuing in roughly its current mix -- so market ten-year-equivalents grow at the
  same rate as market par. Only SOMA's ladder is simulated. Relaxing this needs a view
  on Treasury issuance duration, which is a separate forecast.
* Duration is recomputed each year off a flat curve at the current long yield. The share
  is a ratio of duration-weighted sums using the same curve on both sides, so it is far
  less sensitive to the curve than either leg alone.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply.fetch_us import fetch_mspd, fetch_soma
from supply.tenyear import ladder_durations, reference_duration, ten_year_equivalents

logger = logging.getLogger(__name__)

# Maturity buckets used to match SOMA's purchases to the market's profile. Coarse on
# purpose: the point is the broad long/short composition, not a per-CUSIP replication.
BUCKETS = [(0, 1), (1, 3), (3, 5), (5, 7), (7, 10), (10, 20), (20, 31)]


def _bucket_weights(ladder: pd.DataFrame, asof) -> np.ndarray:
    """Share of par in each maturity bucket."""
    yrs = (pd.to_datetime(ladder["maturity_date"]) - pd.Timestamp(asof)).dt.days / 365.25
    amt = pd.to_numeric(ladder["amount"], errors="coerce")
    w = np.array([amt[(yrs >= lo) & (yrs < hi)].sum() for lo, hi in BUCKETS], dtype=float)
    return w / w.sum() if w.sum() else w


def _flat_curve(y: float):
    return lambda t: np.full_like(np.asarray(t, dtype=float), y)


def project(years: int = 12, soma_growth: float = 0.04, market_growth: float = 0.05,
            yield_pct: float = 4.3) -> pd.DataFrame:
    """Duration share path under convergence to the market's maturity profile."""
    mspd, soma = fetch_mspd(), fetch_soma()
    asof = soma["asof"].max()
    market = mspd[mspd["record_date"] == mspd["record_date"].max()]
    held = soma[soma["asof"] == asof]
    curve = _flat_curve(yield_pct / 100.0)

    mkt_w = _bucket_weights(market, asof)
    ref0 = reference_duration(asof, curve)
    mkt_tye0 = ten_year_equivalents(market["amount"],
                                    ladder_durations(market, asof, curve), ref0)
    mkt_par0 = float(market["amount"].sum())

    # SOMA is carried as a ladder we age year by year; purchases are added as synthetic
    # holdings at each bucket's midpoint, weighted to the market's profile.
    lad = held[["maturity_date", "coupon_pct", "amount"]].copy()
    soma_par0 = float(lad["amount"].sum())
    rows = []
    for k in range(years + 1):
        t = asof + pd.DateOffset(years=k)
        if k:
            # Age the ladder, THEN reinvest maturing par plus growth, and only then
            # measure. Measuring in between would report the portfolio at the one
            # instant each year when the maturities have gone and the reinvestment
            # has not yet happened -- a dip that is an artefact of the loop, not policy.
            lad = lad[pd.to_datetime(lad["maturity_date"]) > t]
            target = soma_par0 * (1 + soma_growth) ** k
            buy = max(target - float(lad["amount"].sum()), 0.0)
            new_buys = pd.DataFrame({
                "maturity_date": [t + pd.DateOffset(days=int(365.25 * (lo + hi) / 2))
                                  for lo, hi in BUCKETS],
                "coupon_pct": [yield_pct] * len(BUCKETS),
                "amount": buy * mkt_w})
            lad = pd.concat([lad, new_buys], ignore_index=True)
        ref = reference_duration(t, curve)
        s_tye = ten_year_equivalents(lad["amount"],
                                     ladder_durations(lad, t, curve), ref)
        s_par = float(lad["amount"].sum())
        m_tye = mkt_tye0 * (1 + market_growth) ** k
        m_par = mkt_par0 * (1 + market_growth) ** k
        rows.append({"year": t.year, "soma_par": s_par,
                     "duration_share": 100 * s_tye / m_tye,
                     "par_share": 100 * s_par / m_par,
                     "soma_duration": s_tye * ref / s_par})
    return pd.DataFrame(rows)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mspd, soma = fetch_mspd(), fetch_soma()
    asof = soma["asof"].max()
    market = mspd[mspd["record_date"] == mspd["record_date"].max()]
    held = soma[soma["asof"] == asof]
    curve = _flat_curve(0.043)
    ref = reference_duration(asof, curve)
    for nm, lad in (("SOMA  ", held), ("market", market)):
        tye = ten_year_equivalents(lad["amount"], ladder_durations(lad, asof, curve), ref)
        par = float(lad["amount"].sum())
        print(f"  {nm}: par ${par/1e6:6.2f}tn   ten-year-equivalents ${tye/1e6:6.2f}tn   "
              f"avg duration {tye * ref / par:5.2f}y")
    print(f"\n  (reference 10y note duration {ref:.2f}y, as of {asof.date()})\n")

    p = project()
    print("  neutral convergence: SOMA par +4%/yr, purchases matched to the market mix,"
          "\n  marketable debt +5%/yr\n")
    print("   year   duration share   par share   gap    12m flow   SOMA dur")
    prev = None
    for _, r in p.iterrows():
        flow = "" if prev is None else f"{r.duration_share - prev:+6.2f}pp"
        print(f"   {int(r.year)}       {r.duration_share:6.2f}%       {r.par_share:6.2f}%"
              f"  {r.duration_share - r.par_share:+6.2f}pp   {flow:>8}  "
              f"{r.soma_duration:5.2f}y")
        prev = r.duration_share
    avg = (p.duration_share.iloc[-1] - p.duration_share.iloc[0]) / (len(p) - 1)
    print(f"\n  average flow over the path: {avg:+.2f}pp/yr")
    for tenor, beta, lam in ((10, -0.0220, 0.269), (3, -0.0358, 0.424)):
        print(f"   {tenor:2}y: {beta * avg * 100:+.1f}bp/yr contribution;  "
              f"steady-state level effect {beta * avg / lam * 100:+.1f}bp")


if __name__ == "__main__":
    _main()
