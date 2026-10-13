"""Ten-year breakeven inflation by scenario: US (CPI-U), UK (RPI) and the euro area
(HICPxT -- one market for OAT€i, BTP€i, Bund€i and the rest).

    BE(Y) = E(Y) + IRP

* E(Y), expected inflation: the scenario's own headline-inflation forecasts averaged over
  the ten years a bond bought at year-end Y covers (Y+1 .. Y+10; the last stated year
  carries on beyond the scenario). Today's E is the Baseline's, from today.
* IRP, the inflation risk premium (with the linker liquidity premium): today's breakeven
  less today's Baseline expected inflation, held constant. Out of sample, reverting the
  breakeven's level toward any fitted value forecast worse than holding it
  (validation/FINDINGS_breakeven.md), so the premium is not mean-reverted.

UK linkers pay RPI. E and IRP are on a CPI footing; the RPI breakeven adds the RPI-CPI
wedge for the part of the ten years still on RPI rules. RPI moves to CPIH methodology in
February 2030 for every RPI cashflow, gilts in issue included (announced 25 November
2020), so the wedge's share of a ten-year window falls to zero by 2030.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MARKETS = ("US", "UK", "EA")
# the scenario file and the headline-CPI column each market reads (the Bund market's
# inflation is already the euro area's)
SCENARIO_MARKET = {"US": "US", "UK": "UK", "EA": "DE"}
BASIS = {"US": "CPI-U", "UK": "RPI", "EA": "HICPxT"}

# Sensitivity of the ten-year breakeven to headline inflation (pp per pp), on monthly
# twelve-month changes with HAC errors (``fit_beta``):
#   US 0.111 (t 5.1), 2004-2026; 0.09-0.13 in every sub-period.
#   UK 0.050 (t 2.2), 2004-2026, without Aug 2022 - Mar 2023 (the LDI crisis, when
#      linker breakevens fell while CPI ran at 10-11%). 0.07-0.15 before 2020, ~0 after.
#   EA to be fitted when the breakeven history is in (``EA_FILE``).
BETA = {"US": 0.11, "UK": 0.05}
EXCLUDE = {"UK": [("2022-08-01", "2023-03-31")]}
START = {"US": "2004-01-01", "UK": "1997-06-01", "EA": "2004-01-01"}

UK_RPI_WEDGE = 0.9                       # pp of RPI over CPI, pre-reform
RPI_REFORM = 2030.083                    # February 2030
REFORM_ANNOUNCED = pd.Timestamp("2020-11-25")
TENOR = 10.0

# The euro-area history: a private file, like the Consensus data (columns date, value;
# percent). Not fetched: the AFT publishes it behind bot protection.
EA_FILE = ROOT / "macro" / "data" / "ea_breakeven_10y.csv"


def _year(t) -> float:
    t = pd.Timestamp(t)
    return t.year + (t.dayofyear - 1) / 365.25


def rpi_share(t, tenor: float = TENOR) -> float:
    """Share of the ``tenor`` years from ``t`` still accruing on RPI rules: all of it
    before the reform was announced, then the part before February 2030."""
    if pd.Timestamp(t) < REFORM_ANNOUNCED:
        return 1.0
    return float(np.clip((RPI_REFORM - _year(t)) / tenor, 0.0, 1.0))


def to_cpi(country: str, be: float, t) -> float:
    return be - UK_RPI_WEDGE * rpi_share(t) if country == "UK" else be


def to_market(country: str, be_cpi: float, t) -> float:
    return be_cpi + UK_RPI_WEDGE * rpi_share(t) if country == "UK" else be_cpi


# ------------------------------------------------------------------------ data
def breakeven(country: str, freq: str = "M") -> pd.Series:
    """The ten-year zero-coupon breakeven, percent, on the market's own index:
    nominal less real zero yield (GSW nominal/TIPS for the US, the BoE's implied
    inflation for the UK -- both the curves the nominal and real legs come from)."""
    if country == "US":
        from us.tips import build_breakeven_panel
        s = build_breakeven_panel(start="1999-01-01", freq=freq)[TENOR] * 100.0
    elif country == "UK":
        from uk.fetch_boe import load_curve
        s = load_curve("inflation", freq=freq)[TENOR] * 100.0
    elif country == "EA":
        if not EA_FILE.exists():
            raise FileNotFoundError(f"{EA_FILE.relative_to(ROOT)} not supplied")
        df = pd.read_csv(EA_FILE)
        s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                      index=pd.DatetimeIndex(pd.to_datetime(df["date"])))
        if freq in ("M", "monthly"):
            s = s.groupby([s.index.year, s.index.month]).tail(1)
    else:
        raise ValueError(country)
    return s.dropna().sort_index().rename("be")


def headline(country: str) -> pd.Series:
    from macro.fetch_headline_cpi import load
    return load()[SCENARIO_MARKET[country]].dropna().rename("head")


def panel(country: str) -> pd.DataFrame:
    """Monthly: the breakeven on its own index and on a CPI footing, and headline
    inflation (y/y, the latest print as of each month)."""
    be = breakeven(country)
    be.index = be.index.to_period("M").to_timestamp()
    d = pd.DataFrame({"be": be}).join(headline(country).rename("head"), how="left")
    d["head"] = headline(country).reindex(d.index.union(headline(country).index)) \
        .ffill().reindex(d.index)
    d["be_cpi"] = [to_cpi(country, v, t + pd.offsets.MonthEnd(0))
                   for t, v in d["be"].items()]
    return d.loc[START[country]:].dropna()


def fit_beta(country: str, lag: int = 12, exclude=None) -> dict:
    """``beta`` on ``lag``-month changes, HAC errors; months in ``exclude`` (default
    ``EXCLUDE``) dropped from either end of a change."""
    import statsmodels.api as sm
    d = panel(country)
    dd = d[["be_cpi", "head"]].diff(lag)
    for a, b in (EXCLUDE.get(country, []) if exclude is None else exclude):
        bad = (dd.index >= a) & (dd.index <= b)
        lagged = (dd.index - pd.DateOffset(months=lag) >= a) & \
                 (dd.index - pd.DateOffset(months=lag) <= b)
        dd = dd[~(bad | lagged)]
    dd = dd.dropna()
    m = sm.OLS(dd["be_cpi"], sm.add_constant(dd[["head"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag})
    return {"beta": float(m.params["head"]), "t": float(m.tvalues["head"]),
            "n": int(len(dd)), "start": str(dd.index.min().date()),
            "end": str(dd.index.max().date())}


def today(country: str) -> dict:
    """Latest daily breakeven (market index and CPI footing) and the latest headline
    print."""
    be = breakeven(country, freq="D")
    t = be.index[-1]
    h = headline(country)
    return {"date": t, "be": float(be.iloc[-1]), "be_cpi": to_cpi(country, float(be.iloc[-1]), t),
            "head": float(h.iloc[-1]), "head_month": h.index[-1]}


# ------------------------------------------------------------------ projection
def expected_inflation(annual: dict, start: float, tenor: float = TENOR) -> float:
    """Average of calendar-year inflation ``annual`` ({year: %}) over [start, start +
    tenor), ``start`` in fractional years; years past the last stated carry it on."""
    last = annual[max(annual)]
    first = annual[min(annual)]
    t, end, acc = float(start), float(start) + tenor, 0.0
    while t < end - 1e-9:
        y = int(np.floor(t))
        nxt = min(y + 1.0, end)
        acc += (nxt - t) * annual.get(y, last if y > max(annual) else first)
        t = nxt
    return acc / tenor


def irp_today(country: str, base: dict, now: dict | None = None) -> float:
    """Today's inflation risk premium on a CPI footing: breakeven less the Baseline's
    expected inflation over the next ten years."""
    now = today(country) if now is None else now
    return now["be_cpi"] - expected_inflation(base, _year(now["date"]))


def project(country: str, annual: dict, years, irp: float) -> dict:
    """Year-end ten-year breakeven on the market's own index: E(Y) + IRP (+ the RPI
    wedge still to run, for the UK)."""
    out = {}
    for y in years:
        t = pd.Timestamp(int(y), 12, 31)
        out[int(y)] = to_market(country, expected_inflation(annual, int(y) + 1.0) + irp, t)
    return out


def scenario_inflation(country: str) -> dict:
    """{scenario: {year: headline inflation}} from the scenario files."""
    m = SCENARIO_MARKET[country]
    df = pd.read_csv(ROOT / "scenarios" / "data" / f"{m.lower()}_scenarios.csv")
    return {s: {int(y): float(v) for y, v in g.set_index("year")["inflation"].dropna().items()}
            for s, g in df.groupby("scenario", sort=False)}


def year_end_table(markets=MARKETS, years=None, decimals: int = 2) -> pd.DataFrame:
    """Rows market x scenario: today's breakeven, its split, then each year-end."""
    years = list(years or range(pd.Timestamp.today().year, pd.Timestamp.today().year + 10))
    rows = []
    for c in markets:
        try:
            now = today(c)
        except FileNotFoundError as exc:
            logger.warning("%s breakeven skipped: %s", c, exc)
            continue
        sc = scenario_inflation(c)
        irp = irp_today(c, sc["Baseline"], now)
        for s, annual in sc.items():
            p = project(c, annual, years, irp)
            rows.append({"market": c, "index": BASIS[c], "scenario": s,
                         "IRP": round(irp, decimals), "Today": round(now["be"], decimals),
                         **{str(y): round(v, decimals) for y, v in p.items()}})
    return pd.DataFrame(rows)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t = year_end_table()
    out = ROOT / "scenarios" / "breakeven_10y_year_end.csv"
    t.to_csv(out, index=False)
    print(t.to_string(index=False))
    print(f"-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    _main()
