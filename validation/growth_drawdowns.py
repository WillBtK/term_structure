"""What actually happens to the curve during a growth drawdown.

The scenario engine converges monotonically toward a terminal fair value. Real
downturns do not: yields fall hard, trough, and recover. Before deciding whether that
is a defect in the engine or a defect in the intuition, it is worth asking the data
what a drawdown actually does -- how far the ten-year falls, when it troughs, how much
has come back three and five years later, and **which leg moves**: the expected
average short rate, or the term premium.

That last split is the whole point. If the fall is expectations, the scenario tool
already has the mechanism and only needs input paths that say "cut, then normalise".
If it is the premium, the tool cannot produce it at all, because the premium model has
no cyclical channel -- and ``FINDINGS_growth_drawdowns.md`` records that a cyclical
term was tested inside the regression and rejected.

**Dating the episodes.** No recession committee, and nothing country-specific: an
episode starts at the local MINIMUM of the unemployment rate that precedes a rise of
at least ``MIN_RISE`` within ``RISE_WINDOW`` months. That is a Sahm-style rule, it
applies unchanged to every market, and it dates the last good month rather than the
month a committee later decided things had turned -- which is the timing an investor
would actually have traded around. Episodes inside ``MIN_GAP`` months of one another
are collapsed, so a double-dip counts once.

**Read the ten-year row first, and read it against the two-year.** The dilution
argument -- that a dated two-year event averaged into a ten-year expected short rate
is necessarily small -- predicts the ten-year should fall much less than the two.
Whether it does is the test.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PANEL = HERE.parent / "viz" / "data_daily.parquet"

# OECD harmonised unemployment, monthly, per market; the US uses its own headline rate
UNEMP = {"US": "UNRATE", "DE": "LRHUTTTTDEM156S", "UK": "LRHUTTTTGBM156S",
         "JP": "LRHUTTTTJPM156S", "CA": "LRHUTTTTCAM156S"}

MIN_RISE = 1.0        # percentage points the unemployment rate must add...
RISE_WINDOW = 24      # ...within this many months, for the trough to count
MIN_GAP = 48          # months; nearer episodes are one episode, not two
HORIZONS = (0, 6, 12, 18, 24, 36, 60)
TENORS = (2.0, 5.0, 10.0, 30.0)


def unemployment(country: str) -> pd.Series:
    from validation.fred_api import fred_series
    return fred_series(UNEMP[country]).dropna().resample("MS").mean().dropna()


def episodes(country: str, min_rise: float = MIN_RISE,
             window: int = RISE_WINDOW, min_gap: int = MIN_GAP) -> list:
    """Start dates of growth drawdowns: the unemployment trough before a real rise."""
    u = unemployment(country)
    # the largest rise over the NEXT `window` months, from each month
    fwd_max = u[::-1].rolling(window, min_periods=1).max()[::-1]
    qualifies = ((fwd_max - u) >= min_rise).to_numpy()
    # Within each contiguous run of qualifying months, take the LOWEST unemployment
    # reading -- the cycle trough. Taking the first qualifying month instead would date
    # the episode at whatever point the coming rise first became large enough to see,
    # which on a slow expansion can be years before the actual low.
    out, i, n = [], 0, len(u)
    while i < n:
        if not qualifies[i]:
            i += 1
            continue
        j = i
        while j < n and qualifies[j]:
            j += 1
        d = u.iloc[i:j].idxmin()
        if not out or (d - out[-1]).days >= min_gap * 30:
            out.append(d)
        i = j
    return out


def _curve(country: str) -> pd.DataFrame:
    """Month-start observed / expected / premium by tenor, in percent."""
    d = pd.read_parquet(PANEL)
    d = d[(d["country"] == country) & (d["instrument"] == "nominal")]
    frames = {}
    for t in TENORS:
        s = d[d["tenor"] == t].set_index("date").sort_index()
        if s.empty:
            continue
        m = s[["observed", "expected", "term_premium"]].resample("MS").mean() * 100.0
        for col, short in (("observed", "y"), ("expected", "e"), ("term_premium", "tp")):
            frames[f"{short}{t:g}"] = m[col]
    return pd.DataFrame(frames)


def event_study(country: str = "US", horizons: tuple = HORIZONS) -> pd.DataFrame:
    """Change in each series from the episode start, in basis points, per horizon."""
    c, rows = _curve(country), []
    for t0 in episodes(country):
        if t0 not in c.index:
            continue
        base = c.loc[t0]
        for h in horizons:
            d = t0 + pd.DateOffset(months=h)
            if d not in c.index:
                continue
            rows.append({"country": country, "episode": t0.strftime("%Y-%m"), "h": h,
                         **{k: (c.at[d, k] - base[k]) * 100.0 for k in c.columns}})
    return pd.DataFrame(rows)


def _aggregate(e: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Average by horizon, tenor by tenor over a COMMON subset.

    Averaging each column over its own non-null rows breaks the identity: the yield
    change and its two legs then run over slightly different episodes and the split
    fails to add up by a few basis points. A decomposition that does not add up is not
    a decomposition, so each tenor's three series are averaged over the rows where all
    three are present.
    """
    out = {}
    for t in TENORS:
        cols = [f"y{t:g}", f"e{t:g}", f"tp{t:g}"]
        if not all(c in e.columns for c in cols):
            continue
        sub = e.dropna(subset=cols)
        if sub.empty:
            continue
        g = sub.groupby("h")
        for c in cols:
            out[c] = getattr(g[c], stat)()
        out[f"n{t:g}"] = g.size()
    return pd.DataFrame(out)


def summary(country: str = "US", stat: str = "mean") -> pd.DataFrame:
    """Average change across episodes, by horizon, with the episode count.

    ``stat="mean"`` for the DECOMPOSITION: the identity ``y = e + tp`` holds
    observation by observation, so means add up and medians do not -- a median split
    can show legs that fail to sum to the median total, which is not a decomposition.
    ``stat="median"`` is the robustness check on the headline yield change, where no
    adding up is required and one violent episode should not carry the answer.
    """
    e = event_study(country)
    if e.empty:
        return e
    return _aggregate(e, stat)


def pooled(countries: tuple = tuple(UNEMP), stat: str = "mean") -> pd.DataFrame:
    """Every market's episodes in one panel -- the cross-market average response."""
    frames = []
    for c in countries:
        try:
            frames.append(event_study(c))
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("%s skipped (%s)", c, exc)
    e = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return _aggregate(e, stat)


INFL_SPLIT = 4.0      # % core inflation at the episode start


def inflation_at(country: str, dates: list) -> dict:
    """Core inflation, year on year, in the month each episode starts."""
    from macro.fetch_core_cpi import core_inflation_yoy
    c = core_inflation_yoy()
    if country not in c:
        return {}
    s = c[country].dropna()
    out = {}
    for d in dates:
        near = s.index[s.index <= d]
        if len(near):
            out[d] = float(s.loc[near[-1]])
    return out


def by_inflation_regime(countries: tuple = tuple(UNEMP), split: float = INFL_SPLIT,
                        stat: str = "mean") -> dict:
    """The same event study, split by whether inflation was high when it started.

    A growth drawdown is not one object. With inflation already high, the central bank
    cannot ease into it and the premium is being paid for inflation risk, not for
    growth risk -- the 1970s US episodes have yields RISING through the downturn. With
    inflation low, policy eases and yields fall. Averaging the two together produces a
    number that describes neither, and it is exactly the distinction between the
    scenario sheet's Stagflation and its Hard Landing.
    """
    lo, hi = [], []
    for c in countries:
        try:
            eps = episodes(c)
            pi = inflation_at(c, eps)
            e = event_study(c)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("%s skipped (%s)", c, exc)
            continue
        if e.empty:
            continue
        e = e.copy()
        e["pi0"] = e["episode"].map(
            {d.strftime("%Y-%m"): v for d, v in pi.items()})
        lo.append(e[e["pi0"] < split])
        hi.append(e[e["pi0"] >= split])
    out = {}
    for name, parts in (("low inflation", lo), ("high inflation", hi)):
        f = pd.concat([x for x in parts if not x.empty], ignore_index=True) \
            if any(len(x) for x in parts) else pd.DataFrame()
        out[name] = (_aggregate(f.drop(columns=["pi0"]), stat), 
                     sorted(set(zip(f["country"], f["episode"])))) if len(f) \
            else (pd.DataFrame(), [])
    return out


def _fmt(df: pd.DataFrame, cols: list, title: str) -> str:
    have = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not have:
        return f"\n{title}\n  (no data)"
    out = [f"\n{title}", "  h(m)" + "".join(f"{c:>9}" for c in have)]
    for h, r in df.iterrows():
        out.append(f"  {h:>4}" + "".join(
            ("      ." if pd.isna(r[c]) else f"{r[c]:>+9.0f}") for c in have))
    return "\n".join(out)


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    for country in UNEMP:
        try:
            eps = episodes(country)
        except Exception as exc:                                    # noqa: BLE001
            print(f"\n{country}: unemployment unavailable ({exc})")
            continue
        s = summary(country)
        n = event_study(country)["episode"].nunique() if not s.empty else 0
        print(f"\n{'='*72}\n{country}: {len(eps)} drawdowns dated, {n} with curve data")
        print("  " + ", ".join(d.strftime('%Y-%m') for d in eps))
        if s.empty:
            continue
        print(_fmt(s, [f"y{t:g}" for t in TENORS] + ["n10"],
                   "  MEAN change in the OBSERVED yield, bp"))
        print(_fmt(s, [f"e{t:g}" for t in TENORS],
                   "  ...of which the EXPECTED-RATE leg, bp"))
        print(_fmt(s, [f"tp{t:g}" for t in TENORS],
                   "  ...of which the TERM PREMIUM, bp"))
        print(_fmt(summary(country, "median"), [f"y{t:g}" for t in TENORS],
                   "  (median yield change, as a robustness check)"))

    for name, (tab, eps) in by_inflation_regime().items():
        print(f"\n{'='*72}\nDRAWDOWNS STARTING IN {name.upper()} "
              f"(core CPI {'<' if name.startswith('low') else '>='} {INFL_SPLIT:.0f}%)")
        print("  " + ", ".join(f"{c} {d}" for c, d in eps))
        if tab.empty:
            continue
        print(_fmt(tab, [f"y{t:g}" for t in TENORS] + ["n10"],
                   "  MEAN change in the OBSERVED yield, bp"))
        print(_fmt(tab, [f"e{t:g}" for t in TENORS], "  ...of which EXPECTED-RATE, bp"))
        print(_fmt(tab, [f"tp{t:g}" for t in TENORS], "  ...of which TERM PREMIUM, bp"))

    p = pooled()
    print(f"\n{'='*72}\nPOOLED across all markets (both regimes together)")
    print(_fmt(p, [f"y{t:g}" for t in TENORS] + ["n10"],
               "  MEAN change in the OBSERVED yield, bp"))
    print(_fmt(p, [f"e{t:g}" for t in TENORS], "  ...of which EXPECTED-RATE, bp"))
    print(_fmt(p, [f"tp{t:g}" for t in TENORS], "  ...of which TERM PREMIUM, bp"))


if __name__ == "__main__":
    _main()
