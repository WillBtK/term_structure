"""Long-run inflation EXPECTATIONS per market, kept current from public sources.

Not an EWMA of realised inflation. That distinction is the whole point: a discounted
moving average of past core CPI reads the mid-1970s as 14.8%, which no investor ever
expected to persist, and overshoots again after COVID on a spike long-horizon
expectations barely registered. What prices a ten-year bond is what holders expect
inflation to AVERAGE over its life, and that is a survey object.

The best long-horizon surveys are proprietary. Consensus Economics 6-10y CPI is the
anchor for three of the four markets and reaches back to 1990, but it arrives as a
one-off snapshot: nobody re-buys it monthly, so left alone it silently freezes at
whatever date it was last refreshed and the model quietly stops seeing the present.

So each market runs the same DEGRADATION LADDER the neutral-rate anchors use. The
proprietary snapshot carries the history; a public, machine-readable series carries the
current edge; and the two are joined by a level calibration over the months they
overlap, so the handover leaves no step to be misread as a change of view.

    US   SPF 10-year CPI (public, quarterly)   <-  Cleveland Fed 10y model, both ends
    DE   Consensus 6-10y                       <-  ECB SPF longer-term
    UK   Consensus 6-10y                       <-  BoE Market Participants Survey, CPI
                                                   five years ahead
    JP   Consensus 6-10y                       <-  nothing public

Japan has no rung, and the module says so out loud rather than freezing quietly: past
the staleness window it logs a warning naming the last observation. Two things make that
tolerable rather than a hole. The Bank of Japan publishes its Tankan inflation outlook
only through a search interface with no stable file, and no Japanese long-horizon survey
is both public and machine-readable. And Japan's inflation coefficients are borrowed from
the US anyway -- six of its 388 monthly observations sit above the inflation target -- so
its expectations series carries almost no weight in the fit. If that changes, the rung is
one entry in EXTENDERS.

These are the same anchors the term-premium decomposition uses for its shifting
endpoints, so the driver model and the premium it explains read the same view of
long-run inflation rather than two different ones.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# The anchor: longest history, proprietary where it has to be. First entry wins.
SURVEYS = {
    "US": [("us/data/anchors/spf_cpi10.csv", "SPF 10y CPI")],
    "DE": [("germany/data/anchors/consensus_cpi_de_6_10y.csv", "Consensus 6-10y")],
    "JP": [("japan/data/anchors/consensus_cpi_6_10y.csv", "Consensus 6-10y")],
    "UK": [("uk/data/anchors/consensus_cpi_uk_6_10y.csv", "Consensus 6-10y")],
}

# The public rungs, refreshed by .github/workflows/refresh-anchors.yml. Each is level-
# calibrated to the anchor over their overlap and used only OUTSIDE the anchor's span --
# before it starts and after it ends. A FRED id is fetched; a path is a committed CSV.
EXTENDERS = {
    "US": [("EXPINF10YR", "Cleveland Fed 10y expected inflation")],
    "DE": [("germany/data/anchors/ecb_spf_infl_lt.csv", "ECB SPF longer-term")],
    "UK": [("uk/data/anchors/boe_mps_cpi_5y.csv", "BoE MPS, CPI five years ahead")],
    "JP": [],
}

# How long an anchor may sit unrefreshed before the absence of a rung is worth saying.
# Consensus rounds are twice-yearly, so a year and a half is two missed rounds.
STALENESS_DAYS = 550
MIN_OVERLAP = 12


def _read(rel: str) -> pd.Series:
    df = pd.read_csv(ROOT / rel)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"]))).dropna()
    return s.sort_index()


def _load(source: str) -> "pd.Series | None":
    """A rung: a committed CSV path, or a FRED series id."""
    try:
        if "/" in source:
            path = ROOT / source
            return _read(source) if path.is_file() else None
        from validation.fred_api import fred_series
        return fred_series(source).dropna()
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("inflation expectations: rung %s unavailable (%s)", source, exc)
        return None


def _bridge(base: pd.Series, rung: pd.Series, label: str, country: str,
            min_overlap: int = MIN_OVERLAP) -> pd.Series:
    """Extend ``base`` outside its own span with ``rung``, level-matched on the overlap.

    The two series measure the same concept on different bases and at different horizons
    -- a five-year-ahead survey is not a six-to-ten-year one -- and a step at the join
    would be read as a change of view rather than a change of source. Subtracting the
    mean gap over the overlap removes it. Inside the anchor's own span nothing changes.
    """
    rung = rung.resample("MS").mean()
    both = pd.concat([base.rename("a"), rung.rename("b")], axis=1).dropna()
    if len(both) < min_overlap:
        logger.info("  %s: %s overlaps the anchor by only %d months (<%d) -- not used",
                    country, label, len(both), min_overlap)
        return base
    off = float((both["a"] - both["b"]).mean())
    outside = rung[(rung.index < base.index.min()) | (rung.index > base.index.max())]
    if outside.empty:
        return base
    fwd = outside[outside.index > base.index.max()]
    logger.info("  %s: %s bridged (offset %+.2fpp, overlap %d months, corr %.2f)%s",
                country, label, off, len(both), both["a"].corr(both["b"]),
                f", extends to {fwd.index.max().date()}" if len(fwd) else ", back only")
    return pd.concat([base, outside + off]).sort_index()


def expectations(country: str, index: pd.DatetimeIndex | None = None) -> pd.Series:
    """Monthly long-run expected inflation in PERCENT.

    Surveys are irregular -- quarterly, sometimes twice-yearly -- so they are placed on a
    month-start grid and carried forward. A survey view is the prevailing view until the
    next round replaces it, which is exactly what forward-filling encodes.
    """
    parts = [_read(p) for p, _ in SURVEYS.get(country, []) if (ROOT / p).is_file()]
    if not parts:
        raise KeyError(f"no committed survey anchor for {country}")
    s = parts[0]
    for extra in parts[1:]:
        s = s.combine_first(extra)
    anchor_end = s.index.max()
    s = s.resample("MS").mean().ffill()

    for source, label in EXTENDERS.get(country, []):
        rung = _load(source)
        if rung is not None and not rung.empty:
            s = _bridge(s, rung, label, country)
    s = s.resample("MS").mean().ffill()

    stale = (pd.Timestamp.today() - max(anchor_end, s.index.max())).days
    if stale > STALENESS_DAYS:
        logger.warning("%s inflation expectations are STALE: last observation %s, "
                       "%d days old, and no public rung reaches further. The anchor "
                       "needs a manual refresh.", country, s.index.max().date(), stale)
    s.name = country
    return s if index is None else s.reindex(index).ffill()


def panel(markets=("US", "UK", "DE", "JP")) -> pd.DataFrame:
    return pd.DataFrame({c: expectations(c) for c in markets})


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = panel()
    print("\nLong-run expected inflation, per market:")
    for c in p:
        s = p[c].dropna()
        anchor = _read(SURVEYS[c][0][0])
        print(f"  {c}: {s.index.min().date()}..{s.index.max().date()}  "
              f"latest {s.iloc[-1]:.2f}%   anchor ends {anchor.index.max().date()}"
              f"   rungs: {', '.join(l for _, l in EXTENDERS[c]) or 'none'}")


if __name__ == "__main__":
    _main()
