"""Long-run inflation EXPECTATIONS per market — the regressor the premium model uses.

Not an EWMA of realised inflation. That distinction is the whole point.

A discounted moving average of past core CPI, in the Cieslak-Povala spirit, is the wrong
object here and fails in both directions. It reads the mid-1970s as 14.8% inflation,
which no investor ever expected to persist, and so implies a term premium far above
anything the market priced. And it overshoots again after COVID, carrying a spike that
long-horizon expectations barely registered. What prices a ten-year bond is what holders
expect inflation to AVERAGE over its life, and that is a survey object, not a lagged
average of what already happened.

Sources, in the order each market prefers them:

* **US** — Survey of Professional Forecasters 10-year CPI (1991-11+), extended back with
  the Cleveland Fed's 10-year expected-inflation model (1982-01+). Furthest back of the
  four, which is why the US carries most of the identification.
* **DE** — Consensus Economics 6-10y CPI (1990+), with the ECB SPF long-term series
  available alongside it.
* **JP** — Consensus Economics 6-10y CPI (1990+).
* **UK** — Consensus Economics 6-10y CPI (2004+).

These are the same anchors the term-premium decomposition itself uses for its shifting
endpoints, so the driver model and the premium it explains are reading the same view of
long-run inflation rather than two different ones.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Committed survey anchors: (path, label). First entry wins where they overlap.
SURVEYS = {
    "US": [("us/data/anchors/spf_cpi10.csv", "SPF 10y CPI")],
    "DE": [("germany/data/anchors/consensus_cpi_de_6_10y.csv", "Consensus 6-10y"),
           ("germany/data/anchors/ecb_spf_infl_lt.csv", "ECB SPF long-term")],
    "JP": [("japan/data/anchors/consensus_cpi_6_10y.csv", "Consensus 6-10y")],
    "UK": [("uk/data/anchors/consensus_cpi_uk_6_10y.csv", "Consensus 6-10y")],
}

# Model-based back-extension, used only BEFORE the survey starts and only where the two
# can be level-matched on a decent overlap.
BACKFILL_FRED = {"US": "EXPINF10YR"}


def _read(rel: str) -> pd.Series:
    df = pd.read_csv(ROOT / rel)
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").values,
                  index=pd.DatetimeIndex(pd.to_datetime(df["date"]))).dropna()
    return s.sort_index()


def expectations(country: str, index: pd.DatetimeIndex | None = None,
                 min_overlap: int = 24) -> pd.Series:
    """Monthly long-run expected inflation in PERCENT.

    Surveys are irregular (quarterly, sometimes twice-yearly), so they are placed on a
    month-start grid and carried forward — a survey view is the prevailing view until the
    next round replaces it, which is exactly what forward-filling encodes.
    """
    parts = [_read(p) for p, _ in SURVEYS.get(country, []) if (ROOT / p).is_file()]
    if not parts:
        raise KeyError(f"no committed survey anchor for {country}")
    s = parts[0]
    for extra in parts[1:]:                      # fill gaps only, never override
        s = s.combine_first(extra)
    s = s.resample("MS").mean().ffill()

    fred_id = BACKFILL_FRED.get(country)
    if fred_id:
        from validation.fred_api import fred_series
        try:
            m = fred_series(fred_id).dropna().resample("MS").mean()
        except Exception as exc:                                # noqa: BLE001
            logger.warning("%s back-extension (%s) unavailable: %s", country, fred_id, exc)
        else:
            both = pd.concat([s.rename("survey"), m.rename("model")], axis=1).dropna()
            if len(both) >= min_overlap:
                # Level-match on the overlap: the two measure the same concept but not on
                # the same basis, and a step at the join would be read as a regime change.
                off = float((both["survey"] - both["model"]).mean())
                early = (m + off)[m.index < s.index.min()]
                logger.info("  %s: %s back-extension from %s (offset %+.2fpp, "
                            "overlap %d months, corr %.2f)", country, fred_id,
                            early.index.min().date() if len(early) else "n/a", off,
                            len(both), both["survey"].corr(both["model"]))
                s = pd.concat([early, s]).sort_index()
    s.name = country
    return s if index is None else s.reindex(index).ffill()


def panel(markets=("US", "UK", "DE", "JP")) -> pd.DataFrame:
    return pd.DataFrame({c: expectations(c) for c in markets})
