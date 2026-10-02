"""Real GDP, quarterly, year-on-year, for the four scenario markets.

The scenario sheet's growth row is the calendar-year average of quarterly y/y rates.
Once part of the current year is published, a scenario that states a rate for the
REST of the year (Stagflation, Hard Landing: "Q3-Q4 at -1.0%") needs the published
quarters to turn that into the annual figure the engine runs on. The Bund market's
growth is the euro area's, as its inflation is.

    US  BEA real GDP, SA                     FRED GDPC1
    UK  ONS real GDP, SA                     FRED NGDPRSAXDCGBQ
    DE  euro-area (EA20) GDP, chain-linked,  Eurostat namq_10_gdp, CLV20_MEUR, SCA
        seasonally and calendar adjusted
    JP  Cabinet Office real GDP, SA          FRED JPNRGDPEXP

    python -m macro.fetch_quarterly_gdp        # refresh the committed cache

Best-effort like every fetcher: a failed source keeps the cached series.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "quarterly_gdp_yoy.parquet"
MARKETS = ("US", "UK", "DE", "JP")
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_FRED_IDS = {"US": "GDPC1", "UK": "NGDPRSAXDCGBQ", "JP": "JPNRGDPEXP"}
_EUROSTAT_EA = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
                "namq_10_gdp?geo=EA20&na_item=B1GQ&s_adj=SCA&unit=CLV20_MEUR"
                "&sinceTimePeriod=2015-Q1&format=JSON&lang=EN")


def _fred(sid: str) -> pd.Series:
    import requests
    r = requests.get(_FRED.format(sid), timeout=90)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    x.columns = ["date", "v"]
    return pd.to_numeric(x.set_index(pd.to_datetime(x["date"]))["v"], errors="coerce").dropna()


def _eurostat_ea() -> pd.Series:
    import requests
    j = requests.get(_EUROSTAT_EA, timeout=90).json()
    idx = {v: k for k, v in j["dimension"]["time"]["category"]["index"].items()}
    return pd.Series({pd.Period(idx[int(k)].replace("-", ""), "Q").to_timestamp(): float(v)
                      for k, v in j["value"].items()}).sort_index()


def _yoy(level: pd.Series) -> pd.Series:
    level = level.sort_index()
    return ((level / level.shift(4) - 1.0) * 100.0).dropna()


def fetch(out: Path = CACHE) -> pd.DataFrame | None:
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    got = {}
    for c in MARKETS:
        try:
            got[c] = _yoy(_eurostat_ea() if c == "DE" else _fred(_FRED_IDS[c]))
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("quarterly GDP %s: %s -- keeping cached series", c, exc)
            if c in have:
                got[c] = have[c].dropna()
    if not got:
        return None
    df = pd.DataFrame(got).sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_parquet(out)
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    if refresh or not CACHE.exists():
        df = fetch()
        if df is not None:
            return df
    return pd.read_parquet(CACHE)


def year_to_date(country: str, year: int, df: pd.DataFrame | None = None) -> dict:
    """The quarters of ``year`` already published: their y/y rates, in order."""
    df = load() if df is None else df
    if country not in df:
        return {"year": int(year), "n": 0, "yoy": [], "last": None}
    s = df[country].dropna()
    s = s[pd.DatetimeIndex(s.index).year == int(year)]
    return {"year": int(year), "n": int(len(s)), "yoy": [round(float(v), 3) for v in s],
            "last": str(pd.Timestamp(s.index.max()).to_period("Q")) if len(s) else None}


def annual_from_remaining(rate: float, ytd: dict) -> float:
    """The calendar-year average y/y when the unpublished quarters run at ``rate``."""
    n = int(ytd.get("n") or 0)
    if n <= 0 or n >= 4:
        return float(rate)
    return (sum(ytd["yoy"]) + (4 - n) * float(rate)) / 4.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = fetch()
    if d is not None:
        for c in MARKETS:
            print(c, year_to_date(c, pd.Timestamp.today().year, d))
