"""Headline CPI, year-on-year, SEASONALLY ADJUSTED, for the four scenario markets.

The scenario sheet's inflation row is a CALENDAR-YEAR AVERAGE of monthly year-on-year
rates (the average of Jan y/y, Feb y/y, ... Dec y/y). Once part of the current year is
published, only the remaining months are still a forecast, and they must be priced at
the rate that delivers the stated average given the months already in. Year-on-year rates
of seasonally adjusted indices, by market (the German Bund market's inflation is the
euro area's, the ECB's target):

    US  BLS CPI-U, all items, SA                FRED CPIAUCSL (index)
    DE  euro-area HICP, overall, SA             ECB data portal HICP/M.U2.Y.000000.4F0.INX
    JP  CPI, all items, SA                      OECD SDMX (COICOP 2018), JPN.M.N.CPI.IX._T.S._Z
    UK  CPI, all items: the ONS publishes NO     ONS D7BT (index), adjusted here: STL on the
        seasonally adjusted CPI                  log index, period 12, seasonal window 13,
                                                 robust (statsmodels)

Year-on-year rates of SA and unadjusted indices differ only as far as seasonal factors
shift from one year to the next; Jan-Aug 2026 the gap is 0.01pp or less except the US,
where BLS's adjusted series also smooths the October 2025 gap (no CPI was published).

    python -m macro.fetch_headline_cpi        # refresh the committed cache

Best-effort like every fetcher: a failed source keeps the cached series.
"""
from __future__ import annotations

import io
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "headline_cpi_yoy.parquet"
MARKETS = ("US", "UK", "DE", "JP")
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_FRED_SA = {"US": "CPIAUCSL"}
_ECB_EA_SA = ("https://data-api.ecb.europa.eu/service/data/HICP/M.U2.Y.000000.4F0.INX"
              "?format=csvdata")
_OECD_JP_SA = ("https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,"
               "DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL,1.0/JPN.M.N.CPI.IX._T.S._Z")
_ONS_INDEX = ("https://www.ons.gov.uk/generator?format=csv&uri=/economy/"
              "inflationandpriceindices/timeseries/d7bt/mm23")
UK_STL_SEASONAL = 13
_UA = "Mozilla/5.0"


def _curl(url: str, accept: str | None = None) -> bytes:
    cmd = ["curl", "-sSL", "--fail", "--max-time", "90", "-A", _UA]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    out = subprocess.run(cmd + [url], capture_output=True, timeout=150)
    if out.returncode != 0 or not out.stdout:
        raise RuntimeError(f"fetch failed ({out.returncode}): {url}")
    return out.stdout


def _yoy(ix: pd.Series) -> pd.Series:
    ix = ix.sort_index()
    return ((ix / ix.shift(12) - 1.0) * 100.0).dropna()


def _fred_index(sid: str) -> pd.Series:
    import requests                       # FRED drops curl's HTTP/2 stream with a UA set
    r = requests.get(_FRED.format(sid), timeout=90)
    r.raise_for_status()
    x = pd.read_csv(io.StringIO(r.text))
    x.columns = ["date", "v"]
    return pd.to_numeric(x.set_index(pd.to_datetime(x["date"]))["v"], errors="coerce").dropna()


def _ecb_ea_sa() -> pd.Series:
    d = pd.read_csv(io.BytesIO(_curl(_ECB_EA_SA)))[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    return pd.Series(d["OBS_VALUE"].to_numpy(float), index=pd.to_datetime(d["TIME_PERIOD"]))


def _oecd_jp_sa() -> pd.Series:
    raw = _curl(_OECD_JP_SA, accept="application/vnd.sdmx.data+csv;version=1.0.0")
    d = pd.read_csv(io.BytesIO(raw))[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    idx = pd.PeriodIndex(d["TIME_PERIOD"].astype(str), freq="M").to_timestamp()
    return pd.Series(d["OBS_VALUE"].to_numpy(float), index=idx)


def _ons_index() -> pd.Series:
    rows = []
    for line in _curl(_ONS_INDEX).decode("utf-8", "ignore").splitlines():
        m = re.match(r'^"(\d{4}) ([A-Z]{3})","([-\d.]+)"', line)
        if m:
            rows.append((pd.to_datetime(f"{m.group(1)} {m.group(2)}", format="%Y %b"),
                         float(m.group(3))))
    if len(rows) < 120:
        raise RuntimeError(f"ONS D7BT: only {len(rows)} monthly rows")
    return pd.Series(dict(rows)).sort_index()


def seasonally_adjust(ix: pd.Series, seasonal: int = UK_STL_SEASONAL) -> pd.Series:
    """STL on the log index (period 12, robust): the index with its seasonal removed."""
    import numpy as np
    from statsmodels.tsa.seasonal import STL
    s = ix.sort_index().asfreq("MS").interpolate(limit_area="inside")
    lg = np.log(s)
    fit = STL(lg, period=12, seasonal=seasonal, robust=True).fit()
    return np.exp(lg - fit.seasonal)


def fetch(out: Path = CACHE) -> pd.DataFrame | None:
    """Refresh every market; a market whose source fails keeps its cached series."""
    have = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    got = {}
    for c, f in (("US", lambda: _yoy(_fred_index(_FRED_SA["US"]))),
                 ("DE", lambda: _yoy(_ecb_ea_sa())),            # euro area, not Germany
                 ("JP", lambda: _yoy(_oecd_jp_sa())),
                 ("UK", lambda: _yoy(seasonally_adjust(_ons_index())))):
        try:
            got[c] = f()
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("headline CPI %s: %s -- keeping cached series", c, exc)
            if c in have:
                got[c] = have[c].dropna()
    if not got:
        return None
    df = pd.DataFrame(got).sort_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_parquet(out)
    for c in df:
        s = df[c].dropna()
        logger.info("headline CPI %s: %s..%s, latest %.2f%%", c, s.index.min().date(),
                    s.index.max().date(), s.iloc[-1])
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    if refresh or not CACHE.exists():
        df = fetch()
        if df is not None:
            return df
    return pd.read_parquet(CACHE)


def year_to_date(country: str, year: int, df: pd.DataFrame | None = None) -> dict:
    """The months of ``year`` already published: count, mean y/y, and the last month."""
    df = load() if df is None else df
    if country not in df:
        return {"year": int(year), "n": 0, "mean": None, "last": None}
    s = df[country].dropna()
    s = s[pd.DatetimeIndex(s.index).year == int(year)]
    return {"year": int(year), "n": int(len(s)),
            "mean": round(float(s.mean()), 4) if len(s) else None,
            "last": str(s.index.max().date()) if len(s) else None}


def remaining_rate(annual_avg: float, ytd: dict) -> float:
    """The average y/y the unpublished months must run at for the year to average
    ``annual_avg``: (12*A - sum of published months) / months remaining. A year with
    nothing published returns ``annual_avg``; a complete year has nothing to price."""
    n = int(ytd.get("n") or 0)
    if n <= 0 or ytd.get("mean") is None:
        return float(annual_avg)
    if n >= 12:
        return float(annual_avg)
    return (12.0 * float(annual_avg) - n * float(ytd["mean"])) / (12 - n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = fetch()
    if d is not None:
        for c in MARKETS:
            print(c, year_to_date(c, pd.Timestamp.today().year, d))
