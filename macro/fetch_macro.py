"""Equity prices and industrial production for the six Atlas markets, from FRED.

These are the two series the stock-bond-correlation channel needs beyond core CPI:
share prices to build the equity leg of the correlation, and a real-activity series to
measure growth volatility.

Why industrial production rather than GDP. The relationship is identified off the
1970s-90s, when the inflation and growth regimes actually changed. FRED's real GDP
series start in 1991-1995 for Germany, Japan and Switzerland — after the interesting
part — whereas OECD industrial production runs from the late 1950s for five of the six.
IP is a noisier proxy for "growth" than GDP, but a noisy proxy over the full regime
history beats a clean one that omits it. UK IP is the exception (1997+), so the UK
sample is short and its estimates should be read accordingly.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
SHARE_PRICES_PARQUET = HERE / "share_prices.parquet"
IND_PROD_PARQUET = HERE / "ind_prod.parquet"

# OECD "Financial Market: Share Prices", monthly index, 1955-1960 starts, current.
SHARE_PRICE_SERIES = {
    "US": "SPASTT01USM661N", "DE": "SPASTT01DEM661N", "JP": "SPASTT01JPM661N",
    "UK": "SPASTT01GBM661N", "CH": "SPASTT01CHM661N", "CA": "SPASTT01CAM661N",
}

# OECD industrial production, quarterly index.
IND_PROD_SERIES = {
    "US": "PRINTO01USQ661N", "DE": "PRINTO01DEQ661N", "JP": "PRINTO01JPQ661N",
    "UK": "PRINTO01GBQ661N", "CH": "PRINTO01CHQ661N", "CA": "PRINTO01CAQ661N",
}


def _load(series: dict, path: Path, refresh: bool) -> pd.DataFrame:
    if not refresh and path.exists():
        return pd.read_parquet(path)
    from validation.fred_api import fred_series
    cols = {}
    for country, sid in series.items():
        try:
            s = fred_series(sid).dropna()
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("fetch failed for %s (%s): %s", country, sid, exc)
            continue
        s.index = pd.DatetimeIndex(s.index)
        cols[country] = s
    if not cols:
        raise RuntimeError(f"no series fetched for {path.name}")
    df = pd.DataFrame(cols).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def load_share_prices(refresh: bool = False) -> pd.DataFrame:
    """Monthly equity price index per market."""
    return _load(SHARE_PRICE_SERIES, SHARE_PRICES_PARQUET, refresh)


def load_ind_prod(refresh: bool = False) -> pd.DataFrame:
    """Quarterly industrial production index per market."""
    return _load(IND_PROD_SERIES, IND_PROD_PARQUET, refresh)


def growth_yoy_monthly(refresh: bool = False) -> pd.DataFrame:
    """Year-on-year industrial production growth in PERCENT, interpolated to monthly.

    Interpolation is `limit_area="inside"` so it never extrapolates past the ends of the
    quarterly series — a projected growth rate beyond the data would be an invention,
    and the trailing windows built on top would silently inherit it.
    """
    ip = load_ind_prod(refresh=refresh)
    g = (ip / ip.shift(4) - 1.0) * 100.0
    return g.resample("MS").interpolate(limit_area="inside")
