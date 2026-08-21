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
#
# The UK is the exception. Its OECD quarterly series (PRINTO01GBQ661N) only starts in
# 1997, which — after a 10-year rolling window — leaves a UK sample beginning in 2007,
# entirely inside one stock-bond-correlation regime and therefore uninformative about the
# regime shift the model exists to capture. GBRPROINDMISMEI is the same concept (total
# industry production volume) back to 1948. On the 1998-2024 overlap the two agree
# closely on year-on-year growth: correlation 0.923, mean gap +0.006pp.
#
# It is MONTHLY where the others are quarterly, and monthly year-on-year growth is more
# volatile than quarterly (4.41 vs 3.86 on the overlap). Since sigma_g enters the model
# as a RATIO, leaving that difference in would hand the UK a systematically inflated
# growth volatility and a correspondingly deflated ratio. So it is resampled to quarterly
# first and then flows through exactly the same transformation as every other market.
IND_PROD_SERIES = {
    "US": "PRINTO01USQ661N", "DE": "PRINTO01DEQ661N", "JP": "PRINTO01JPQ661N",
    "UK": "GBRPROINDMISMEI", "CH": "PRINTO01CHQ661N", "CA": "PRINTO01CAQ661N",
}

# Series that arrive monthly and must be collapsed to quarterly for comparability.
_MONTHLY_IND_PROD = {"UK"}


def _load(series: dict, path: Path, refresh: bool,
          to_quarterly: frozenset = frozenset()) -> pd.DataFrame:
    """Fetch each series and assemble one frame.

    ``to_quarterly`` names series that arrive MONTHLY and must be collapsed to quarters
    BEFORE assembly. Doing it afterwards would be a silent disaster: a monthly index
    unioned into a quarterly frame leaves every other column NaN on the intervening
    months, and the downstream year-on-year transform is a positional ``shift(4)`` — so
    on a mixed index "four rows back" silently stops meaning "four quarters back" for
    every market, not just the monthly one.
    """
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
        if country in to_quarterly:
            s = s.resample("QS").mean().dropna()
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
    """Quarterly industrial production index per market.

    Markets in ``_MONTHLY_IND_PROD`` are sourced monthly and averaged to quarters, so
    every column carries the same sampling and their growth volatilities are comparable.
    """
    return _load(IND_PROD_SERIES, IND_PROD_PARQUET, refresh,
                 to_quarterly=frozenset(_MONTHLY_IND_PROD))


def growth_yoy_monthly(refresh: bool = False) -> pd.DataFrame:
    """Year-on-year industrial production growth in PERCENT, interpolated to monthly.

    Interpolation is `limit_area="inside"` so it never extrapolates past the ends of the
    quarterly series — a projected growth rate beyond the data would be an invention,
    and the trailing windows built on top would silently inherit it.
    """
    ip = load_ind_prod(refresh=refresh)
    g = (ip / ip.shift(4) - 1.0) * 100.0
    return g.resample("MS").interpolate(limit_area="inside")
