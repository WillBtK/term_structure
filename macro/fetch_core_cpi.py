"""Core CPI panel for the six Atlas markets, from FRED.

CORE, not headline, and deliberately so. The object we want is a slow-moving inflation
REGIME — its average level and how much it moves around that average. Headline CPI
injects energy swings that a trailing window cannot distinguish from a genuine change
in the regime, so a headline-based reading would report "uncertainty" that is really
just the oil price. Core is the right series for a level-to-uncertainty map.

Sources. The OECD MEI "all items non-food non-energy" indices give the long history
that actually identifies the relationship — every market spans the 1970s-80s inflation
cycle and the low-inflation decades that followed. The US uses BLS core CPI directly
(CPILFESL), which is both longer and current.

Coverage caveat, stated plainly because it bounds what this data can do: the OECD feed
was discontinued, so most series end in 2025 and Japan's ends 2021-06. That is
immaterial for ESTIMATING the map, which is identified off the high-inflation era, and
it is why anchoring to a currently observed uncertainty reading is a separate step
(``OkunParams.anchored_to``) rather than something read off the tail of this panel.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORE_CPI_PARQUET = HERE / "core_cpi.parquet"

# One FRED id per market. All are indices (not rates), so year-on-year is ours to take.
SERIES = {
    "US": "CPILFESL",          # BLS core CPI, SA, 1957+, current
    "DE": "DEUCPICORMINMEI",   # OECD core, 1962+
    "JP": "JPNCPICORMINMEI",   # OECD core, 1955+ (feed ends 2021-06)
    "UK": "GBRCPICORMINMEI",   # OECD core, 1970+
    "CH": "CHECPICORMINMEI",   # OECD core, 1955+
    "CA": "CANCPICORMINMEI",   # OECD core, 1961+
}


def load_core_cpi(refresh: bool = False) -> pd.DataFrame:
    """Monthly core CPI INDEX level per market, columns = country codes."""
    if not refresh and CORE_CPI_PARQUET.exists():
        return pd.read_parquet(CORE_CPI_PARQUET)
    from validation.fred_api import fred_series
    cols = {}
    for country, sid in SERIES.items():
        try:
            s = fred_series(sid).dropna()
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("core CPI fetch failed for %s (%s): %s", country, sid, exc)
            continue
        s.index = pd.DatetimeIndex(s.index)
        cols[country] = s
        logger.info("  %s %s: %d obs %s..%s", country, sid, len(s),
                    s.index.min().date(), s.index.max().date())
    if not cols:
        raise RuntimeError("no core CPI series could be fetched")
    px = pd.DataFrame(cols).sort_index()
    CORE_CPI_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CORE_CPI_PARQUET)
    return px


def core_inflation_yoy(refresh: bool = False) -> pd.DataFrame:
    """Year-on-year core inflation in PERCENT, per market."""
    px = load_core_cpi(refresh=refresh)
    return (px / px.shift(12) - 1.0) * 100.0


# Two and a half years of effective memory. Profiled against the term premium the
# half-life plateau is broad -- anything from two to four years fits equally well -- so
# the choice is made on behaviour rather than on the third decimal of an R2.
UNCERTAINTY_HALFLIFE = 30


def core_inflation_vol(halflife: int = UNCERTAINTY_HALFLIFE,
                       refresh: bool = False) -> pd.DataFrame:
    """Exponentially weighted volatility of year-on-year core inflation, in PERCENT.

    A rectangular rolling window has the same flaw here that it had in the inflation
    LEVEL, and for a projection it is worse. An observation inside the window counts
    fully, an observation one month older counts nothing, so the measure moves on dates
    when nothing happened -- it moves because something LEFT. The US measure has been
    falling steadily through 2025 and 2026, from 1.53 to 1.25, entirely because the
    2021-22 spike is ageing out of a sixty-month window, and it will step down again when
    the peak months themselves drop out. A model reading that would forecast a falling
    term premium on the strength of a calendar.

    Exponential weighting has no edge to fall off. Old observations fade rather than
    vanish, so the measure only moves when inflation does. Across the four markets it
    roughly halves the largest artificial month-on-month jump (the UK's worst falls from
    0.37pp to 0.12pp), and it fits marginally better besides.

    Volatility is measured about the series' OWN exponentially weighted mean rather than
    a full-sample constant, so a period of high but steady inflation reads as certain --
    which is the distinction the term premium actually prices.
    """
    infl = core_inflation_yoy(refresh=refresh)
    mean = infl.ewm(halflife=halflife).mean()
    return np.sqrt(((infl - mean) ** 2).ewm(halflife=halflife).mean())


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    infl = core_inflation_yoy(refresh=True)
    print(f"\ncore CPI panel: {infl.shape[1]} markets, {infl.index.min().date()}"
          f"..{infl.index.max().date()}")
    print("\nDecade means of YoY core inflation (%):")
    print(infl.resample("10YS").mean().round(2).to_string())
    vol = core_inflation_vol()
    print(f"\nEWMA inflation uncertainty (halflife {UNCERTAINTY_HALFLIFE}m), latest:")
    print(vol.dropna(how="all").iloc[-1].round(2).to_string())


if __name__ == "__main__":
    _main()
