"""Core CPI panel for the six Atlas markets, from FRED.

CORE, not headline, and deliberately so. The object we want is a slow-moving inflation
REGIME — its average level and how much it moves around that average. Headline CPI
injects energy swings that a trailing window cannot distinguish from a genuine change
in the regime, so a headline-based reading would report "uncertainty" that is really
just the oil price. Core is the right series for a level-to-uncertainty map.

Sources. The US uses BLS core CPI directly (CPILFESL via FRED). Everything else comes
from the OECD Data Explorer's SDMX API, which serves the year-on-year rate of "all
items excluding food and energy" directly — long history AND current, every market
spanning the 1970s-80s inflation cycle and the low-inflation decades after it.

That OECD route replaced FRED's mirrors of the discontinued OECD MEI, and the reason
is worth recording because the staleness was not harmless. The MEI mirrors froze:
Germany and the UK at 2025-03, Japan at **2021-06** with a last reading of -1.0%.
For ESTIMATING the level-to-uncertainty map that was immaterial — the map is
identified off the high-inflation era — but the scenario engine reads the tail of
this panel to decide what "inflation continuing as it is" means, and a five-year-old
-1.0% is not that. A Japanese scenario of 2% inflation was being scored as a three-
point departure from a status quo that had not existed since 2021.

Japan is served by the COICOP-2018 flow and the others by the COICOP-1999 one; the
OECD has not migrated every country to the newer classification, so both are queried
and the one reaching FURTHEST is used. Taking whichever answers first is not enough:
the 1999 flow still answers for Japan, with 786 observations frozen at the same
2021-06 reading, and would have quietly reinstated exactly the staleness this
replaced.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
CORE_CPI_PARQUET = HERE / "core_cpi.parquet"

# FRED indices (not rates), so year-on-year is ours to take. Retained for the US, whose
# BLS series is both longer and current, and as the fallback if OECD is unreachable.
SERIES = {
    "US": "CPILFESL",          # BLS core CPI, SA, 1957+, current
    "DE": "DEUCPICORMINMEI",   # OECD MEI mirror, frozen 2025-03
    "JP": "JPNCPICORMINMEI",   # OECD MEI mirror, frozen 2021-06
    "UK": "GBRCPICORMINMEI",   # OECD MEI mirror, frozen 2025-03
    "CH": "CHECPICORMINMEI",
    "CA": "CANCPICORMINMEI",
}

# OECD Data Explorer: the year-on-year RATE of all items excluding food and energy.
OECD_API = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS"
OECD_FLOWS = ("DSD_PRICES@DF_PRICES_ALL",                    # COICOP 1999
              "DSD_PRICES_COICOP2018@DF_PRICES_C2018_ALL")   # COICOP 2018 (Japan)
OECD_KEY = "{iso}.M.N.CPI.PA._TXCP01_NRG.N.GY"
OECD_ISO = {"DE": "DEU", "JP": "JPN", "UK": "GBR", "CH": "CHE", "CA": "CAN"}
OECD_RATES_PARQUET = HERE / "core_cpi_rates.parquet"
_TIMEOUT = 90


def _oecd_rate(country: str) -> "pd.Series | None":
    """Year-on-year core inflation in PERCENT, straight from the OECD, or None."""
    import io
    import subprocess
    iso = OECD_ISO.get(country)
    if iso is None:
        return None
    best = None
    for flow in OECD_FLOWS:
        url = f"{OECD_API},{flow},1.0/" + OECD_KEY.format(iso=iso)
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT), "-A",
                 "Mozilla/5.0", "-H",
                 "Accept: application/vnd.sdmx.data+csv;version=1.0.0", url],
                capture_output=True, timeout=_TIMEOUT + 30)
            if out.returncode != 0 or not out.stdout:
                continue
            df = pd.read_csv(io.BytesIO(out.stdout))
            if "OBS_VALUE" not in df:
                continue
            df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
            idx = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="M").to_timestamp()
            s = pd.Series(df["OBS_VALUE"].to_numpy(float), index=idx).sort_index()
            if len(s) > 100 and (best is None or s.index.max() > best.index.max()):
                best = s
        except Exception as exc:                                # noqa: BLE001
            logger.debug("OECD %s/%s: %s", country, flow, exc)
    return None if best is None else best.rename(country)


def oecd_core_rates(markets=tuple(OECD_ISO), refresh: bool = False) -> pd.DataFrame:
    """Cached panel of OECD year-on-year core rates."""
    if not refresh and OECD_RATES_PARQUET.is_file():
        return pd.read_parquet(OECD_RATES_PARQUET)
    cols = {}
    for c in markets:
        s = _oecd_rate(c)
        if s is not None:
            cols[c] = s
            logger.info("OECD core %s: %d obs, %s..%s, last %.2f%%", c, len(s),
                        s.index.min().date(), s.index.max().date(), s.iloc[-1])
        else:
            logger.warning("OECD core %s unavailable -- falling back to the FRED "
                           "index mirror, which is stale", c)
    df = pd.DataFrame(cols)
    if not df.empty:
        df.to_parquet(OECD_RATES_PARQUET)
    return df


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
    """Year-on-year core inflation in PERCENT, per market.

    The OECD rate is preferred wherever it is available, since the FRED mirrors of the
    discontinued MEI are frozen years back. The index-differenced fallback fills any
    market OECD does not answer for, and the US keeps its BLS series throughout.
    """
    px = load_core_cpi(refresh=refresh)
    fred = (px / px.shift(12) - 1.0) * 100.0
    try:
        oecd = oecd_core_rates(refresh=refresh)
    except Exception as exc:                                    # noqa: BLE001
        logger.warning("OECD core rates unavailable (%s); using FRED mirrors", exc)
        return fred
    out = fred.copy()
    for c in oecd.columns:
        if c in out:
            # Splice rather than replace: OECD covers the whole span for these markets,
            # but keep any earlier FRED history it does not reach.
            out[c] = oecd[c].reindex(out.index.union(oecd.index)).combine_first(
                out[c].reindex(out.index.union(oecd.index)))
    return out.sort_index()


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
