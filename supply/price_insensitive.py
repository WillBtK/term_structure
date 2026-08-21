"""Price-insensitive ownership, per market, as a share of GDP.

The US result is that only the PRICE-INSENSITIVE holder compresses the term premium:
foreign official enters at -0.064 (t=-12.6) while foreign private enters at -0.029
(t=-1.2), and the official coefficient lands within a whisker of the Federal Reserve's
own. A reserve manager and a central bank both hold the bond for a reason that has
nothing to do with its yield; a private foreign investor responds to the price rather
than setting it. So the driver is not "how much paper is held abroad" -- it is how much
of the stock sits with someone who does not care what it costs.

Carrying that to the other three markets means finding each one's equivalent holder,
and they are not the same holder:

    US   Federal Reserve       + foreign official (Z.1)
    JP   Bank of Japan         + COFER yen reserves
    DE   Eurosystem            (PSPP + PEPP holdings of German paper)
    UK   Bank of England APF   + overseas (ONS)

Two deliberate choices.

**Japan uses COFER's yen share applied to TOTAL reserves, not to allocated reserves.**
The allocated series covers only the reporters who disclose their currency composition,
and that reporter set grows: a rise in allocated yen holdings can be new reporters
rather than new buying. Total reserves are the honest denominator.

**The UK measure is overseas holdings in aggregate, not reserve managers alone.** ONS
publishes the full ownership decomposition -- the sum equals the stock -- but it does
not split the rest-of-the-world sector into official and private. So the UK number
mixes the two in a way the US number does not, and its coefficient should be read as
correspondingly attenuated. Stated rather than hidden.

Germany is the cleanest of the four and the shortest: the Eurosystem held essentially
no German government paper before PSPP began in March 2015, so a series that is zero
until then and cumulates net purchases thereafter is not an approximation -- it is what
happened.
"""
from __future__ import annotations

import io
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from macro.fetch_ons import ons_series
from validation.fred_api import fred_series

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ECB_PSPP = "https://www.ecb.europa.eu/mopo/pdf/PSPP_breakdown_history.csv"
ECB_PEPP = "https://www.ecb.europa.eu/mopo/pdf/PEPP_public_sector_securities_breakdown_history.csv"
IMF_SDMX = "https://api.imf.org/external/sdmx/2.1/data"
COFER = f"{IMF_SDMX}/IMF.STA,COFER,7.0.1"

# Nominal GDP. The US pair is already handled inside model.tp_drivers; these three are
# the denominators for everything below. DE and UK are quarterly LEVELS and so are put
# on an annual rate by *4; JP is published at an annual rate already.
GDP = {"JP": ("JPNNGDP", 1.0), "DE": ("CPMNACSCAB1GQDE", 4.0), "UK": ("UKNGDP", 4.0)}

# ONS: the two halves of the gilt stock. The liability of the whole UK economy is what
# central government has issued; the asset of the whole UK economy is what UK residents
# hold. The difference is what the rest of the world holds -- and because ONS builds the
# decomposition to sum to the stock, no separate overseas series is needed.
ONS_GILTS_ISSUED = ("NYXQ", "ukea")     # S.1 liability: the whole gilt stock
ONS_GILTS_UK_HELD = ("NYXP", "ukea")    # S.1 asset: the part UK residents hold
ONS_APF = ("MEX2", "pusf")              # BoE Asset Purchase Facility, nominal value

_TIMEOUT = 90


def _curl(url: str, retries: int = 4, accept: str = "") -> bytes:
    last = None
    hdr = ["-H", f"Accept: {accept}"] if accept else []
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", *hdr, url],
                capture_output=True, timeout=_TIMEOUT + 30)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
            return out.stdout
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            logger.warning("fetch %s attempt %d: %s", url, attempt + 1, exc)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _quarterly(s: pd.Series) -> pd.Series:
    """Quarter-end mean, indexed on the quarter-end date."""
    return s.resample("Q").mean().dropna()


# --------------------------------------------------------------------------- Japan

def boj_jgb_holdings() -> pd.Series:
    """Bank of Japan JGB holdings, billions of yen, quarterly.

    Summed from the issue-level ``mei`` release already cached by ``supply.fetch_jp``.
    Amounts there are in 100-million yen, so the conversion to billions is /10.
    """
    raw = pd.read_parquet(DATA / "jp_boj_raw.parquet")
    tot = raw.groupby("asof")["amount"].sum() / 10.0
    tot.index = pd.DatetimeIndex(tot.index)
    return _quarterly(tot.sort_index())


def _cofer(indicator: str, currency: str, transform: str) -> pd.Series:
    key = f"G001.{indicator}.{currency}.{transform}.Q"
    # The ?format= query parameter is honoured only on the wildcard path; with an
    # explicit key the server answers XML unless the content type is negotiated.
    blob = _curl(f"{COFER}/{key}",
                 accept="application/vnd.sdmx.data+csv;version=1.0.0")
    df = pd.read_csv(io.BytesIO(blob), low_memory=False)
    df = df[["TIME_PERIOD", "OBS_VALUE"]].dropna()
    idx = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q").to_timestamp(how="end")
    return pd.Series(df["OBS_VALUE"].to_numpy(float),
                     index=idx.normalize()).sort_index()


def cofer_yen_reserves() -> pd.Series:
    """Official yen FX reserves worldwide, billions of yen, quarterly.

    COFER's yen SHARE (of the reporters who disclose composition) applied to TOTAL
    reserves (all reporters). Converted at the period-average yen rate.
    """
    share = _cofer("AFXRA", "CI_JPY", "SHRO_PT") / 100.0
    total_usd = _cofer("TFXRA", "CI_T", "NV_USD")             # US dollars
    fx = _quarterly(fred_series("EXJPUS"))                     # yen per dollar
    idx = share.index.intersection(total_usd.index).intersection(fx.index)
    yen = share[idx] * total_usd[idx] * fx[idx] / 1e9          # -> billions of yen
    return yen.dropna()


# ------------------------------------------------------------------------- Germany

def _ecb_dates(cells: list) -> pd.DatetimeIndex:
    """Header dates. PSPP writes '31/03/2015', PEPP writes 'Mar-20'."""
    raw = pd.Series([c.strip() for c in cells])
    out = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    alt = pd.to_datetime(raw, format="%b-%y", errors="coerce")
    return pd.DatetimeIndex(out.fillna(alt))


def _ecb_breakdown(url: str, country: str) -> pd.Series:
    """Cumulative net purchases for one country, EUR millions, month-end.

    Both ECB files lead with a 'net purchases' block: a date header row, one row per
    country, terminated by 'Total'. Later blocks (weighted-average maturity, and for
    PSPP the issuer split) repeat the same country names, so parsing stops at 'Total'.
    A trailing 'Cumulative' column is present in the PEPP file and is dropped -- it is
    the sum of the row, not another month.
    """
    rows = list(csv_rows(_curl(url).decode("utf-8", "replace")))
    header = next(i for i, r in enumerate(rows)
                  if r and r[0].strip() == "" and len(r) > 3)
    dates = _ecb_dates(rows[header][1:])
    ok = ~dates.isna()
    for r in rows[header + 1:]:
        if not r:
            continue
        name = r[0].strip()
        if name.lower() == "total":
            break
        if name.lower().removeprefix("the ") != country.lower():
            continue
        vals = pd.to_numeric(pd.Series(r[1:1 + len(dates)]).replace("", None),
                             errors="coerce").reindex(range(len(dates)))
        s = pd.Series(vals.to_numpy(float)[ok], index=dates[ok])
        return s.fillna(0.0).cumsum().resample("M").last()
    raise KeyError(f"{country} not found in {url}")


def csv_rows(text: str):
    import csv as _csv
    return _csv.reader(io.StringIO(text))


def eurosystem_de_holdings() -> pd.Series:
    """Eurosystem holdings of German government securities, EUR millions, quarterly.

    PSPP from March 2015 plus PEPP from March 2020, each cumulated from the published
    monthly net purchases. Zero before PSPP began, which is the fact rather than an
    assumption.
    """
    pspp = _ecb_breakdown(ECB_PSPP, "Germany")
    pepp = _ecb_breakdown(ECB_PEPP, "Germany")
    idx = pspp.index.union(pepp.index)
    tot = pspp.reindex(idx).ffill().fillna(0.0) + pepp.reindex(idx).ffill().fillna(0.0)
    return _quarterly(tot)


# ------------------------------------------------------------------------------ UK

def uk_price_insensitive_gbp() -> tuple:
    """(BoE APF gilt holdings, overseas gilt holdings), GBP millions, quarterly.

    The APF sits inside the UK-resident total, so the two do not double-count.
    """
    apf = _quarterly(ons_series(*ONS_APF, freq="quarters"))
    issued = _quarterly(ons_series(*ONS_GILTS_ISSUED, freq="quarters"))
    uk_held = _quarterly(ons_series(*ONS_GILTS_UK_HELD, freq="quarters"))
    idx = issued.index.intersection(uk_held.index)
    overseas = (issued[idx] - uk_held[idx]).clip(lower=0.0)
    return apf, overseas


# --------------------------------------------------------------------------- shared

def _gdp(country: str) -> pd.Series:
    sid, scale = GDP[country]
    return _quarterly(fred_series(sid)) * scale


def price_insensitive(country: str, refresh: bool = False) -> pd.DataFrame:
    """Price-insensitive holdings as a percent of GDP, quarterly.

    Columns: ``central_bank``, ``official_foreign`` (absent for DE), ``total``.
    """
    cache = DATA / f"price_insensitive_{country.lower()}.parquet"
    if not refresh and cache.exists():
        return pd.read_parquet(cache)
    gdp = _gdp(country)
    if country == "JP":
        cols = {"central_bank": boj_jgb_holdings(),
                "official_foreign": cofer_yen_reserves()}
    elif country == "DE":
        cols = {"central_bank": eurosystem_de_holdings()}
    elif country == "UK":
        apf, overseas = uk_price_insensitive_gbp()
        cols = {"central_bank": apf, "official_foreign": overseas}
    else:
        raise KeyError(f"no price-insensitive definition for {country}; "
                       "the US pair lives in model.tp_drivers")
    out = pd.DataFrame({k: v.reindex(gdp.index).ffill(limit=1) / gdp * 100.0
                        for k, v in cols.items()})
    # The central bank starts at zero, not at NaN: before it bought, it held nothing.
    out["central_bank"] = out["central_bank"].where(
        out.index >= cols["central_bank"].index.min(), 0.0)
    out["total"] = out.sum(axis=1, min_count=1)
    out = out.dropna(how="all")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import sys
    refresh = "--refresh" in sys.argv
    for c in ("JP", "DE", "UK"):
        d = price_insensitive(c, refresh=refresh)
        d = d.dropna(subset=["total"])
        print(f"\n=== {c}  n={len(d)}  {d.index[0].date()} .. {d.index[-1].date()}")
        print(d.tail(4).round(1).to_string())
        print("peak total %.1f%% of GDP on %s"
              % (d["total"].max(), d["total"].idxmax().date()))


if __name__ == "__main__":
    _main()
