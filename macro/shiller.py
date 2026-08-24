"""Shiller's long US market history, and the stock-bond correlation built from it.

``model/stock_bond.py`` explains why the stock-bond correlation is in the argument at
all: a term premium is modelled, so it cannot be checked directly, but the correlation
CAN be, which splits an otherwise untestable chain into two separately falsifiable
links. That check is only as good as its sample, and ours was too short. The six-market
panel starts in 1981, so it contains exactly ONE inflation regime change -- the
disinflation -- and a relationship fitted on one turning point is a story, not evidence.

Robert Shiller's ``ie_data.xls`` (the *Irrational Exuberance* dataset) fixes that. It
carries monthly S&P composite price, dividend, CPI and the long government bond yield
from **1871**, which is long enough to contain the 1910s inflation, the 1940s one, the
Great Inflation and the disinflation -- four regime changes rather than one.

**Constructing the two returns.** Equity is direct: price change plus one month of the
annual dividend. Bonds are not -- Shiller publishes a YIELD, not a return -- so the
constant-maturity ten-year total return is approximated

    r_bond = y/12  -  D_mod * dy            D_mod = (1 - (1+y)^-10) / y

carry less duration times the yield change, with duration recomputed each month from
that month's yield rather than held fixed. This is a first-order approximation and is
stated as one: it omits convexity, assumes a par bond, and ignores the roll down a
non-flat curve. Convexity was tested and moves the rolling correlation by at most
0.006 (in Feb 1980, the most volatile month in the sample; mean 0.0008), which is far
inside the noise of a 120-month window, so it is left out rather than carried as false
precision. What the approximation CANNOT do is
manufacture the sign changes the exhibit rests on: those come from the covariance of
equity returns with yield CHANGES, which is exactly what the data measures.

**Provenance.** ``shillerdata.com`` serves the file from a hashed CDN path that changes
when Shiller republishes, so the page is scraped for the current link rather than the
link being hard-coded; the last-known URL is the fallback, and the committed parquet is
the fallback after that. ``econ.yale.edu`` -- the address the file is usually cited
under -- returns 403 to automated requests and is not used.
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
# CSV, not parquet: `*.parquet` is gitignored, so a parquet cache would never
# actually ship -- and a fallback that is not committed is not a fallback.
CACHE = HERE / "data" / "shiller_monthly.csv"

PAGE = "https://shillerdata.com/"
# last known good direct link; only used if the page scrape finds nothing
FALLBACK = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53"
            "/downloads/e27e58c1-8ae0-488c-a976-a298708c7175/ie_data.xls")
_TIMEOUT = 120

WINDOW = 120        # months in the rolling correlation window; 10y, as in the AQR work
TENOR = 10.0        # the constant-maturity bond being approximated


def _curl(url: str, retries: int = 3) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", url],
                capture_output=True, timeout=_TIMEOUT + 30)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
            return out.stdout
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            logger.warning("shiller fetch %s attempt %d: %s", url, attempt + 1, exc)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _workbook_url() -> str:
    """Current ie_data.xls link, scraped; the CDN path carries a hash that rotates."""
    try:
        html = _curl(PAGE).decode("utf-8", "replace")
        hits = re.findall(r'href="(//[^"]*?ie_data\.xls[^"]*)"', html)
        if hits:
            return "https:" + hits[0]
        logger.warning("no ie_data.xls link on %s; using the last known URL", PAGE)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("shillerdata.com unreachable (%s); using the last known URL", exc)
    return FALLBACK


def _parse(blob: bytes) -> pd.DataFrame:
    """Month-start frame of price, dividend, CPI and the long bond yield (decimal).

    The sheet carries several banner rows above the header and a trailing block of
    notes below the data, so the header is FOUND (the row whose first cell is "Date")
    rather than counted, and rows are kept only while the date column parses. Dates are
    ``YYYY.MM`` floats, where 1871.1 means October -- so the month is recovered by
    rounding ``date * 100``, not by taking the fractional part.
    """
    raw = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None)
    hdr = raw.index[raw[0].astype(str).str.strip().eq("Date")]
    if not len(hdr):
        raise ValueError("no 'Date' header row in the Shiller workbook")
    body = raw.iloc[hdr[0] + 1:]
    stamp = pd.to_numeric(body[0], errors="coerce")
    body, stamp = body[stamp.notna()], stamp.dropna()
    idx = pd.to_datetime({"year": stamp.astype(int),
                          "month": (stamp * 100).round().astype(int) % 100,
                          "day": 1})
    num = lambda c: pd.to_numeric(body[c], errors="coerce").values   # noqa: E731
    df = pd.DataFrame({"price": num(1), "dividend": num(2), "cpi": num(4),
                       "y10": num(6) / 100.0}, index=idx)
    df = df.dropna(subset=["price", "y10"])
    # the last month or two carry a price and yield but no dividend yet
    df["dividend"] = df["dividend"].ffill()
    return df.sort_index()


def refresh() -> pd.DataFrame:
    """Fetch, parse and cache. Raises if the source cannot be read at all."""
    df = _parse(_curl(_workbook_url()))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index_label="date")
    logger.info("shiller: %d months, %s -> %s", len(df),
                df.index[0].date(), df.index[-1].date())
    return df


def load(refresh_data: bool = False) -> pd.DataFrame:
    """Cached monthly frame; the committed parquet is the fallback if the CDN is down."""
    if not refresh_data and CACHE.is_file():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    try:
        return refresh()
    except Exception as exc:                                        # noqa: BLE001
        if CACHE.is_file():
            logger.warning("shiller refresh failed (%s); using the cache", exc)
            return pd.read_csv(CACHE, index_col=0, parse_dates=True)
        raise


def returns(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly equity and constant-maturity 10y bond total returns, decimal."""
    d = load() if df is None else df
    eq = (d["price"] + d["dividend"] / 12.0) / d["price"].shift(1) - 1.0
    y = d["y10"]
    dur = (1.0 - (1.0 + y) ** (-TENOR)) / y          # par-bond modified duration
    bd = y / 12.0 - dur * y.diff()
    return pd.DataFrame({"equity": eq, "bond": bd}).dropna()


def stock_bond_correlation(window: int = WINDOW) -> pd.Series:
    """Rolling correlation of US equity and 10y bond monthly returns, 1881-present."""
    r = returns()
    return (r["equity"].rolling(window).corr(r["bond"])
            .dropna().rename("stock_bond_corr"))


def regime_means(corr: pd.Series | None = None) -> pd.DataFrame:
    """The correlation averaged over the inflation regimes the exhibit is about.

    Eras are named for what happened to INFLATION, not for the correlation, so the
    table cannot be accused of being drawn around its own answer.

    The window is CENTRED here, unlike the plotted series. A trailing ten-year window
    dates a regime by when it ended: it puts the Great Inflation's correlation in the
    1980s and 90s, which reads as the disinflation having the higher correlation
    (+0.32 against +0.19) when the opposite is true. Centred, the ordering is the one
    the argument claims -- Great Inflation +0.28, disinflation +0.23, inflation
    targeting -0.26. The chart keeps the trailing convention because that is what an
    investor could actually have observed at the time, and because it runs to today.
    """
    c = stock_bond_correlation() if corr is None else corr
    c = c.copy()
    c.index = c.index - pd.DateOffset(months=WINDOW // 2)
    eras = [("1881-1914", "Classical gold standard", "1881", "1914"),
            ("1915-1929", "First World War inflation", "1915", "1929"),
            ("1930-1945", "Depression and war", "1930", "1945"),
            ("1946-1965", "Bretton Woods", "1946", "1965"),
            ("1966-1982", "The Great Inflation", "1966", "1982"),
            ("1983-1999", "Disinflation", "1983", "1999"),
            ("2000-2020", "Inflation targeting", "2000", "2020"),
            ("2021-", "Inflation returns", "2021", None)]
    rows = []
    for span, name, a, b in eras:
        seg = c.loc[a:b] if b else c.loc[a:]
        if len(seg):
            rows.append({"era": span, "regime": name,
                         "corr": round(float(seg.mean()), 2), "n": int(len(seg))})
    return pd.DataFrame(rows)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    df = refresh()
    c = stock_bond_correlation()
    print(f"\nShiller monthly: {df.index[0]:%Y-%m} -> {df.index[-1]:%Y-%m} "
          f"({len(df):,} months)")
    print(f"stock-bond correlation ({WINDOW}m window): "
          f"{c.index[0]:%Y-%m} -> {c.index[-1]:%Y-%m}, latest {c.iloc[-1]:+.2f}\n")
    print(regime_means(c).to_string(index=False))


if __name__ == "__main__":
    _main()
