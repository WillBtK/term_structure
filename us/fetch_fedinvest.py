"""Fetch + cache U.S. Treasury **per-security** end-of-day prices (TreasuryDirect
FedInvest) — the raw material for a nominal-Treasury Hu-Pan-Wang "noise" gauge.

The Treasury publishes, for every marketable security outstanding on a given date, an
end-of-day price alongside the CUSIP, security type, coupon rate and maturity date.
That is everything needed to put each individual bond on a yield basis and measure how
far it sits from a smooth curve — the US nominal analogue of the TIPS gauge
(`validation/tips_liquidity.py`) and the Bund gauge (`validation/bund_liquidity.py`),
and closer to Hu-Pan-Wang's original construction, which is a *nominal* Treasury
measure.

Unlike the yield sources behind the other two gauges this one publishes PRICES, so the
price -> yield step (`model/bond_math.py`) is ours to do.

Access: the endpoint takes an arbitrary past date, so unlike the Bundesbank workbooks
history here is BACK-FILLABLE rather than merely retainable — the cache can be
extended over any window. treasurydirect.gov is egress-blocked from the dev sandbox,
so this is exercised on the GitHub runner: ``python -m us.fetch_fedinvest --probe``
dumps the endpoint that answers, the parsed columns, and the reachable history depth.
"""
from __future__ import annotations

import io
import logging

import pandas as pd

from us import config as cfg

logger = logging.getLogger(__name__)

# The same form is served from two hosts (treasurydirect.gov and its savingsbonds.gov
# alias) and from a legacy path; try each so a single redirect/rename cannot take the
# fetcher down. Confirmed live by --probe.
ENDPOINTS = [
    "https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate",
    "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate",
    "https://www.treasurydirect.gov/GA-FI/FedInvest/todaySecurityPriceDate",
]

# Columns as published: CUSIP | SECURITY TYPE | RATE | MATURITY DATE | CALL DATE |
# BUY | SELL | END OF DAY. Detected by keyword so a header rename does not break us.
_CUSIP_KEYS = ("cusip",)
_TYPE_KEYS = ("security type", "type")
_RATE_KEYS = ("rate", "coupon")
_MAT_KEYS = ("maturity",)
_EOD_KEYS = ("end of day", "eod")


def _find_col(cols, keys):
    for c in cols:
        cl = str(c).strip().lower()
        if any(k in cl for k in keys):
            return c
    return None


def _post(url: str, date: pd.Timestamp, timeout: int = 60) -> str:
    """POST the date form and return the HTML body."""
    import requests
    data = {
        "priceDate.month": date.month,
        "priceDate.day": date.day,
        "priceDate.year": date.year,
        "submit": "Show Prices",
    }
    r = requests.post(url, data=data, timeout=timeout,
                      headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.text


def parse_price_table(html: str, date: pd.Timestamp) -> pd.DataFrame:
    """Parse the FedInvest price table to tidy ``(date, cusip, kind, coupon, maturity,
    price)``.

    ``coupon`` is the annual coupon RATE IN PERCENT as published; ``price`` is the
    end-of-day CLEAN price per 100 face. Returns an empty frame when the page carries
    no price table (weekend/holiday, or a date before the archive starts).
    """
    cols = ["date", "cusip", "kind", "coupon", "maturity", "price"]
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame(columns=cols)
    for t in tables:
        t.columns = [str(c).strip() for c in t.columns]
        c_cusip = _find_col(t.columns, _CUSIP_KEYS)
        c_mat = _find_col(t.columns, _MAT_KEYS)
        c_eod = _find_col(t.columns, _EOD_KEYS)
        if not (c_cusip and c_mat and c_eod):
            continue
        c_type = _find_col(t.columns, _TYPE_KEYS)
        c_rate = _find_col(t.columns, _RATE_KEYS)
        out = pd.DataFrame({
            "date": date,
            "cusip": t[c_cusip].astype(str).str.strip(),
            "kind": (t[c_type].astype(str).str.strip().str.upper()
                     if c_type else ""),
            "coupon": (pd.to_numeric(t[c_rate], errors="coerce") if c_rate else float("nan")),
            "maturity": pd.to_datetime(t[c_mat], errors="coerce"),
            "price": pd.to_numeric(t[c_eod], errors="coerce"),
        })
        return out.dropna(subset=["maturity", "price"]).reset_index(drop=True)
    return pd.DataFrame(columns=cols)


def fetch_day(date: pd.Timestamp) -> pd.DataFrame:
    """One trading day's cross-section, trying each known endpoint in turn."""
    date = pd.Timestamp(date)
    last = None
    for url in ENDPOINTS:
        try:
            df = parse_price_table(_post(url, date), date)
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            continue
        if len(df):
            return df
    if last is not None:
        logger.debug("no prices for %s (last error: %s)", date.date(), last)
    return pd.DataFrame(columns=["date", "cusip", "kind", "coupon", "maturity", "price"])


# Only NOMINAL coupon securities belong in the nominal noise fit: bills have no coupon
# and quote on a discount basis, TIPS quote off an inflation-adjusted principal, and
# FRNs have no fixed cashflow schedule. Each is excluded by security type.
_NOMINAL_KINDS = ("NOTE", "BOND")


def nominal_coupon_bonds(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only fixed-coupon nominal notes/bonds (drop bills, TIPS, FRNs, STRIPS)."""
    if "kind" not in panel.columns:
        return panel
    k = panel["kind"].astype(str).str.upper()
    keep = k.str.contains("|".join(_NOMINAL_KINDS), regex=True)
    keep &= ~k.str.contains("TIPS|INFLATION|FLOATING|FRN|STRIP", regex=True)
    return panel.loc[keep].copy()


def _probe(n_days: int = 3) -> None:
    """Diagnostic for the runner: which endpoint answers, what the table looks like,
    and how far back the archive reaches."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)
    print("=== endpoint reachability ===")
    probe_date = today - pd.Timedelta(days=5)
    for url in ENDPOINTS:
        try:
            html = _post(url, probe_date)
            df = parse_price_table(html, probe_date)
            print(f"  OK   {url}  html={len(html)}B rows={len(df)}")
            if len(df):
                print(df.head(5).to_string(index=False))
                print("  kinds:", sorted(df['kind'].unique())[:12])
        except Exception as exc:                                    # noqa: BLE001
            print(f"  FAIL {url}  {type(exc).__name__}: {str(exc)[:120]}")

    print("\n=== history depth (first date that returns a price table) ===")
    for year in (1990, 1995, 2000, 2005, 2010, 2015, 2020):
        d = pd.Timestamp(f"{year}-06-15")
        df = fetch_day(d)
        nom = nominal_coupon_bonds(df)
        print(f"  {d.date()}  rows={len(df):4d}  nominal notes/bonds={len(nom):4d}")


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="TreasuryDirect FedInvest per-security prices")
    ap.add_argument("--probe", action="store_true", help="dump endpoint/layout/depth")
    args = ap.parse_args()
    if args.probe:
        _probe()
        return
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = fetch_day(pd.Timestamp.utcnow().normalize().tz_localize(None) - pd.Timedelta(days=3))
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    _main()
