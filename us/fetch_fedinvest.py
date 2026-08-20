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
import time

import pandas as pd

from us import config as cfg

logger = logging.getLogger(__name__)

# The same form is served from two hosts (treasurydirect.gov and its savingsbonds.gov
# alias) and from a legacy path; try each so a single redirect/rename cannot take the
# fetcher down. Confirmed live by --probe.
ENDPOINTS = [
    "https://www.treasurydirect.gov/GA-FI/FedInvest/selectSecurityPriceDate",
    "https://savingsbonds.gov/GA-FI/FedInvest/selectSecurityPriceDate",
]

# Columns as published: CUSIP | SECURITY TYPE | RATE | MATURITY DATE | CALL DATE |
# BUY | SELL | END OF DAY. Detected by keyword so a header rename does not break us.
_CUSIP_KEYS = ("cusip",)
_TYPE_KEYS = ("security type", "type")
_RATE_KEYS = ("rate", "coupon")
_MAT_KEYS = ("maturity",)
_CALL_KEYS = ("call",)
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
    cols = ["date", "cusip", "kind", "coupon", "maturity", "call_date", "price"]
    try:
        # Force the lxml flavor explicitly. Left to auto-detect, pandas falls back to
        # BeautifulSoup whenever lxml raises on a given page -- silently, since that
        # fallback itself then throws an uncaught ImportError (beautifulsoup4 isn't a
        # dependency here). That propagated out of this function on any day lxml
        # stumbled on, fetch_day's try/except swallowed it and moved to the next
        # mirror, and if that mirror hit the same page shape the day came back with
        # ZERO rows rather than an empty-but-valid frame -- which a build_noise merge
        # then silently reads as "nothing new" and leaves the STALE row in the store.
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except (ValueError, ImportError):
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
        # CALL DATE is blank for the (overwhelming majority) non-callable rows, which
        # to_datetime coerces to NaT — exactly the "not callable" sentinel we want.
        # Treasury's last callable long bond had a call window into Nov 2009, so this
        # only bites a handful of very old securities, but silently pricing one to its
        # final maturity while the market is pricing the call blows up a single bond's
        # implied yield and can dominate the whole day's fitting residual.
        c_call = _find_col(t.columns, _CALL_KEYS)
        out = pd.DataFrame({
            "date": date,
            "cusip": t[c_cusip].astype(str).str.strip(),
            "kind": (t[c_type].astype(str).str.strip().str.upper()
                     if c_type else ""),
            # RATE is published as a STRING with a percent sign ("4.250%"). Feeding that
            # straight to to_numeric coerces every coupon to NaN — silently, since the
            # column is still there and the row count is unchanged.
            "coupon": (pd.to_numeric(t[c_rate].astype(str).str.replace("%", "", regex=False)
                                     .str.strip(), errors="coerce")
                       if c_rate else float("nan")),
            "maturity": pd.to_datetime(t[c_mat], errors="coerce"),
            "call_date": (pd.to_datetime(t[c_call], errors="coerce")
                          if c_call else pd.NaT),
            "price": pd.to_numeric(t[c_eod], errors="coerce"),
        })
        return out.dropna(subset=["maturity", "price"]).reset_index(drop=True)
    return pd.DataFrame(columns=cols)


def fetch_day(date: pd.Timestamp, attempts: int = 3, backoff: float = 1.5) -> pd.DataFrame:
    """One trading day's cross-section, trying each known endpoint in turn.

    Both endpoints intermittently answer a ~7.8 kB STUB page carrying no price table
    (the real page is ~60-70 kB), seemingly under load during a long sequential crawl.
    An empty result is therefore ambiguous: it means either "genuine non-trading day"
    or "the server hiccuped". Treating the two alike is what let a corrective re-run
    silently skip days and leave stale values in the derived store, so a day that comes
    back empty is RETRIED with backoff before being accepted as genuinely absent.
    """
    date = pd.Timestamp(date)
    last = None
    for attempt in range(attempts):
        for url in ENDPOINTS:
            try:
                df = parse_price_table(_post(url, date), date)
            except Exception as exc:                                # noqa: BLE001
                last = exc
                continue
            if len(df):
                return df
        if attempt + 1 < attempts:
            time.sleep(backoff * (2 ** attempt))
    if last is not None:
        logger.debug("no prices for %s (last error: %s)", date.date(), last)
    return pd.DataFrame(columns=["date", "cusip", "kind", "coupon", "maturity", "call_date", "price"])


# Security types published (confirmed live): MARKET BASED BILL / BOND / FRN / NOTE and
# TIPS. Only fixed-coupon NOMINAL notes and bonds belong in the nominal noise fit —
# bills carry no coupon and quote on a discount basis, TIPS price off an
# inflation-adjusted principal, and FRNs have no fixed cashflow schedule.
_NOMINAL_KINDS = ("NOTE", "BOND")


def nominal_coupon_bonds(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep only fixed-coupon nominal notes/bonds (drop bills, TIPS, FRNs, STRIPS)."""
    if "kind" not in panel.columns:
        return panel
    k = panel["kind"].astype(str).str.upper()
    keep = k.str.contains("|".join(_NOMINAL_KINDS), regex=True)
    keep &= ~k.str.contains("TIPS|INFLATION|FLOATING|FRN|STRIP", regex=True)
    # Treasury's last callable long bonds (issued through 1984, called through ~2009)
    # still show up in the archive's early years. Pricing one to its final maturity
    # while the market prices it to the call is a mispricing this module has no way to
    # correct (that needs option-adjusted pricing, out of scope here) — so it gets
    # excluded rather than silently fed to a bullet-bond fit, same as bills/TIPS/FRNs.
    if "call_date" in panel.columns:
        keep &= panel["call_date"].isna()
    out = panel.loc[keep].copy()
    # A coupon that failed to parse cannot be priced; drop rather than carry a None into
    # the bond math, where it surfaces as an unrelated-looking TypeError deep in the stack.
    if "coupon" in out.columns:
        before = len(out)
        out = out[pd.to_numeric(out["coupon"], errors="coerce").notna()]
        if len(out) < before:
            logger.warning("dropped %d/%d nominal rows with an unparseable coupon",
                           before - len(out), before)
    return out


def business_days(start, end) -> pd.DatetimeIndex:
    """Weekdays in [start, end]. Market holidays simply return no table and are skipped."""
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))


def load_prices(start=None, end=None, refresh: bool = False,
                cache_only: bool = False) -> pd.DataFrame:
    """Per-security price panel over a date range, cached to parquet.

    Unlike the Bundesbank workbooks, this endpoint accepts an ARBITRARY past date, so
    history here is back-fillable rather than merely retainable: a missing window can
    always be re-requested. The cache therefore exists to avoid re-fetching, not to
    preserve anything — only days absent from it are requested.
    """
    from validation import config as vcfg
    path = vcfg.UST_PRICES_PARQUET
    cached = pd.read_parquet(path) if path.exists() else \
        pd.DataFrame(columns=["date", "cusip", "kind", "coupon", "maturity", "call_date", "price"])
    if cache_only:
        return cached
    start = start or (cached["date"].max() + pd.Timedelta(days=1)
                      if len(cached) else vcfg.UST_PRICES_START)
    end = end or pd.Timestamp.utcnow().normalize().tz_localize(None)
    have = (set(pd.to_datetime(cached["date"]).unique())
            if len(cached) and not refresh else set())
    want = [d for d in business_days(start, end) if d not in have]
    if not want:
        return cached
    logger.info("FedInvest: fetching %d trading days %s..%s",
                len(want), want[0].date(), want[-1].date())
    got, empty = [], 0
    for i, d in enumerate(want, 1):
        df = fetch_day(d)
        if len(df):
            got.append(df)
        else:
            empty += 1
        if i % 250 == 0:
            logger.info("  %d/%d days (%d without a table)", i, len(want), empty)
    if not got:
        logger.warning("FedInvest returned no data for any of %d requested days", len(want))
        return cached
    panel = (pd.concat([cached, *got], ignore_index=True)
             .drop_duplicates(["date", "cusip"], keep="last")
             .sort_values(["date", "maturity"]).reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    logger.info("FedInvest panel: %d rows, %s..%s, %d securities",
                len(panel), panel["date"].min().date(), panel["date"].max().date(),
                panel["cusip"].nunique())
    return panel


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

    print("\n=== archive start (bisect on a table being returned) ===")
    lo, hi = pd.Timestamp("1995-01-04"), pd.Timestamp("2012-01-04")
    while (hi - lo).days > 20:
        mid = lo + (hi - lo) / 2
        window = business_days(mid, mid + pd.Timedelta(days=10))
        if any(len(fetch_day(d)) for d in window):
            hi = mid
        else:
            lo = mid
    print(f"  first data lands between {lo.date()} and {hi.date()}")

    print("\n=== end-to-end noise on a recent day ===")
    from us.fetch_gsw import load_gsw_params
    from validation.ust_liquidity import noise_from_panel
    day = nominal_coupon_bonds(fetch_day(probe_date))
    gsw = load_gsw_params(refresh=True)
    out = noise_from_panel(day, gsw_params=gsw)
    print(out.assign(bp=lambda d: (d["noise"] * 1e4).round(2),
                     bp_gsw=lambda d: (d["noise_gsw"] * 1e4).round(2))
          [["date", "bp", "bp_gsw", "n_bonds"]].to_string(index=False))


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
