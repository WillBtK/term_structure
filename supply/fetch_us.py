"""US ladders: marketable Treasury debt outstanding, and the Federal Reserve's holdings.

Two security-level sources, both giving par amount, coupon and maturity date, which is
everything ``supply.tenyear`` needs:

* **MSPD table 3** (fiscaldata.treasury.gov) -- every marketable Treasury security
  outstanding, monthly from 2001-01. Amounts in $mn.
* **SOMA holdings** (markets.newyorkfed.org) -- the Fed's portfolio by CUSIP, weekly.
  Amounts in dollars.

Only NOMINAL coupon debt and bills are kept. TIPS carry real rather than nominal
duration and the premium we model is the nominal one; floating-rate notes reset
quarterly and carry almost no duration; and the ``Total Marketable`` summary row must be
dropped or it double-counts the entire ladder.
"""
from __future__ import annotations

import io
import json
import logging
import subprocess
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MSPD_RAW = DATA / "us_mspd_raw.parquet"
SOMA_RAW = DATA / "us_soma_raw.parquet"

MSPD_API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
            "/v1/debt/mspd/mspd_table_3_market")
SOMA_API = "https://markets.newyorkfed.org/api/soma"

# Nominal, duration-bearing, non-aggregate. The two sources spell their categories
# differently -- MSPD splits "Notes" from "Bonds", SOMA lumps them as "NotesBonds" --
# and getting that wrong is silent: an unmatched whitelist drops every coupon security
# and leaves a bills-only ladder whose duration share is near zero but whose par share
# looks fine. Both raw ladders are cached UNFILTERED for that reason, so revising these
# sets costs a reload rather than another download.
KEEP_CLASSES = {"Notes", "Bonds", "Bills Maturity Value"}
SOMA_KEEP = {"Bills", "NotesBonds"}
_TIMEOUT = 90


def _get(url: str, retries: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", url],
                capture_output=True, timeout=_TIMEOUT + 20)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:150]!r}")
            return json.load(io.BytesIO(out.stdout))
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch failed for {url[:110]}: {last}")


def fetch_mspd(start_year: int = 2003, end_year: int = 2026,
               refresh: bool = False) -> pd.DataFrame:
    """Marketable Treasury securities outstanding, one row per security-month."""
    if not refresh and MSPD_RAW.exists():
        return _shape_mspd(pd.read_parquet(MSPD_RAW))
    fields = ("record_date,security_class1_desc,security_class2_desc,"
              "interest_rate_pct,maturity_date,outstanding_amt")
    frames = []
    for yr in range(start_year, end_year + 1):
        url = (f"{MSPD_API}?filter=record_date:gte:{yr}-01-01,"
               f"record_date:lte:{yr}-12-31&fields={fields}"
               f"&page%5Bsize%5D=10000")
        rows = _get(url).get("data") or []
        if not rows:
            logger.warning("MSPD %d: no rows", yr)
            continue
        frames.append(pd.DataFrame(rows))
        logger.info("MSPD %d: %d security-months", yr, len(rows))
    df = pd.concat(frames, ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MSPD_RAW)
    return _shape_mspd(df)


def _shape_mspd(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["security_class1_desc"].isin(KEEP_CLASSES)].copy()
    df["record_date"] = pd.to_datetime(df["record_date"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"], errors="coerce")
    # Bills are quoted at maturity value with no coupon; MSPD leaves the rate blank.
    df["coupon_pct"] = pd.to_numeric(df["interest_rate_pct"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df["outstanding_amt"], errors="coerce")   # $mn
    df = df.dropna(subset=["maturity_date", "amount"])
    df = df[df["amount"] > 0][["record_date", "security_class2_desc",
                               "maturity_date", "coupon_pct", "amount"]]
    return df.rename(columns={"security_class2_desc": "cusip"})


def soma_asof_dates() -> list:
    return list(_get(f"{SOMA_API}/asofdates/list.json")["soma"]["asOfDates"])


def fetch_soma(refresh: bool = False, monthly: bool = True) -> pd.DataFrame:
    """Federal Reserve SOMA Treasury holdings by CUSIP.

    SOMA publishes weekly; only the last as-of date in each month is kept, to match the
    monthly MSPD snapshots without fetching four times the data.
    """
    if not refresh and SOMA_RAW.exists():
        return _shape_soma(pd.read_parquet(SOMA_RAW))
    dates = sorted(soma_asof_dates())
    if monthly:
        by_month = {}
        for d in dates:
            by_month[d[:7]] = d          # sorted, so the last wins
        dates = sorted(by_month.values())
    frames = []
    for i, d in enumerate(dates):
        try:
            rows = _get(f"{SOMA_API}/tsy/get/all/asof/{d}.json")["soma"]["holdings"]
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("SOMA %s failed: %s", d, exc)
            continue
        if not rows:
            continue
        f = pd.DataFrame(rows)
        f["asof"] = pd.Timestamp(d)
        frames.append(f)
        if i % 24 == 0:
            logger.info("SOMA %s (%d/%d)", d, i + 1, len(dates))
    df = pd.concat(frames, ignore_index=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SOMA_RAW)
    return _shape_soma(df)


def _shape_soma(df: pd.DataFrame) -> pd.DataFrame:
    unknown = set(df["securityType"].unique()) - SOMA_KEEP - {"TIPS", "FRN", "FRNs"}
    if unknown:
        logger.warning("SOMA: unrecognised securityType(s) %s -- excluded", unknown)
    df = df[df["securityType"].isin(SOMA_KEEP)].copy()
    df["maturity_date"] = pd.to_datetime(df["maturityDate"], errors="coerce")
    df["coupon_pct"] = pd.to_numeric(df["coupon"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df["parValue"], errors="coerce") / 1e6      # -> $mn
    df = df.dropna(subset=["maturity_date", "amount"])
    return df[df["amount"] > 0][["asof", "cusip", "maturity_date",
                                "coupon_pct", "amount"]]


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    m = fetch_mspd(refresh=True)
    print(f"MSPD: {len(m):,} security-months, {m['record_date'].min().date()}"
          f"..{m['record_date'].max().date()}, {m['record_date'].nunique()} snapshots")
    s = fetch_soma(refresh=True)
    # NB: s["asof"], never s.asof -- ``asof`` is a DataFrame METHOD, so attribute
    # access silently returns the bound method instead of the column.
    print(f"SOMA: {len(s):,} security-months, {s['asof'].min().date()}"
          f"..{s['asof'].max().date()}, {s['asof'].nunique()} snapshots")


if __name__ == "__main__":
    _main()
