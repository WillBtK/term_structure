"""Japan ladders: JGBs outstanding, and the Bank of Japan's holdings.

Japan needs a join that the US did not. The BoJ publishes its holdings issue by issue
(``mei`` files, three times a month, in this parseable form back to 2014-05) but identifies each one only by
ORIGINAL TENOR and ISSUE NUMBER -- "10-Year JGB #348" -- with no maturity date and no
coupon. The Ministry of Finance auction-results workbook supplies both, one sheet per
tenor, keyed on the same issue number, so the two join cleanly.

That workbook also gives the DENOMINATOR. Each auction row carries the amount accepted,
so summing across an issue's auctions (a JGB is reopened many times) recovers the amount
outstanding per issue without needing a separate outstanding-by-issue source.

Amounts are in 100 million yen throughout, on both sides, so the share is unit-free.
"""
from __future__ import annotations

import io
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
MOF_REF_PARQUET = DATA / "jp_mof_issues.parquet"
BOJ_RAW_PARQUET = DATA / "jp_boj_raw.parquet"

MOF_AUCTIONS = "https://www.mof.go.jp/jgbs/reference/appendix/jgb_historical_data.xls"
BOJ_BASE = "https://www.boj.or.jp/en/statistics/boj/other/mei/release"

# MoF sheet -> the label the BoJ file uses. Nominal, duration-bearing tenors only:
# the inflation-indexed sheet carries real rather than nominal duration, and the
# 15-year floating-rate note resets and carries almost none.
SHEETS = {
    "40年債": "40-Year JGB", "30年債": "30-Year JGB", "20年債": "20-Year JGB",
    "10年債": "10-Year JGB", "6年債": "6-Year JGB", "5年債": "5-Year JGB",
    "4年債": "4-Year JGB", "2年債": "2-Year JGB",
    "GX10年債": "10-Year Japan Climate Transition Bond",
    "GX5年債": "5-Year Japan Climate Transition Bond",
}
_TIMEOUT = 120


def _curl(url: str, retries: int = 4) -> bytes:
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
            time.sleep(2 ** attempt)
    raise RuntimeError(f"fetch failed for {url}: {last}")


def fetch_mof_issues(refresh: bool = False) -> pd.DataFrame:
    """One row per JGB issue: tenor, issue number, maturity, coupon, amount issued."""
    if not refresh and MOF_REF_PARQUET.exists():
        return pd.read_parquet(MOF_REF_PARQUET)
    book = pd.ExcelFile(io.BytesIO(_curl(MOF_AUCTIONS)))
    rows = []
    for sheet, label in SHEETS.items():
        if sheet not in book.sheet_names:
            logger.warning("MoF workbook has no sheet %s", sheet)
            continue
        d = book.parse(sheet, header=None)
        # The header block is a bilingual two-row affair; data starts at the first row
        # whose first column is a bare integer issue number.
        num = pd.to_numeric(d[0], errors="coerce")
        body = d[num.notna()].copy()
        body["issue"] = num[num.notna()].astype(int)
        body["maturity_date"] = pd.to_datetime(body[3], errors="coerce")
        body["coupon_pct"] = pd.to_numeric(body[4], errors="coerce")
        body["allotted"] = pd.to_numeric(body[7], errors="coerce")   # 落札・割当額
        body = body.dropna(subset=["maturity_date"])
        g = body.groupby("issue").agg(
            maturity_date=("maturity_date", "first"),
            coupon_pct=("coupon_pct", "first"),
            amount=("allotted", "sum"),
            n_auctions=("allotted", "size"))
        g["bond_type"] = label
        rows.append(g.reset_index())
        logger.info("  MoF %-8s %3d issues, %s..%s", sheet, len(g),
                    g["maturity_date"].min().date(), g["maturity_date"].max().date())
    df = pd.concat(rows, ignore_index=True)
    df["coupon_pct"] = df["coupon_pct"].fillna(0.0)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MOF_REF_PARQUET)
    return df


# Labels are matched case-insensitively: the file says "2-Year JGB" in the current
# workbooks and "2-year JGB" in the 2013-2016 ones.
_TYPE_MAP = {k.lower(): v for k, v in
             {**SHEETS, **{en: en for en in SHEETS.values()}}.items()}


def _year_files(year: int) -> list:
    """The ``mei`` files published in one year, from that year's index page."""
    html = _curl(f"{BOJ_BASE}/{year}/index.htm").decode("utf-8", "ignore")
    # Six-digit names only. The 2002-2013 archive uses four-digit monthly names in
    # HTML and PDF, which carry the same data in a form we do not parse.
    return sorted(set(re.findall(r"mei\d{6}\.(?:xlsx|xls|csv|zip)", html)))


def _spreadsheet(blob: bytes, name: str) -> bytes:
    """Unwrap the 2013-2016 snapshots, which ship the workbook inside a zip."""
    if not name.endswith(".zip"):
        return blob
    import zipfile
    z = zipfile.ZipFile(io.BytesIO(blob))
    inner = [i.filename for i in z.infolist()
             if i.filename.lower().endswith((".xls", ".xlsx", ".csv"))]
    if not inner:
        raise ValueError(f"{name}: zip contains no workbook ({z.namelist()})")
    return z.read(inner[0])


def _parse_mei(blob: bytes, name: str) -> pd.DataFrame:
    """One BoJ holdings snapshot -> (bond_type, issue, amount).

    Columns are LOCATED, not assumed. The layout has to hold across three archive
    formats: the current bilingual workbook heads the amount column 保有残高 and names
    the bond type once per block, while the 2013-2016 English workbooks head it
    "Amount Outstanding" and repeat the type on every row. Anchoring on the amount
    header and stepping left handles both.
    """
    d = pd.read_excel(io.BytesIO(_spreadsheet(blob, name)), header=None)
    amt_col = None
    for c in d.columns:
        text = d[c].astype(str)
        if text.str.contains("保有残高").any() or \
           text.str.contains("Amount Outstanding", case=False).any():
            amt_col = c
            break
    if amt_col is None:
        raise ValueError(f"{name}: could not locate the amount column")
    left = [c for c in d.columns if c < amt_col]
    if not left:
        raise ValueError(f"{name}: amount column has nothing to its left")
    issue_col = max(left)                       # issue number sits directly left
    # Forward-fill the block HEADING, then map -- not the other way round. The current
    # workbook names each bond type once and leaves the rows beneath it blank, so some
    # ffill is required; filling the MAPPED label instead silently re-labels every block
    # we do not recognise as the last one we did. "Floating-rate JGB" and the
    # inflation-indexed block follow 40-year in the sheet and were being stamped
    # "40-Year JGB", which both inflated the holdings total the ownership driver reads
    # and collided with genuine 40-year issue numbers. Filling the text first leaves an
    # unrecognised heading unmapped, so its rows drop out as intended.
    labels = {c: d[c].astype(str).str.strip().replace({"nan": None}).ffill()
                     .str.lower().map(_TYPE_MAP)
              for c in left if c != issue_col}
    if not labels:
        raise ValueError(f"{name}: no bond-type column")
    type_col = max(labels, key=lambda c: labels[c].notna().sum())
    if labels[type_col].notna().sum() == 0:
        raise ValueError(f"{name}: no recognised bond-type labels")
    out = pd.DataFrame({
        "bond_type": labels[type_col],      # already heading-filled; see above
        "issue": pd.to_numeric(d[issue_col], errors="coerce"),
        "amount": pd.to_numeric(d[amt_col], errors="coerce")}).dropna()
    out["issue"] = out["issue"].astype(int)
    return out[out["amount"] > 0]


def fetch_boj(start: int = 2001, end: int | None = None,
              refresh: bool = False) -> pd.DataFrame:
    """BoJ JGB holdings by issue, one snapshot per month (the last of each month).

    INCREMENTAL. A cold build walks every year and takes a few minutes; once the store
    exists, ``refresh=True`` fetches only the months missing from it, which on a daily
    schedule is nothing at all on most days and one file after each month end. That is
    what makes this cheap enough to sit in the daily job, and the daily job is what
    keeps it moving.

    The CURRENT month is always re-fetched, never treated as done. The BoJ posts three
    files a month and this takes the last one available, so a snapshot captured
    mid-month is not the month-end snapshot; accepting it as complete would freeze that
    month at whatever partial figure happened to be up when it was first seen.

    Effective history starts 2014-05: earlier years are indexed under four-digit names
    in HTML and PDF only (see ``_year_files``), so the six-digit workbook series this
    parses does not reach them.
    """
    end = end or pd.Timestamp.today().year
    have = pd.read_parquet(BOJ_RAW_PARQUET) if BOJ_RAW_PARQUET.exists() else None
    if have is not None and not refresh:
        return have

    now = pd.Timestamp.today()
    if have is None:
        years, done = range(start, end + 1), set()
    else:
        done = {(d.year, d.month) for d in pd.DatetimeIndex(have["asof"])}
        done.discard((now.year, now.month))          # see the docstring
        years = range(max(start, now.year - 1), end + 1)

    frames = []
    for year in years:
        try:
            names = _year_files(year)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("BoJ %d index unavailable: %s", year, exc)
            continue
        by_month = {}
        for n in names:
            by_month[n[3:7]] = n            # yymm -> last file that month (sorted)
        wanted = {ym: n for ym, n in by_month.items()
                  if (2000 + int(ym[:2]), int(ym[2:])) not in done}
        for ym, name in sorted(wanted.items()):
            try:
                f = _parse_mei(_curl(f"{BOJ_BASE}/{year}/{name}"), name)
            except Exception as exc:                                # noqa: BLE001
                logger.warning("  %s failed: %s", name, exc)
                continue
            f["asof"] = pd.Timestamp(f"20{name[3:5]}-{name[5:7]}-{name[7:9]}")
            frames.append(f)
        logger.info("BoJ %d: %d monthly snapshots (%d already stored)",
                    year, len(wanted), len(by_month) - len(wanted))

    if not frames:
        logger.info("BoJ store already current: %d snapshots to %s",
                    have["asof"].nunique(), have["asof"].max().date())
        return have

    df = pd.concat(frames, ignore_index=True)
    if have is not None:
        # A re-fetched month REPLACES its stored rows rather than doubling them --
        # which is how the current month's partial snapshot gets promoted to the
        # month-end one.
        fresh = {(d.year, d.month) for d in pd.DatetimeIndex(df["asof"])}
        kept = pd.DatetimeIndex(have["asof"]).map(lambda d: (d.year, d.month) not in fresh)
        df = pd.concat([have[np.asarray(kept, bool)], df], ignore_index=True)
    df = df.sort_values(["asof", "bond_type", "issue"]).reset_index(drop=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BOJ_RAW_PARQUET)
    logger.info("BoJ store: %d issue-months, %d snapshots, %s..%s", len(df),
                df["asof"].nunique(), df["asof"].min().date(), df["asof"].max().date())
    return df


def _main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ref = fetch_mof_issues(refresh="--refresh" in sys.argv)
    print(f"\nMoF reference: {len(ref)} issues across {ref.bond_type.nunique()} tenors")
    print(ref.groupby("bond_type").agg(
        issues=("issue", "size"), total_bn=("amount", "sum"),
        first_maturity=("maturity_date", "min"),
        last_maturity=("maturity_date", "max")).to_string())
    boj = fetch_boj(refresh="--refresh" in sys.argv)
    print(f"\nBoJ holdings: {len(boj):,} issue-months, "
          f"{boj['asof'].min().date()}..{boj['asof'].max().date()}, "
          f"{boj['asof'].nunique()} snapshots")
    print(boj.groupby("bond_type")["issue"].size().to_string())


if __name__ == "__main__":
    _main()
