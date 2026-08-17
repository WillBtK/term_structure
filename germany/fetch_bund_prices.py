"""Fetch + cache Deutsche Bundesbank per-ISIN "Prices and yields of listed Federal
securities" (the raw material for the Bund liquidity 'noise' gauge).

Unlike the term-structure curve (SDMX flow BBSIS), the per-security reference prices
and yields are a **service download**: one monthly workbook (recent months XLSX,
older PDF) reached from the index page, at opaque per-file blob URLs. So this fetcher
scrapes the index -> monthly landing pages -> the XLSX blob, parses each workbook to a
tidy ``(date, isin, maturity, yield)`` panel, and caches the union to parquet.

Egress: Bundesbank is only reachable from the daily GitHub job (open egress), not the
dev sandbox — so this is exercised in CI. Run ``python -m germany.fetch_bund_prices
--probe`` on a runner to dump the discovered file URLs and one workbook's structure
(used to pin the column mapping); ``--refresh`` (default in the daily job) rebuilds the
cache.

STATUS: the index/workbook layout is confirmed via the --probe run; the column
mapping below uses keyword detection and is hardened against layout drift.
"""
from __future__ import annotations

import io
import logging
import re

import numpy as np
import pandas as pd

from validation import config as vcfg

logger = logging.getLogger(__name__)

_BLOB_ANY_RE = re.compile(r'href="(/resource/blob/[^"]+?\.(?:xlsx|xls|csv|pdf))"', re.I)
_BLOB_DATA_RE = re.compile(r'href="(/resource/blob/[^"]+?\.(?:xlsx|xls|csv))"', re.I)  # parseable only
_MONTH_LINK_RE = re.compile(r'href="(/en/service/federal-securities/prices-and-yields/[^"]+?-\d{5,})"', re.I)
_HOST = "https://www.bundesbank.de"


def _get(url: str, timeout: int = 60) -> bytes:
    """GET raw bytes, requests with a urllib fallback."""
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.content
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()


def _abs(href: str) -> str:
    return href if href.startswith("http") else _HOST + href


def discover_workbook_urls(limit: int | None = None) -> list[str]:
    """Scrape the index -> monthly landing pages -> XLSX/CSV blob URLs (newest first)."""
    index_html = _get(vcfg.BUND_PRICES_INDEX).decode("utf-8", "replace")
    month_pages = []
    seen = set()
    for m in _MONTH_LINK_RE.finditer(index_html):
        u = _abs(m.group(1))
        if u not in seen:
            seen.add(u); month_pages.append(u)
    logger.info("discovered %d monthly landing pages", len(month_pages))
    files: list[str] = []
    for page in (month_pages if limit is None else month_pages[:limit]):
        try:
            html = _get(page).decode("utf-8", "replace")
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("landing page failed %s: %s", page, exc)
            continue
        blobs = [_abs(b.group(1)) for b in _BLOB_DATA_RE.finditer(html)]
        if blobs:
            files.append(blobs[0])                                 # one data file per month
    logger.info("discovered %d workbook files", len(files))
    return files


# Column-keyword detection (English + German headers).
_ISIN_KEYS = ("isin",)
_YIELD_KEYS = ("yield", "rendite")
_MAT_KEYS = ("maturity", "redemption", "fälligkeit", "faelligkeit")
_DATE_KEYS = ("date", "datum", "trading day", "day")


def _find_col(cols, keys):
    for c in cols:
        cl = str(c).strip().lower()
        if any(k in cl for k in keys):
            return c
    return None


def parse_workbook(content: bytes) -> pd.DataFrame:
    """Parse one monthly workbook to tidy ``(date, isin, maturity, yield)`` (yield decimal).

    Header row and columns are found by keyword so minor layout drift is tolerated.
    Returns an empty frame (logged) if the expected columns are absent."""
    xls = pd.ExcelFile(io.BytesIO(content))
    frames = []
    for sheet in xls.sheet_names:
        raw = xls.parse(sheet, header=None, dtype=object)
        # find the header row: the first row mentioning ISIN
        hdr = None
        for i in range(min(15, len(raw))):
            rowvals = [str(v).lower() for v in raw.iloc[i].tolist()]
            if any("isin" in v for v in rowvals):
                hdr = i; break
        if hdr is None:
            continue
        df = xls.parse(sheet, header=hdr)
        df.columns = [str(c).strip() for c in df.columns]
        c_isin = _find_col(df.columns, _ISIN_KEYS)
        c_yield = _find_col(df.columns, _YIELD_KEYS)
        c_mat = _find_col(df.columns, _MAT_KEYS)
        c_date = _find_col(df.columns, _DATE_KEYS)
        if not (c_isin and c_yield and c_mat and c_date):
            logger.warning("sheet %r missing cols (isin=%s yield=%s mat=%s date=%s)",
                           sheet, c_isin, c_yield, c_mat, c_date)
            continue
        sub = df[[c_date, c_isin, c_mat, c_yield]].copy()
        sub.columns = ["date", "isin", "maturity", "yield"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["date", "isin", "maturity", "yield"])
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    out["maturity"] = pd.to_datetime(out["maturity"], errors="coerce", dayfirst=True)
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    # Bundesbank yields are in PERCENT -> decimal. Guard against already-decimal.
    yv = out["yield"].abs()
    if yv.dropna().median() > 1.0:
        out["yield"] = out["yield"] / 100.0
    out["isin"] = out["isin"].astype(str).str.strip()
    return out.dropna(subset=["date", "maturity", "yield"])


def load_bund_yields(refresh: bool = True, limit: int | None = None) -> pd.DataFrame:
    """Union of parsed monthly workbooks -> tidy ``(date, isin, maturity, yield)`` panel,
    cached to parquet; falls back to the cache on any network failure."""
    path = vcfg.BUND_PRICES_RAW_PARQUET
    if not refresh and path.exists():
        return pd.read_parquet(path)
    try:
        urls = discover_workbook_urls(limit=limit)
        frames = []
        for u in urls:
            try:
                frames.append(parse_workbook(_get(u)))
            except Exception as exc:                                # noqa: BLE001
                logger.warning("parse failed %s: %s", u, exc)
        panel = (pd.concat(frames, ignore_index=True).drop_duplicates(["date", "isin"])
                 if frames else pd.DataFrame(columns=["date", "isin", "maturity", "yield"]))
        if len(panel):
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(path)
            logger.info("Bund per-ISIN yields: %d rows, %s..%s, %d ISINs",
                        len(panel), panel["date"].min().date(), panel["date"].max().date(),
                        panel["isin"].nunique())
        return panel
    except Exception as exc:                                        # noqa: BLE001
        if path.exists():
            logger.warning("fetch failed (%s); using cached per-ISIN panel", exc)
            return pd.read_parquet(path)
        raise


def _probe() -> None:
    """Diagnostic: dump discovered URLs + one workbook's structure (run on a runner)."""
    index_html = _get(vcfg.BUND_PRICES_INDEX).decode("utf-8", "replace")
    month_pages = []
    seen = set()
    for m in _MONTH_LINK_RE.finditer(index_html):
        u = _abs(m.group(1))
        if u not in seen:
            seen.add(u); month_pages.append(u)
    print(f"=== index {vcfg.BUND_PRICES_INDEX}")
    print(f"    monthly landing pages found: {len(month_pages)}")
    for u in month_pages[:5]:
        print("   ", u)
    if not month_pages:
        print("!! no landing pages — dumping index sample for regex tuning:")
        print(index_html[:2000])
        return
    # Show ALL file links (any extension) on the newest landing page.
    lp = month_pages[0]
    lp_html = _get(lp).decode("utf-8", "replace")
    all_files = [_abs(b.group(1)) for b in _BLOB_ANY_RE.finditer(lp_html)]
    print(f"\n=== landing page {lp}")
    print(f"    file links (any ext): {len(all_files)}")
    for u in all_files:
        print("   ", u)
    urls = discover_workbook_urls(limit=3)
    print(f"\n=== parseable (xlsx/csv) workbook URLs: {len(urls)} ===")
    for u in urls:
        print(" ", u)
    if not urls:
        print("!! no XLSX/CSV — files are likely PDF-only (would need pdf parsing).")
        return
    content = _get(urls[0])
    print(f"\n=== workbook {urls[0]} ({len(content)} bytes) ===")
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        print("sheets:", xls.sheet_names)
        for sheet in xls.sheet_names[:2]:
            raw = xls.parse(sheet, header=None, dtype=object)
            print(f"\n--- sheet {sheet!r} first 12 rows ---")
            print(raw.head(12).to_string(max_cols=12))
    except Exception as exc:                                        # noqa: BLE001
        print("not a parseable workbook:", exc)
    print("\n=== parse_workbook result ===")
    df = parse_workbook(content)
    print(df.head(8).to_string(index=False))
    print("rows:", len(df), "| cols:", list(df.columns))


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Bundesbank per-ISIN Bund prices/yields")
    ap.add_argument("--probe", action="store_true", help="dump discovered URLs + one workbook")
    ap.add_argument("--limit", type=int, default=None, help="max months to fetch")
    args = ap.parse_args()
    if args.probe:
        _probe(); return
    panel = load_bund_yields(refresh=True, limit=args.limit)
    print(f"per-ISIN panel: {len(panel)} rows"
          + (f", {panel['date'].min().date()}..{panel['date'].max().date()}, "
             f"{panel['isin'].nunique()} ISINs" if len(panel) else " (empty)"))


if __name__ == "__main__":
    _main()
