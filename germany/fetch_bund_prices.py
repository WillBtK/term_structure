"""Fetch + cache Deutsche Bundesbank per-ISIN "Prices and yields of listed Federal
securities" (the raw material for the Bund liquidity 'noise' gauge).

Unlike the term-structure curve (SDMX flow BBSIS), the per-security reference prices
and yields are a **service download**: one monthly workbook per month, linked directly
from the index page at opaque per-file blob URLs. Each month exposes both an Excel file
(``YYYY-MM-excel-data.xlsx``) and a PDF (``YYYY-MM-data.pdf``); we take the XLSX. The
index shows the most recent ~5 months, so the fetcher accumulates the union into a
parquet cache (history grows month-by-month as the daily job runs). This fetcher scrapes
the index for the ``-excel-data.xlsx`` blobs, parses each workbook to a tidy
``(date, isin, maturity, yield)`` panel, and caches the union.

Egress: Bundesbank is only reachable from the daily GitHub job (open egress), not the
dev sandbox — so this is exercised in CI. Run ``python -m germany.fetch_bund_prices
--probe`` on a runner to dump the discovered file URLs and one workbook's structure
(used to pin the column mapping); ``--refresh`` (default in the daily job) rebuilds the
cache.

STATUS: the index/workbook layout is confirmed via the --probe run (files linked
directly from the index; XLSX available); the column mapping uses keyword detection
and is hardened against layout drift.
"""
from __future__ import annotations

import io
import logging
import re

import numpy as np
import pandas as pd

from validation import config as vcfg

logger = logging.getLogger(__name__)

# Monthly Excel workbooks are linked directly from the index, e.g.
#   /resource/blob/972742/<hash>/<hash>/2026-08-excel-data.xlsx
# Capture the URL and the YYYY-MM stamp so we can order newest-first.
_EXCEL_LINK_RE = re.compile(
    r'(/resource/blob/[^"\']+?/(\d{4})-(\d{2})-excel-data\.xlsx)', re.I)
_BLOB_ANY_RE = re.compile(r'(/resource/blob/[^"\']+?\.(?:xlsx|xls|csv|pdf))', re.I)
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
    """Scrape the index for monthly ``-excel-data.xlsx`` blob URLs, newest first.

    Files are linked directly from the index page (no per-month landing pages); the
    page exposes roughly the most recent five months."""
    index_html = _get(vcfg.BUND_PRICES_INDEX).decode("utf-8", "replace")
    found = {}
    for m in _EXCEL_LINK_RE.finditer(index_html):
        url, yy, mm = _abs(m.group(1)), m.group(2), m.group(3)
        found[(yy, mm)] = url                       # dedupe by month, keep the URL
    ordered = [found[k] for k in sorted(found, reverse=True)]   # newest month first
    logger.info("discovered %d monthly XLSX workbooks from the index", len(ordered))
    return ordered if limit is None else ordered[:limit]


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
    print(f"=== index {vcfg.BUND_PRICES_INDEX}")
    urls = discover_workbook_urls()
    print(f"    monthly XLSX workbooks discovered: {len(urls)}")
    for u in urls:
        print("   ", u)
    if not urls:
        print("!! no XLSX links found — dumping any blob file links for tuning:")
        index_html = _get(vcfg.BUND_PRICES_INDEX).decode("utf-8", "replace")
        for m in _BLOB_ANY_RE.finditer(index_html):
            print("   ", _abs(m.group(1)))
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
