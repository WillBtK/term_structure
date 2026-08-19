"""Fetch + cache Deutsche Bundesbank per-ISIN "Prices and yields of listed Federal
securities" (the raw material for the Bund liquidity 'noise' gauge).

Unlike the term-structure curve (SDMX flow BBSIS), the per-security reference prices
and yields are a **service download**: one monthly workbook per month, linked directly
from the index page at opaque per-file blob URLs. Each month exposes both an Excel file
(``YYYY-MM-excel-data.xlsx``) and a PDF (``YYYY-MM-data.pdf``); we take the XLSX. This
fetcher scrapes the index for the ``-excel-data.xlsx`` blobs, parses each workbook to a
tidy ``(date, isin, maturity, yield)`` panel, and unions it into the store.

HISTORY IS RETAINED, NOT RE-FETCHABLE. The index exposes only the most recent ~5
months, and no Bundesbank source reachable from the pipeline backfills the rest:

  * the SDMX per-ISIN yield flow ``BBSSY`` drops a security roughly four years after
    it matures (verified 2026-08: the disseminated set spans maturities 2022-2056;
    ``DE0001135051``, the Bund 4.125% of 2008, returns HTTP 404, and on 2005-06-15
    only 10 of the 121 series carry a value — the survivors, not the contemporaneous
    cross-section);
  * the Kapitalmarktstatistik monthly supplements are aggregate tables with no
    per-security rows.

So the union is written to ``vcfg.BUND_PRICES_STORE`` — a VERSIONED file the daily CI
job commits — and every run merges the fresh months into it. A trading day that scrolls
off the index survives only because it was captured while it was there.

Run ``python -m germany.fetch_bund_prices --probe`` to dump the discovered file URLs
and one workbook's structure (used to pin the column mapping); ``--refresh`` (the daily
job's default) scrapes the exposed months and merges them into the store.

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


# Column-keyword detection (English + German headers). The trading DATE is not a
# column — each sheet is one trading day, named "DD.MM.YYYY" (see parse_workbook).
_ISIN_KEYS = ("isin",)
_YIELD_KEYS = ("yield", "rendite")
_MAT_KEYS = ("maturity", "redemption", "fälligkeit", "faelligkeit")
_DESC_KEYS = ("description", "bezeichnung", "name")


def _find_col(cols, keys):
    for c in cols:
        cl = str(c).strip().lower()
        if any(k in cl for k in keys):
            return c
    return None


def parse_workbook(content: bytes) -> pd.DataFrame:
    """Parse one monthly workbook to tidy ``(date, isin, maturity, yield)`` (yield decimal).

    Layout (confirmed via --probe): one sheet per trading day, named ``DD.MM.YYYY``;
    each sheet has a title row, then a header row (``ISIN | Coupon | Description |
    Maturity | Residual life | Issue volume | Price | Yield in % | Dirty price``), then
    one row per outstanding security. The trading date is the SHEET NAME, not a column.
    Header row and columns are found by keyword so minor layout drift is tolerated.
    Returns an empty frame (logged) if no sheet yields the expected columns."""
    xls = pd.ExcelFile(io.BytesIO(content))
    frames = []
    for sheet in xls.sheet_names:
        # The sheet name is the trading day; skip non-date sheets (notes/cover).
        sheet_date = pd.to_datetime(str(sheet).strip(), format="%d.%m.%Y", errors="coerce")
        if pd.isna(sheet_date):
            continue
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
        c_desc = _find_col(df.columns, _DESC_KEYS)
        if not (c_isin and c_yield and c_mat):
            logger.warning("sheet %r missing cols (isin=%s yield=%s mat=%s)",
                           sheet, c_isin, c_yield, c_mat)
            continue
        cols = [c_isin, c_mat, c_yield] + ([c_desc] if c_desc else [])
        sub = df[cols].copy()
        sub.columns = ["isin", "maturity", "yield"] + (["name"] if c_desc else [])
        if not c_desc:
            sub["name"] = ""
        sub.insert(0, "date", sheet_date)
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["date", "isin", "maturity", "yield", "name"])
    out = pd.concat(frames, ignore_index=True)
    out["name"] = out["name"].astype(str).str.strip()
    out["maturity"] = pd.to_datetime(out["maturity"], errors="coerce", dayfirst=True)
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    # Bundesbank yields are in PERCENT -> decimal. Guard against already-decimal.
    yv = out["yield"].abs()
    if yv.dropna().median() > 1.0:
        out["yield"] = out["yield"] / 100.0
    out["isin"] = out["isin"].astype(str).str.strip()
    return out.dropna(subset=["date", "maturity", "yield"])


_PANEL_COLS = ["date", "isin", "maturity", "yield", "name"]


def _read_store() -> pd.DataFrame:
    """The accumulated per-ISIN archive, or an empty frame when there is none."""
    path = vcfg.BUND_PRICES_STORE
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("unreadable Bund store %s: %s", path, exc)
    return pd.DataFrame(columns=_PANEL_COLS)


def _merge(*frames: pd.DataFrame) -> pd.DataFrame:
    """Union of cross-sections, newest parse winning on a (date, isin) collision."""
    parts = [f for f in frames if f is not None and len(f)]
    if not parts:
        return pd.DataFrame(columns=_PANEL_COLS)
    out = pd.concat(parts, ignore_index=True)
    # keep="last" so the freshly scraped workbook supersedes an older parse of the
    # same trading day (revisions), while every day only IT has survives untouched.
    return (out.sort_values(["date", "isin"])
               .drop_duplicates(["date", "isin"], keep="last")
               .reset_index(drop=True))


def load_bund_yields(refresh: bool = True, limit: int | None = None) -> pd.DataFrame:
    """Tidy ``(date, isin, maturity, yield)`` panel of the per-ISIN cross-section.

    This ACCUMULATES. The Bundesbank index publishes only the most recent ~5
    monthly workbooks, and its per-ISIN yield archive (SDMX flow ``BBSSY``) retains
    a security only about four years past maturity — so a date that scrolls off the
    index cannot be re-fetched from anywhere. Each run therefore unions the freshly
    scraped months into the versioned store (``BUND_PRICES_STORE``, committed by
    CI) and returns the union, not just what the index happened to expose today.
    """
    store = _read_store()
    cache = vcfg.BUND_PRICES_RAW_PARQUET
    if not refresh:
        if len(store):
            return store
        return pd.read_parquet(cache) if cache.exists() else store
    scraped = []
    try:
        for u in discover_workbook_urls(limit=limit):
            try:
                scraped.append(parse_workbook(_get(u)))
            except Exception as exc:                                # noqa: BLE001
                logger.warning("parse failed %s: %s", u, exc)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("index fetch failed (%s); using the accumulated store", exc)
        if not len(store):
            if cache.exists():
                return pd.read_parquet(cache)
            raise
    panel = _merge(store, *scraped)
    if len(panel):
        before = len(store)
        for path in (vcfg.BUND_PRICES_STORE, cache):
            path.parent.mkdir(parents=True, exist_ok=True)
            panel.to_parquet(path)
        logger.info("Bund per-ISIN yields: %d rows (+%d new), %s..%s, %d ISINs",
                    len(panel), len(panel) - before,
                    panel["date"].min().date(), panel["date"].max().date(),
                    panel["isin"].nunique())
    return panel


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


def _diagnose() -> None:
    """Diagnostic: on the latest date, fit the NS curve and dump the worst-fitting
    bonds by name + a name-prefix breakdown, to spot non-nominal contaminants
    (inflation-linked / green) that inflate the 'noise'."""
    from validation.bund_liquidity import _ns_design
    panel = load_bund_yields(refresh=True, limit=1)
    if not len(panel):
        print("empty panel"); return
    last = panel["date"].max()
    g = panel[panel["date"] == last].copy()
    g["tau"] = (pd.to_datetime(g["maturity"]) - last).dt.days / 365.25
    g = g[(g["tau"] >= 1.0) & (g["tau"] <= 30.0)].dropna(subset=["yield"])
    tau = g["tau"].to_numpy(); y = g["yield"].to_numpy()
    X = _ns_design(tau)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    g["resid_bp"] = (y - X @ beta) * 1e4
    rms = float(np.sqrt(np.mean(((y - X @ beta) * 1e4) ** 2)))
    print(f"=== latest date {last.date()} | {len(g)} bonds in [1,30]y | RMS {rms:.1f} bp")
    g["prefix"] = g["name"].str.split().str[0].str.upper()
    print("\n-- name-prefix counts --")
    print(g["prefix"].value_counts().to_string())
    print("\n-- 12 worst-fitting bonds (|resid|) --")
    worst = g.reindex(g["resid_bp"].abs().sort_values(ascending=False).index).head(12)
    print(worst[["isin", "name", "tau", "yield", "resid_bp"]]
          .assign(tau=lambda d: d["tau"].round(1),
                  yld_bp=lambda d: (d["yield"] * 1e4).round(0),
                  resid_bp=lambda d: d["resid_bp"].round(0))
          [["isin", "name", "tau", "yld_bp", "resid_bp"]].to_string(index=False))
    # RMS excluding inflation-linked (heuristic: name/ISIN markers) for comparison.
    linker = g["name"].str.contains(r"\bei\b|inflation|index|infl|/i", case=False, regex=True) \
             | g["isin"].str.contains("103", na=False)
    gc = g[~linker]
    if len(gc) >= 6:
        Xc = _ns_design(gc["tau"].to_numpy())
        bc, *_ = np.linalg.lstsq(Xc, gc["yield"].to_numpy(), rcond=None)
        rc = float(np.sqrt(np.mean(((gc["yield"].to_numpy() - Xc @ bc) * 1e4) ** 2)))
        print(f"\n-- RMS excluding {int(linker.sum())} suspected linkers: {rc:.1f} bp "
              f"({len(gc)} bonds) --")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Bundesbank per-ISIN Bund prices/yields")
    ap.add_argument("--probe", action="store_true", help="dump discovered URLs + one workbook")
    ap.add_argument("--diagnose", action="store_true", help="fit latest date, dump worst bonds")
    ap.add_argument("--limit", type=int, default=None, help="max months to fetch")
    args = ap.parse_args()
    if args.probe:
        _probe(); return
    if args.diagnose:
        _diagnose(); return
    panel = load_bund_yields(refresh=True, limit=args.limit)
    print(f"per-ISIN panel: {len(panel)} rows"
          + (f", {panel['date'].min().date()}..{panel['date'].max().date()}, "
             f"{panel['isin'].nunique()} ISINs" if len(panel) else " (empty)"))


if __name__ == "__main__":
    _main()
