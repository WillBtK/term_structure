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


# Column detection. The Bundesbank changed the workbook layout between the June and
# July 2026 files, and the old one is still in the window:
#
#   new (2026-07 onward, 9 cols):  ISIN | Coupon | Description | Maturity | Residual
#                                  life | Issue volume | Price | Yield in % | Dirty price
#   old (through 2026-06, 11 cols): the ISIN is SPLIT ACROSS THREE CELLS
#                                  ("DE000" | "114183" | "6"), the coupon column has no
#                                  header at all, the maturity is dd.mm.yyyy text, and
#                                  the header itself is spread over three rows.
#
# Reading the old layout with the new one's rules silently produced ISINs like "114183"
# and descriptions of "DE000" — which also made drop_inflation_linked a no-op on those
# months, since it matches on the description. So the ISIN is reconstructed by CONTENT
# (concatenate cells until they form a valid ISIN) and the rest is found through a
# header joined across the header rows, with the coupon falling back to the column
# immediately after the ISIN block — true in both layouts.
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
_ISIN_KEYS = ("isin",)
_YIELD_KEYS = ("yield", "rendite")
_MAT_KEYS = ("maturity", "redemption", "fälligkeit", "faelligkeit")
_COUPON_KEYS = ("coupon", "kupon")
_DESC_KEYS = ("description", "bezeichnung", "name")
_HEADER_SPAN = 2                  # rows below the ISIN header that still carry labels


def _cell(v) -> str:
    """Cell as text, without the float formatting that turns 114183 into '114183.0'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return str(v).strip()


def _isin_span(rows, c_hdr: int, max_parts: int = 3):
    """``(start_column, width)`` of the cells that together form the ISIN.

    Decided by CONTENT, not by the header position: in the old layout the "ISIN 1)"
    label sits over the SECOND of the three cells ("DE000" | "114183" | "6"), so the
    block can begin one column to the left of its own header.
    """
    need = max(3, len(rows) // 2)
    for start in (c_hdr - 1, c_hdr):
        if start < 0:
            continue
        for k in range(1, max_parts + 1):
            hits = sum(1 for r in rows
                       if _ISIN_RE.match("".join(_cell(v) for v in r[start:start + k])))
            if hits >= need:
                return start, k
    return c_hdr, 1


def _joined_header(raw: pd.DataFrame, hdr: int) -> list[str]:
    """One label per column, joining the header row with the rows just below it — the
    old layout spreads 'Residual life', 'Price as of' and the like across three rows."""
    lo, hi = max(hdr - 1, 0), min(hdr + _HEADER_SPAN + 1, len(raw))
    return [" ".join(_cell(raw.iat[r, c]) for r in range(lo, hi)).strip().lower()
            for c in range(raw.shape[1])]


def _find_col(cols, keys, skip=()):
    for i, c in enumerate(cols):
        if i in skip:
            continue
        if any(k in str(c).strip().lower() for k in keys):
            return i
    return None


def parse_workbook(content: bytes) -> pd.DataFrame:
    """Parse one monthly workbook to tidy ``(date, isin, maturity, yield, name, coupon)``
    (yield decimal, coupon in percent).

    One sheet per trading day, named ``DD.MM.YYYY`` — the trading date is the SHEET
    NAME, not a column. Both published layouts are handled (see the notes above): the
    ISIN is rebuilt from however many cells it spans, and the remaining fields are found
    through a header joined across the header rows. Sheets that yield no valid ISINs are
    skipped with a warning rather than silently producing junk.
    """
    xls = pd.ExcelFile(io.BytesIO(content))
    frames = []
    for sheet in xls.sheet_names:
        sheet_date = pd.to_datetime(str(sheet).strip(), format="%d.%m.%Y", errors="coerce")
        if pd.isna(sheet_date):
            continue                                   # cover/notes sheet
        raw = xls.parse(sheet, header=None, dtype=object)
        hdr = None
        for i in range(min(15, len(raw))):
            if any("isin" in _cell(v).lower() for v in raw.iloc[i].tolist()):
                hdr = i
                break
        if hdr is None:
            continue
        header = _joined_header(raw, hdr)
        c_isin = _find_col(header, _ISIN_KEYS)
        if c_isin is None:
            continue
        body = raw.iloc[hdr + 1:].reset_index(drop=True)
        rows = [r.tolist() for _, r in body.iterrows()]
        c_isin, width = _isin_span(rows, c_isin)
        isin_cols = set(range(c_isin, c_isin + width))

        c_yield = _find_col(header, _YIELD_KEYS, skip=isin_cols)
        c_mat = _find_col(header, _MAT_KEYS, skip=isin_cols)
        c_desc = _find_col(header, _DESC_KEYS, skip=isin_cols)
        c_cpn = _find_col(header, _COUPON_KEYS, skip=isin_cols)
        if c_cpn is None and c_isin + width < raw.shape[1]:
            # The old layout leaves the coupon column unlabelled; in both layouts it is
            # the column immediately after the ISIN block.
            c_cpn = c_isin + width
        if c_yield is None or c_mat is None:
            logger.warning("sheet %r: no yield/maturity column (yield=%s mat=%s)",
                           sheet, c_yield, c_mat)
            continue

        recs = []
        for r in rows:
            isin = "".join(_cell(v) for v in r[c_isin:c_isin + width])
            if not _ISIN_RE.match(isin):
                continue                               # header remnant / footnote row
            recs.append({
                "date": sheet_date, "isin": isin,
                "maturity": _cell(r[c_mat]),
                "yield": r[c_yield],
                "name": _cell(r[c_desc]) if c_desc is not None else "",
                "coupon": r[c_cpn] if c_cpn is not None else np.nan,
            })
        if not recs:
            logger.warning("sheet %r: header found but no valid ISINs parsed", sheet)
            continue
        frames.append(pd.DataFrame(recs))

    cols = ["date", "isin", "maturity", "yield", "name", "coupon"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    out["maturity"] = pd.to_datetime(out["maturity"], errors="coerce", dayfirst=True,
                                     format="mixed")
    out["yield"] = pd.to_numeric(out["yield"], errors="coerce")
    # The coupon is carried because yield to maturity is not a function of maturity
    # alone — two bonds redeeming the same day with different coupons trade at different
    # YTMs — so the noise fit regresses it out (validation/noise_core).
    out["coupon"] = pd.to_numeric(out["coupon"], errors="coerce")
    # Bundesbank yields are in PERCENT -> decimal. Guard against already-decimal.
    if out["yield"].abs().dropna().median() > 1.0:
        out["yield"] = out["yield"] / 100.0
    return out.dropna(subset=["date", "maturity", "yield"])[cols]


_PANEL_COLS = ["date", "isin", "maturity", "yield", "name", "coupon"]


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
