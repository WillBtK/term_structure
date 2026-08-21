"""Refresh the PUBLIC neutral-anchor fallback inputs so japan.neutral_anchor's lower
rungs self-maintain when the (proprietary) Consensus survey lapses:

  - Ferreira r*  : Fed FEDS Note "Real-Time Global Longer-Run Neutral Rates" (real,
                   semi-annual) -> japan/data/anchors/ferreira_rstar_jp.csv
  - Tankan 5y    : BoJ "Inflation Outlook of Enterprises", 5-years-ahead, all industries
                   -> japan/data/anchors/tankan_infl_5y.csv

Runs on the CI runner (open egress); both hosts are blocked in the dev sandbox. Each
fetcher SCRAPES the official landing page for its data file (robust to filename churn),
parses defensively, and writes ONLY on a clean parse — it never overwrites a good
committed file on failure. URLs are env-overridable (`FERREIRA_URL`, `FERREIRA_CSV`,
`TANKAN_URL`, `TANKAN_CSV`). Best-effort: confirm the parse on the first CI run. Until
then the committed Consensus CSVs stay authoritative and nothing here changes the
published premium.

    python -m japan.fetch_neutral            # refresh both (prints a preview; never fails hard)
"""
from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)
ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
_UA = {"User-Agent": "term-premium-atlas/1.0 (research; contact via repo)"}
_TIMEOUT = 40

FERREIRA_NOTE = os.environ.get(
    "FERREIRA_URL",
    "https://www.federalreserve.gov/econres/notes/feds-notes/real-time-global-longer-run-neutral-rates-20250409.html")
TANKAN_NOTE = os.environ.get(
    "TANKAN_URL", "https://www.boj.or.jp/en/statistics/tk/bukka/index.htm")


def _find_data_href(page_url: str, exts=(".csv", ".xlsx", ".xls"), _depth: int = 1) -> "str | None":
    """Fetch a landing page and return the first absolute link to a data file. If none is
    directly linked, follow up to a few 'data/comprehensive/release' sub-links one level."""
    from urllib.parse import urljoin
    r = requests.get(page_url, headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', r.text, flags=re.I)
    cand = [h for h in hrefs if h.lower().split("?")[0].endswith(exts)]
    if cand:
        return urljoin(page_url, cand[0])
    if _depth > 0:
        subs = [urljoin(page_url, h) for h in hrefs
                if re.search(r"(comprehensive|dataset|/data/|release|bukka)", h, re.I)][:6]
        for su in subs:
            try:
                hit = _find_data_href(su, exts, _depth=0)
                if hit:
                    return hit
            except Exception:                                     # noqa: BLE001
                continue
    return None


def _period_to_date(x) -> "pd.Timestamp | None":
    """Parse a Ferreira half-year period label -> a month-end date. Handles
    '2024H1'/'2024:H1'/'2024:1'/'2024-S1'/'2024.5' (-> Jun-30), '2024H2'/'2024'/'2024.0'
    (-> Dec-31), quarter decimals '.25'/'.75', 'YYYYMM' ints, and pandas-parseable dates."""
    s = str(x).strip()
    m = re.match(r"^(\d{4})\s*[:\-_/]?\s*[HhSsQq]?\s*([12])$", s)          # 2024H1 / 2024:1 / 2024-S2
    if m:
        y, half = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(y, 6, 30) if half == 1 else pd.Timestamp(y, 12, 31)
    m = re.match(r"^(\d{4})\.(\d+)$", s)                                   # decimal year: 2024.0/.25/.5/.75
    if m:
        y, frac = int(m.group(1)), float("0." + m.group(2))
        month = {0.0: 12, 0.25: 3, 0.5: 6, 0.75: 9}.get(round(frac, 2))
        if month:
            return pd.Timestamp(y, month, 1) + pd.offsets.MonthEnd(0)
    if re.match(r"^\d{4}$", s):                                            # bare year -> year-end
        return pd.Timestamp(int(s), 12, 31)
    if re.match(r"^\d{6}$", s):                                            # YYYYMM
        return pd.Timestamp(int(s[:4]), int(s[4:6]), 1) + pd.offsets.MonthEnd(0)
    try:
        return pd.Timestamp(pd.to_datetime(s))
    except Exception:                                                      # noqa: BLE001
        return None


def _write_if_ok(name: str, s: pd.Series, out_dir=None) -> bool:
    """Write a clean date,value(percent) series to an anchor CSV; skip if empty.
    ``out_dir`` targets another market's anchor directory (default: Japan's)."""
    s = s.dropna().sort_index()
    if len(s) < 4:
        logger.warning("fetch_neutral: %s parse yielded <4 points -> keeping committed file", name)
        return False
    out = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in s.index],
                        "value": np.round(s.to_numpy(dtype=float), 4)})
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv((Path(out_dir) if out_dir is not None else ANCHOR_DIR) / name, index=False)
    logger.info("fetch_neutral: wrote %s (%d rows, %s..%s, latest %.3f)", name, len(out),
                out["date"].iloc[0], out["date"].iloc[-1], float(out["value"].iloc[-1]))
    return True


def fetch_ferreira_economy(code: str, aliases, out_name: str, out_dir=None) -> bool:
    """Scrape the FEDS Note for its data file and extract ONE economy's longer-run real
    neutral rate.

    Ferreira publishes every economy in a single file with 2-letter codes (JA=Japan,
    EA=euro area, US, UK, CA, SZ=Switzerland, ...), so every market's r* rung is served
    by this one weekly fetch — adding a market costs a column, not a new data source.
    Best-effort; never raises, and never overwrites a good committed file on a bad parse.
    """
    try:
        url = os.environ.get("FERREIRA_CSV") or _find_data_href(FERREIRA_NOTE)
        if not url:
            logger.warning("fetch_neutral: no Ferreira data link found on %s", FERREIRA_NOTE)
            return False
        raw = requests.get(url, headers=_UA, timeout=_TIMEOUT); raw.raise_for_status()
        df = (pd.read_excel(io.BytesIO(raw.content)) if url.lower().split("?")[0].endswith(("xlsx", "xls"))
              else pd.read_csv(io.StringIO(raw.content.decode("utf-8", "replace"))))
        cols = {str(c).strip().lower(): c for c in df.columns}
        want = tuple(a.lower() for a in aliases)
        col = next((cols[k] for k in cols if k in want or any(a in k for a in want if len(a) > 3)), None)
        pcol = next((cols[k] for k in cols if k in ("date", "period", "time", "year") or "date" in k or "period" in k),
                    df.columns[0])
        if col is None:
            logger.warning("fetch_neutral: no %s column in Ferreira file (cols=%s)",
                           code, list(df.columns)[:12])
            return False
        idx = df[pcol].map(_period_to_date)
        logger.info("fetch_neutral: Ferreira %s col=%s period=%s -> parsed %d/%d",
                    code, col, pcol, int(idx.notna().sum()), len(idx))
        s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=idx)
        s = s[s.index.notna()]
        return _write_if_ok(out_name, s, out_dir=out_dir)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("fetch_neutral: Ferreira %s fetch failed (%s)", code, exc)
        return False


def fetch_ferreira() -> bool:
    """Japan's rung: Ferreira longer-run real neutral -> ferreira_rstar_jp.csv."""
    return fetch_ferreira_economy("JA", ("japan", "jpn", "jp", "ja"), "ferreira_rstar_jp.csv")


def fetch_tankan() -> bool:
    """Delegates to japan.fetch_tankan.

    This used to scrape the standalone "Inflation Outlook of Enterprises" release page
    for a data link. That release was discontinued after 2019 and its landing page became
    an archive index -- still HTTP 200, but listing only the 2014 and 2016 sub-directories
    and carrying no data link at all -- so the scrape found nothing, logged a warning and
    returned False on every run since it was written. The rung was documented in the
    anchors README as Japan's public fallback and never once produced a file. The table
    now lives in section 7 of the main Tankan summary PDF, which the new module parses.
    """
    from japan.fetch_tankan import fetch_tankan as _fetch
    return _fetch()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ok_f, ok_t = fetch_ferreira(), fetch_tankan()
    logger.info("fetch_neutral done: ferreira=%s tankan=%s", ok_f, ok_t)


if __name__ == "__main__":
    main()
