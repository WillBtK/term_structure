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


def _find_data_href(page_url: str, exts=(".csv", ".xlsx", ".xls")) -> "str | None":
    """Fetch a landing page and return the first absolute link to a data file."""
    r = requests.get(page_url, headers=_UA, timeout=_TIMEOUT)
    r.raise_for_status()
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', r.text, flags=re.I)
    cand = [h for h in hrefs if h.lower().split("?")[0].endswith(exts)]
    if not cand:
        return None
    from urllib.parse import urljoin
    return urljoin(page_url, cand[0])


def _period_to_date(x) -> "pd.Timestamp | None":
    """Parse a Ferreira period label -> a month-end date: '2024H1'/'2024:H1'/'2024.5'
    -> Jun-30; '2024H2'/'2024' -> Dec-31; else pandas date parse."""
    s = str(x).strip()
    m = re.match(r"^(\d{4})\s*[:\-]?\s*[Hh]?\s*([12])$", s)
    if m:
        y, half = int(m.group(1)), int(m.group(2))
        return pd.Timestamp(y, 6, 30) if half == 1 else pd.Timestamp(y, 12, 31)
    if re.match(r"^\d{4}(\.0)?$", s):
        return pd.Timestamp(int(float(s)), 12, 31)
    if re.match(r"^\d{4}\.5$", s):
        return pd.Timestamp(int(float(s)), 6, 30)
    try:
        return pd.Timestamp(pd.to_datetime(s))
    except Exception:                                              # noqa: BLE001
        return None


def _write_if_ok(name: str, s: pd.Series) -> bool:
    """Write a clean date,value(percent) series to the anchor CSV; skip if empty."""
    s = s.dropna().sort_index()
    if len(s) < 4:
        logger.warning("fetch_neutral: %s parse yielded <4 points -> keeping committed file", name)
        return False
    out = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in s.index],
                        "value": np.round(s.to_numpy(dtype=float), 4)})
    ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(ANCHOR_DIR / name, index=False)
    logger.info("fetch_neutral: wrote %s (%d rows, %s..%s, latest %.3f)", name, len(out),
                out["date"].iloc[0], out["date"].iloc[-1], float(out["value"].iloc[-1]))
    return True


def fetch_ferreira() -> bool:
    """Scrape the FEDS Note for its data file, extract the Japan longer-run real neutral,
    write ferreira_rstar_jp.csv. Best-effort; never raises."""
    try:
        url = os.environ.get("FERREIRA_CSV") or _find_data_href(FERREIRA_NOTE)
        if not url:
            logger.warning("fetch_neutral: no Ferreira data link found on %s", FERREIRA_NOTE)
            return False
        raw = requests.get(url, headers=_UA, timeout=_TIMEOUT); raw.raise_for_status()
        df = (pd.read_excel(io.BytesIO(raw.content)) if url.lower().split("?")[0].endswith(("xlsx", "xls"))
              else pd.read_csv(io.StringIO(raw.content.decode("utf-8", "replace"))))
        cols = {str(c).strip().lower(): c for c in df.columns}
        jp = next((cols[k] for k in cols if k in ("japan", "jpn", "jp") or "japan" in k), None)
        pcol = next((cols[k] for k in cols if k in ("date", "period", "time", "year") or "date" in k or "period" in k),
                    df.columns[0])
        if jp is None:
            logger.warning("fetch_neutral: no Japan column in Ferreira file (cols=%s)", list(df.columns)[:12])
            return False
        idx = df[pcol].map(_period_to_date)
        s = pd.Series(pd.to_numeric(df[jp], errors="coerce").to_numpy(), index=idx)
        s = s[s.index.notna()]
        return _write_if_ok("ferreira_rstar_jp.csv", s)
    except Exception as e:                                         # noqa: BLE001
        logger.warning("fetch_neutral: Ferreira refresh failed (%s) -> keeping committed file", e)
        return False


def fetch_tankan() -> bool:
    """Scrape the BoJ 'Inflation Outlook of Enterprises' page for its data file and
    extract the 5-years-ahead all-enterprises average. Best-effort; never raises.
    (BoJ's comprehensive file is complex/Shift-JIS — confirm the parse on first CI run.)"""
    try:
        url = os.environ.get("TANKAN_CSV") or _find_data_href(TANKAN_NOTE)
        if not url:
            logger.warning("fetch_neutral: no Tankan data link found on %s", TANKAN_NOTE)
            return False
        raw = requests.get(url, headers=_UA, timeout=_TIMEOUT); raw.raise_for_status()
        if url.lower().split("?")[0].endswith(("xlsx", "xls")):
            df = pd.read_excel(io.BytesIO(raw.content), header=None)
        else:
            txt = raw.content.decode("shift_jis", "replace")
            df = pd.read_csv(io.StringIO(txt), header=None)
        # Heuristic: a row of quarterly period labels + a '5 year'/'5-year' column tag.
        flat = df.astype(str)
        col5 = None
        for j in range(flat.shape[1]):
            if flat[j].str.contains(r"5\s*[- ]?\s*year", case=False, regex=True).any():
                col5 = j; break
        if col5 is None:
            logger.warning("fetch_neutral: no '5-year' column found in Tankan file -> keeping committed file")
            return False
        per = df.iloc[:, 0].map(lambda v: pd.to_datetime(str(v), errors="coerce"))
        s = pd.Series(pd.to_numeric(df.iloc[:, col5], errors="coerce").to_numpy(), index=per)
        s = s[s.index.notna()]
        return _write_if_ok("tankan_infl_5y.csv", s)
    except Exception as e:                                         # noqa: BLE001
        logger.warning("fetch_neutral: Tankan refresh failed (%s) -> keeping committed file", e)
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ok_f, ok_t = fetch_ferreira(), fetch_tankan()
    logger.info("fetch_neutral done: ferreira=%s tankan=%s", ok_f, ok_t)


if __name__ == "__main__":
    main()
