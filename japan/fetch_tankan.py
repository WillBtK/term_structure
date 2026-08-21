"""Tankan 'Inflation Outlook of Enterprises': the public rung for Japanese long-run
inflation expectations.

The Bank of Japan asks every Tankan respondent what it expects GENERAL prices to do one,
three and five years ahead. The five-year figure for all enterprises across all
industries is the only public, machine-readable, long-horizon Japanese inflation
expectation there is, which makes it the rung that keeps the proprietary Consensus
6-10y anchor current once that snapshot stops being refreshed.

Getting it is not as simple as it should be, and the previous attempt at this failed
silently for a reason worth recording. Between 2014 and 2019 the BoJ published the
inflation outlook as its own release (``tk/bukka/.../tkc<yymm>.pdf``); the fetcher that
existed scraped that landing page for a data link. From 2020 the standalone release was
discontinued and the table folded into section 7 of the main Tankan summary
(``tk/gaiyo/<era>/tka<yymm>.pdf``). The old page still returns HTTP 200 -- it is now an
archive index listing only the 2014 and 2016 sub-directories -- so the scrape found no
link, logged a warning, returned False, and the anchor quietly went without its rung.

The current data lives in BoJ's stat-search behind a CGI that will not answer a script,
and its bulk export carries the latest release only, with 48,430 unlabelled series
codes. The release PDF is the reliable route, and it is a fixed table.

Section 7 is a 5 x 3 grid -- large/small enterprises crossed with manufacturing and
non-manufacturing, then all enterprises, each with the three horizons -- and every cell
is a pair of rows, previous projection and current. Each row carries four numbers:
output-price level and change, then general-price level and change. So the fifteenth and
last 'Current projection' row is all enterprises, all industries, five years ahead, and
the third number in it is the one we want.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
OUT = "tankan_infl_5y.csv"

BASE = "https://www.boj.or.jp"
# BoJ files the Tankan summary by publication era, not by year of content, so every era
# directory has to be walked to build the history. 2011 is where the price-outlook
# question begins (2014Q1) comfortably inside the range.
ERAS = ("2011", "2016", "2021", "2026")
INDEX = BASE + "/en/statistics/tk/gaiyo/{era}/index.htm"

_UA = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 60
# Four numbers a row: output price, its change, general price, its change.
_ROW = re.compile(r"Current projection\s+(-?\d+\.\d)\s+(-?\d+\.\d)\s+"
                  r"(-?\d+\.\d)\s+(-?\d+\.\d)")
N_BLOCKS = 15                       # 5 enterprise groups x 3 horizons


def release_urls() -> list:
    """Every Tankan summary PDF the BoJ still publishes, oldest first."""
    out = []
    for era in ERAS:
        try:
            r = requests.get(INDEX.format(era=era), headers=_UA, timeout=_TIMEOUT)
            r.raise_for_status()
        except Exception as exc:                                    # noqa: BLE001
            logger.warning("tankan: era %s index unavailable (%s)", era, exc)
            continue
        out += re.findall(r'href="(/en/statistics/tk/gaiyo/[^"]+/tka\d{4}\.pdf)"', r.text)
    return sorted(set(out))


def _release_date(url: str) -> "pd.Timestamp | None":
    """``tka2606.pdf`` is the June 2026 round. Dated at the survey month's end."""
    m = re.search(r"tka(\d{2})(\d{2})\.pdf", url)
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=2000 + yy, month=mm, day=1) + pd.offsets.MonthEnd(0)


def parse_release(blob: bytes) -> "float | None":
    """Five-year-ahead general-price expectation, all enterprises, all industries."""
    import pypdf
    text = ""
    for page in pypdf.PdfReader(io.BytesIO(blob)).pages:
        t = page.extract_text() or ""
        if "Inflation Outlook of Enterprises" in t:
            text = t
            break
    if not text:
        return None
    rows = _ROW.findall(text)
    if len(rows) < N_BLOCKS:
        logger.debug("tankan: found %d current-projection rows, expected %d",
                     len(rows), N_BLOCKS)
        return None
    value = float(rows[N_BLOCKS - 1][2])                # general prices, 5 years ahead
    return value if 0.0 <= value <= 10.0 else None


def fetch_tankan(out_dir: "Path | None" = None) -> bool:
    """Walk every release and write the five-year series. Best-effort; never raises."""
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    rows, missed = [], 0
    try:
        for url in release_urls():
            date = _release_date(url)
            if date is None:
                continue
            try:
                r = requests.get(BASE + url, headers=_UA, timeout=_TIMEOUT)
                r.raise_for_status()
                v = parse_release(r.content)
            except Exception as exc:                                # noqa: BLE001
                logger.debug("tankan: %s failed (%s)", url, exc)
                v = None
            if v is None:
                missed += 1                 # pre-2014 rounds predate the question
                continue
            rows.append({"date": date, "value": v})
        if len(rows) < 20:
            raise ValueError(f"only {len(rows)} rounds parsed -- refusing to overwrite")
        df = pd.DataFrame(rows).drop_duplicates("date").sort_values("date")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / OUT, index=False)
        logger.info("Tankan 5y general prices: %d rounds %s..%s, latest %.1f%% "
                    "(%d releases carried no table)", len(df), df["date"].min().date(),
                    df["date"].max().date(), df["value"].iloc[-1], missed)
        return True
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("Tankan refresh failed (%s) -- keeping committed file", exc)
        return False


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_tankan()


if __name__ == "__main__":
    _main()
