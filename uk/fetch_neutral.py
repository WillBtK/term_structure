"""Fetch the UK endpoint survey — the Bank of England **Market Participants Survey**.

Published per MPC round since 2022 as one xlsx of aggregate percentiles from ~70-90
professional respondents. The anchor object is the DIRECT neutral-rate question,
"where do you see the level of Bank Rate at which monetary policy is neither
expansionary nor contractionary", asked every round since June 2023.

The question's NUMBER drifts between rounds (1g, 1f, 1e, 1ei, 1b, 1bi ...) and the
workbook carries no question text, so the number is read off the round's HTML results
page by the question's wording and then matched to the workbook sheet. Matching on
sheet name alone silently picked up other questions (Sep-2026 fell back to the
five-years-ahead Bank Rate after the sheet was renamed Q1bi). Rounds without the
neutral question are skipped, not substituted: a different question is a different
object.

Central estimate: the INTERQUARTILE MEAN (p25+p50+p75)/3. The survey publishes no
mean, and the median is a step function on the 25bp grid respondents answer in.

This is the UK anchor's living carry-forward after the one-off Consensus snapshot,
level-bridged over their overlap. Best-effort: rounds that do not exist 404 and are
skipped; a bad parse never overwrites the committed file.
"""
from __future__ import annotations

import html as _html
import io
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
OUT = "boe_mps_neutral.csv"
OUT_CPI = "boe_mps_cpi_5y.csv"
_URL = ("https://www.bankofengland.co.uk/-/media/boe/files/markets/market-intelligence/"
        "survey/{year}/market-participants-survey-results-{month}-{year}.xlsx")
_PAGE = ("https://www.bankofengland.co.uk/markets/market-intelligence/survey-results/"
         "{year}/market-participants-survey-results-{month}-{year}")
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]
FIRST_YEAR = 2022
_NEUTRAL_Q = re.compile(r"(\d+[a-z]+(?:\s*\(?[ivx]+\)?)?)\)[^)]{0,160}?"
                        r"neither expansionary nor contractionary", re.I)


def _norm(label: str) -> str:
    return re.sub(r"[\s()]", "", str(label)).lower().lstrip("q")


def neutral_label(page_html: str) -> "str | None":
    """The neutral question's number on one round's results page ('1bi'), or None."""
    txt = re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", page_html)))
    m = _NEUTRAL_Q.search(txt)
    return _norm(m.group(1)) if m else None


def _round_value(content: bytes, label: str):
    """(p25, p50, p75) of the neutral question from one round's workbook, or None."""
    x = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for name, df in x.items():
        if _norm(name) != label or df.shape[0] < 2:
            continue
        hdr = df.iloc[0].astype(str).str.lower().tolist()
        try:
            cols = [next(i for i, h in enumerate(hdr) if p in h)
                    for p in ("25th", "50th", "75th")]
        except StopIteration:
            return None
        v = [pd.to_numeric(df.iloc[1, c], errors="coerce") for c in cols]
        if all(pd.notna(q) and 0.0 <= float(q) <= 8.0 for q in v) and v[0] <= v[1] <= v[2]:
            return tuple(float(q) for q in v)
    return None


def _round_cpi(content: bytes):
    """Median five-years-ahead CPI inflation from one round's workbook, or None.

    The MPS asks its inflation question on a sheet whose rows run 'End-2026 Q2 ...
    One year ahead ... Five years ahead'. That last row is the UK's public long-run
    inflation expectation, and it is what keeps the Consensus anchor current once the
    proprietary snapshot goes stale. The sheet is found by its row labels rather than by
    name, since the question numbering drifts between rounds -- and it is distinguished
    from the Bank Rate sheet, which also carries a 'Five years ahead' row, by having no
    MPC meeting dates in it.
    """
    x = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)
    for name, df in x.items():
        if 0 not in df.columns:
            continue
        col0 = df[0].astype(str)
        if col0.str.contains("MPC", na=False).sum() >= 3:
            continue                                  # that is the Bank Rate path
        if not col0.str.contains(r"(?:One|Two|Three) years? ahead", case=False,
                                 regex=True, na=False).any():
            continue                                  # inflation path has this ladder
        hdr = df.iloc[0].astype(str).str.lower().tolist()
        cols = [i for i, h in enumerate(hdr) if "50th" in h]
        rows = df[col0.str.contains("Five years ahead", case=False, na=False)]
        if cols and len(rows):
            v = pd.to_numeric(rows.iloc[0, cols[0]], errors="coerce")
            if pd.notna(v) and 0.0 <= float(v) <= 10.0:
                return float(v)
    return None


def fetch_boe_mps(out_dir: "Path | None" = None) -> bool:
    """Crawl every round since 2022 and write the neutral-question series."""
    import requests
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    rows, cpi = [], []
    today = pd.Timestamp.today()
    ua = {"User-Agent": "Mozilla/5.0"}
    try:
        for year in range(FIRST_YEAR, today.year + 1):
            for mi, month in enumerate(_MONTHS, start=1):
                if pd.Timestamp(year=year, month=mi, day=1) > today:
                    break
                try:
                    r = requests.get(_URL.format(year=year, month=month), timeout=45,
                                     headers=ua)
                    if r.status_code != 200 or b"PK" != r.content[:2]:  # xlsx = zip
                        continue
                    stamp = pd.Timestamp(year=year, month=mi, day=15)
                    infl = _round_cpi(r.content)
                    if infl is not None:
                        cpi.append({"date": stamp, "value": infl})
                    pg = requests.get(_PAGE.format(year=year, month=month), timeout=45,
                                      headers=ua)
                    label = neutral_label(pg.text) if pg.status_code == 200 else None
                    q = _round_value(r.content, label) if label else None
                    if q:
                        rows.append({"date": stamp, "value": round(sum(q) / 3.0, 4),
                                     "p25": q[0], "p50": q[1], "p75": q[2],
                                     "question": label})
                except Exception as exc:                            # noqa: BLE001
                    logger.debug("MPS %s-%d: %s", month, year, exc)
        if len(rows) < 8:
            raise ValueError(f"only {len(rows)} rounds parsed — refusing to overwrite")
        df = pd.DataFrame(rows).sort_values("date")
        if not (0.0 <= df["value"].median() <= 8.0):
            raise ValueError("implausible values")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / OUT, index=False, date_format="%Y-%m-%d")
        logger.info("BoE MPS neutral: %d rounds %s..%s; latest IQ mean %.2f%% "
                    "(quartiles %s)", len(df), df["date"].min().date(),
                    df["date"].max().date(), df["value"].iloc[-1],
                    df[["p25", "p50", "p75"]].iloc[-1].tolist())
        if len(cpi) >= 8:
            c = pd.DataFrame(cpi).sort_values("date")
            c.to_csv(out / OUT_CPI, index=False)
            logger.info("BoE MPS 5y CPI: %d rounds %s..%s, latest %.2f%%", len(c),
                        c["date"].min().date(), c["date"].max().date(),
                        c["value"].iloc[-1])
        else:
            logger.warning("BoE MPS 5y CPI: only %d rounds parsed - keeping committed "
                           "file", len(cpi))
        return True
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("BoE MPS fetch failed (%s) — keeping committed file", exc)
        return False


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_boe_mps()


if __name__ == "__main__":
    _main()
