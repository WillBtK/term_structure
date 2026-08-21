"""Fetch the UK endpoint survey — the Bank of England **Market Participants Survey**.

Published per MPC round since 2022 as one xlsx of aggregate percentiles from ~70-80
professional respondents. Two usable objects:

  * ``Q1b`` — "the level of Bank Rate at which monetary policy is neither expansionary
    nor contractionary": a DIRECT neutral-rate question, the endpoint concept itself;
  * ``Q1a`` — the Bank Rate path table, whose "Five years ahead" row is the fallback
    for rounds that predate the neutral question.

The median becomes the UK anchor's public carry-forward rung, level-bridged onto the
Consensus 6-10y history over their multi-year overlap. Best-effort: rounds that do not
exist 404 and are skipped; a bad parse never overwrites the committed file.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
OUT = "boe_mps_neutral.csv"
OUT_CPI = "boe_mps_cpi_5y.csv"
_URL = ("https://www.bankofengland.co.uk/-/media/boe/files/markets/market-intelligence/"
        "survey/{year}/market-participants-survey-results-{month}-{year}.xlsx")
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]
FIRST_YEAR = 2022


def _round_value(content: bytes):
    """(median, which_question) from one round's workbook, or None.

    Q1b is a single percentile row; if absent, fall back to Q1a's 'Five years ahead'
    row. Sheets are located by name, values by header text — resilient to the row
    count changing between rounds.
    """
    x = pd.read_excel(io.BytesIO(content), sheet_name=None, header=None)

    def _ok(v):                       # a Bank Rate level, not a stock or an index
        return pd.notna(v) and 0.0 <= float(v) <= 8.0

    if "Q1b" in x:
        df = x["Q1b"]
        hdr = df.iloc[0].astype(str).str.lower().tolist()
        if any("50th" in h for h in hdr):
            col = [i for i, h in enumerate(hdr) if "50th" in h][0]
            v = pd.to_numeric(df.iloc[1, col], errors="coerce")
            if _ok(v):
                return float(v), "neutral"
    # Fallback for rounds predating the neutral question: the five-years-ahead point of
    # the BANK RATE path. Sheet numbering drifts across rounds and several sheets carry
    # a "Five years ahead" row (CPI, gilt stock — one early round read 400 from it), so
    # the Bank Rate sheet is identified by what only it has: rows labelled with MPC
    # meeting dates.
    for name, df in x.items():
        col0 = df[0].astype(str) if 0 in df.columns else pd.Series(dtype=str)
        if col0.str.contains("MPC", na=False).sum() < 3:
            continue
        hdr = df.iloc[0].astype(str).str.lower().tolist()
        cols = [i for i, h in enumerate(hdr) if "50th" in h]
        rows = df[col0.str.contains("Five years ahead", case=False, na=False)]
        if cols and len(rows):
            v = pd.to_numeric(rows.iloc[0, cols[0]], errors="coerce")
            if _ok(v):
                return float(v), "five_years"
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
    """Crawl every round since 2022 and write the median series."""
    import requests
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    rows, cpi = [], []
    today = pd.Timestamp.today()
    try:
        for year in range(FIRST_YEAR, today.year + 1):
            for mi, month in enumerate(_MONTHS, start=1):
                if pd.Timestamp(year=year, month=mi, day=1) > today:
                    break
                try:
                    r = requests.get(_URL.format(year=year, month=month), timeout=45,
                                     headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code != 200 or b"PK" != r.content[:2]:  # xlsx = zip
                        continue
                    stamp = pd.Timestamp(year=year, month=mi, day=15)
                    got = _round_value(r.content)
                    if got:
                        rows.append({"date": stamp, "value": got[0],
                                     "question": got[1]})
                    infl = _round_cpi(r.content)
                    if infl is not None:
                        cpi.append({"date": stamp, "value": infl})
                except Exception as exc:                            # noqa: BLE001
                    logger.debug("MPS %s-%d: %s", month, year, exc)
        if len(rows) < 8:
            raise ValueError(f"only {len(rows)} rounds parsed — refusing to overwrite")
        df = pd.DataFrame(rows).sort_values("date")
        med = df["value"]
        if not (0.0 <= med.median() <= 8.0):
            raise ValueError("implausible medians")
        out.mkdir(parents=True, exist_ok=True)
        df[["date", "value", "question"]].to_csv(out / OUT, index=False)
        n_neu = int((df["question"] == "neutral").sum())
        logger.info("BoE MPS: %d rounds %s..%s (%d neutral-question, %d five-year); "
                    "latest %.2f%%", len(df), df["date"].min().date(),
                    df["date"].max().date(), n_neu, len(df) - n_neu, df["value"].iloc[-1])
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
