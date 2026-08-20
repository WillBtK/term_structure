"""Fetch the Canada endpoint survey — the Bank of Canada **Market Participants Survey**.

Quarterly since 2023Q1, ~25-30 professional respondents, published as an HTML results
page ~two weeks after the MPR. The question this fetcher extracts is the endpoint
concept verbatim: "What is your estimate of the long-term nominal neutral rate in
Canada?" — median, with quartiles around it.

The median becomes the Canadian anchor's public carry-forward rung, level-bridged onto
the Consensus 6-10y T-bill history (the ~25bp gap between a T-bill average and a
neutral policy rate is exactly what the bridge calibration absorbs). Best-effort:
missing rounds are skipped; a bad parse never overwrites the committed file.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
OUT = "boc_mps_neutral.csv"
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}
FIRST = (2023, 1)

_NEUTRAL_RE = re.compile(
    r"long-term nominal neutral rate.{0,400}?Median of responses\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE | re.DOTALL)


def _candidates(year: int, q: int):
    """Plausible publication URLs for one survey quarter (published ~1-2 months after)."""
    months = {1: [(year, 4), (year, 5)], 2: [(year, 7), (year, 8)],
              3: [(year, 10), (year, 11)], 4: [(year + 1, 1), (year + 1, 2)]}[q]
    for (py, pm) in months:
        yield (f"https://www.bankofcanada.ca/{py}/{pm:02d}/"
               f"market-participants-survey-{_ORD[q]}-quarter-of-{year}/")


def fetch_boc_mps(out_dir: "Path | None" = None) -> bool:
    import requests
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    rows = []
    today = pd.Timestamp.today()
    try:
        y, q = FIRST
        while pd.Timestamp(year=y, month=3 * q, day=1) <= today:
            for url in _candidates(y, q):
                try:
                    r = requests.get(url, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
                    if r.status_code != 200:
                        continue
                    text = re.sub(r"<[^>]+>", " ", r.text)
                    text = re.sub(r"\s+", " ", text)
                    m = _NEUTRAL_RE.search(text)
                    if m:
                        # date the observation to the survey quarter's last month
                        rows.append({"date": pd.Timestamp(year=y, month=3 * q, day=15),
                                     "value": float(m.group(1))})
                        break
                except Exception as exc:                            # noqa: BLE001
                    logger.debug("BoC MPS %dQ%d: %s", y, q, exc)
            q += 1
            if q == 5:
                y, q = y + 1, 1
        if len(rows) < 4:
            raise ValueError(f"only {len(rows)} rounds parsed — refusing to overwrite")
        df = pd.DataFrame(rows).sort_values("date")
        if not (0.5 <= df["value"].median() <= 6.0):
            raise ValueError("implausible medians")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / OUT, index=False)
        logger.info("BoC MPS neutral: %d quarters %s..%s; latest %.2f%%", len(df),
                    df["date"].min().date(), df["date"].max().date(), df["value"].iloc[-1])
        return True
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("BoC MPS fetch failed (%s) — keeping committed file", exc)
        return False


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_boc_mps()


if __name__ == "__main__":
    _main()
