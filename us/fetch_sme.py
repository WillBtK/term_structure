"""NY Fed Survey of Market Expectations (SME): the high-frequency US rate anchor.

Why this exists. The US nominal anchor is the SPF's BILL10 -- the 10-year average
T-bill forecast -- and the SPF asks that question in its FIRST-QUARTER round only
(verified against the Philadelphia Fed's own file: Q1 filled every year, Q2-Q4 never).
So between February rounds the anchor could not move at all, while 2026 brought a war,
a new Fed chair and a 40bp+ rise in ten-year yields.

The SME is fielded before every FOMC meeting (eight rounds a year) and asks the SAME
object directly -- "expectation for the average federal funds rate over the next 10
years" -- alongside the modal policy path for the next few years, all from one panel
in one round. That lets the 2-10y forward the anchor is built on be computed exactly,
round by round, with no model:

    fwd_2_10 = (10 * avg10  -  integral of the modal path over years 0-2) / 8

Data: the per-round tidy ``<mon>-<year>-data.xlsx`` files the NY Fed links from the
survey's landing page (2023 onward, when the survey took its current form). Combined
panel (dealers + market participants). Values are converted from decimals to percent.

Central estimate: the INTERQUARTILE MEAN, (p25 + p50 + p75)/3, not the median. The
SME publishes no cross-respondent mean for these questions (only quartiles; its
averaged probability distributions stop at next year-end), and the median is a step
function: it sat at 3.13 for seven rounds (Jul-25 to Apr-26) while both quartiles
moved. Averaging the three quartiles is the closest published object to a mean. The fed funds rate sits above the 3-month bill on average; the
basis is calibrated against the SPF in ``us.anchor`` rather than assumed here.

Best-effort like every anchor fetcher: never raises, never overwrites a good committed
file with a worse one.
"""
from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
OUT = ANCHOR_DIR / "sme_ffr.csv"
LANDING = "https://www.newyorkfed.org/markets/market-intelligence/survey-of-market-expectations"
HOST = "https://www.newyorkfed.org"
_LINK = re.compile(r'href="([^"]*?-data\.xlsx)"', re.I)
_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url: str) -> bytes:
    import requests
    r = requests.get(url, headers=_UA, timeout=90)
    r.raise_for_status()
    return r.content


def data_links(html: str) -> list:
    """Absolute URLs of every per-round data workbook on the landing page."""
    out = []
    for h in _LINK.findall(html):
        out.append(h if h.startswith("http") else HOST + h)
    return sorted(set(out))


def _horizon_date(tag: str, release: pd.Timestamp):
    """The date a path-of-modes entry refers to, from its value tag.

    ``fftr_pathofmodes_20260916`` -> that meeting; ``..._2027q3`` -> mid-quarter;
    ``..._2029`` -> mid-year. None for anything else."""
    m = re.search(r"pathofmodes_(\d{8})$", tag)
    if m:
        return pd.Timestamp(m.group(1))
    m = re.search(r"pathofmodes_(\d{4})q([1-4])$", tag)
    if m:
        return pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)) - 1, 15)
    m = re.search(r"pathofmodes_(\d{4})$", tag)
    if m:
        return pd.Timestamp(int(m.group(1)), 7, 1)
    return None


QUARTILES = ("pctl25", "pctl50", "pctl75")


def _central(d: pd.DataFrame) -> pd.Series:
    """value_tag -> interquartile mean (percent-decimal); the median where a quartile
    is missing."""
    w = d.pivot_table(index="value_tag", columns="aggregation",
                      values="aggregation_value", aggfunc="first")
    w.index = w.index.astype(str)
    full = w.reindex(columns=list(QUARTILES)).notna().all(axis=1)
    out = w.reindex(columns=list(QUARTILES)).mean(axis=1)
    if "pctl50" in w:
        out[~full] = w.loc[~full, "pctl50"]
    return out.dropna()


def parse_round(blob: bytes) -> dict | None:
    """One round -> {date, avg10, longer_run, near2, fwd_2_10} in percent, or None.

    ``near2`` is the modal path's average over the two years from the survey date,
    integrating the stepwise path (each entry holds until the next); ``fwd_2_10`` is
    the 2-10y forward implied by the round's own 10-year average."""
    df = pd.read_excel(io.BytesIO(blob))
    need = {"panel_type", "subject", "aggregation", "value_tag", "aggregation_value",
            "survey_due_date"}
    if not need <= set(df.columns):
        return None
    d = df[(df["subject"] == "fed_funds_target_range")
           & df["aggregation"].isin(QUARTILES)
           & (df["panel_type"].astype(str).str.lower() == "combined")]
    if d.empty:
        return None
    central = _central(d)
    avg10 = central[central.index.str.endswith("_10yravg")]
    lr = central[central.index.str.endswith("_longerrun") & central.index.str.contains("modalpe")]
    if avg10.empty:
        return None
    asof = pd.Timestamp(df["survey_due_date"].dropna().iloc[0])
    path = []
    for t, v in central.items():
        if "pathofmodes" in t:
            hd = _horizon_date(t, asof)
            if hd is not None and pd.notna(v):
                path.append((hd, float(v)))
    path.sort()
    near2 = None
    if path:
        # stepwise: the rate in force at each day is the latest entry dated at or
        # before it (the first entry holds from the survey date)
        grid = pd.date_range(asof, asof + pd.DateOffset(years=2), freq="D")[:-1]
        dates = np.array([p[0] for p in path], dtype="datetime64[ns]")
        vals = np.array([p[1] for p in path])
        idx = np.searchsorted(dates, grid.values, side="right") - 1
        rate = np.where(idx < 0, vals[0], vals[np.clip(idx, 0, None)])
        if path[-1][0] >= grid[-1] - pd.Timedelta(days=200):    # path reaches far enough
            near2 = float(rate.mean())
    a10 = float(avg10.iloc[0]) * 100.0
    out = {"date": asof.normalize(), "avg10": round(a10, 4),
           "longer_run": round(float(lr.iloc[0]) * 100.0, 4) if not lr.empty else np.nan,
           "near2": round(near2 * 100.0, 4) if near2 is not None else np.nan}
    out["fwd_2_10"] = (round((10.0 * a10 - 2.0 * out["near2"]) / 8.0, 4)
                       if not np.isnan(out["near2"]) else np.nan)
    return out


def fetch(out: Path = OUT) -> pd.DataFrame | None:
    """Every round on the landing page -> committed CSV, merged with the rounds already
    committed (the page can drop old rounds; a re-read round replaces the stored one).
    Keeps the committed file untouched if the fetch fails."""
    try:
        links = data_links(_get(LANDING).decode("utf-8", "ignore"))
        rows = []
        for u in links:
            try:
                r = parse_round(_get(u))
            except Exception as exc:                                # noqa: BLE001
                logger.warning("SME round %s unreadable: %s", u.rsplit("/", 1)[-1], exc)
                continue
            if r:
                rows.append(r)
            else:
                logger.warning("SME round %s lacks the 10-year average question",
                               u.rsplit("/", 1)[-1])
        if not rows:
            raise RuntimeError(f"no usable rounds among {len(links)} links")
        df = pd.DataFrame(rows)
        if out.exists():
            have = pd.read_csv(out, parse_dates=["date"])
            df = pd.concat([have, df], ignore_index=True)
        df = df.sort_values("date", kind="mergesort").drop_duplicates("date", keep="last")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, date_format="%Y-%m-%d")
        last = df.iloc[-1]
        logger.info("SME: %d rounds %s..%s; latest 10y avg %.2f%%, 2-10y fwd %.2f%%",
                    len(df), df["date"].min().date(), df["date"].max().date(),
                    last["avg10"], last["fwd_2_10"])
        return df
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("SME fetch failed (%s) -- keeping committed file", exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    d = fetch()
    if d is not None:
        print(d.to_string(index=False))
