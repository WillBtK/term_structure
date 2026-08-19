"""Refresh the PUBLIC neutral-anchor inputs for Germany, so germany.neutral_anchor's
lower rungs self-maintain after the (proprietary, one-off) Consensus snapshot lapses:

  - Ferreira r* (euro area) : Fed FEDS Note "Real-Time Global Longer-Run Neutral Rates".
                              One file covers every economy — the ``EA`` column sits
                              beside the ``JA`` one Japan already uses, so this costs a
                              column rather than a new data source.
                              -> germany/data/anchors/ferreira_rstar_ea.csv
  - ECB SPF longer-term     : ECB Data Portal, Survey of Professional Forecasters,
                              average of point forecasts for HICP inflation at the
                              longer-term horizon (quarterly, 1999+).
                              -> germany/data/anchors/ecb_spf_infl_lt.csv

This is the half of the anchor that CAN be maintained. The Consensus CSVs beside them
are frozen; when the Consensus short-rate series passes its staleness window the anchor
falls through to ``r* + pi^e`` built from these two, and keeps tracking with no manual
step. Best-effort: a failed fetch keeps the committed file rather than overwriting it.

    python -m germany.fetch_neutral
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import pandas as pd
import requests

from japan.fetch_neutral import _UA, _TIMEOUT, _write_if_ok, fetch_ferreira_economy

logger = logging.getLogger(__name__)
ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"

# Key: FREQ.REF_AREA.FCT_TOPIC.FCT_BREAKDOWN.FCT_HORIZON.SURVEY_FREQ.FCT_SOURCE.
# FCT_SOURCE=AVG is the average across forecasters (individuals are 001, 002, ...);
# FCT_HORIZON=LT is the longer-term round (4-5 calendar years ahead).
ECB_SPF_URL = os.environ.get(
    "ECB_SPF_URL",
    "https://data-api.ecb.europa.eu/service/data/SPF/Q.U2.HICP.POINT.LT.Q.AVG?format=csvdata")


def fetch_ferreira_ea() -> bool:
    """Euro-area longer-run real neutral rate -> ferreira_rstar_ea.csv."""
    return fetch_ferreira_economy("EA", ("euro area", "euro", "ea"),
                                  "ferreira_rstar_ea.csv", out_dir=ANCHOR_DIR)


def fetch_ecb_spf() -> bool:
    """ECB SPF longer-term HICP expectation -> ecb_spf_infl_lt.csv. Never raises."""
    try:
        r = requests.get(ECB_SPF_URL, headers=_UA, timeout=_TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", "replace")))
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            logger.warning("fetch_neutral[DE]: unexpected SPF columns %s", list(df.columns)[:8])
            return False
        # TIME_PERIOD is a survey ROUND ("2026-Q3"); date it to the quarter end, which is
        # when the round is published — the anchor treats these as release-dated reads.
        idx = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q").to_timestamp(how="end").normalize()
        s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(), index=idx)
        logger.info("fetch_neutral[DE]: ECB SPF longer-term %d rounds %s..%s",
                    len(s), s.index.min().date(), s.index.max().date())
        return _write_if_ok("ecb_spf_infl_lt.csv", s, out_dir=ANCHOR_DIR)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("fetch_neutral[DE]: ECB SPF fetch failed (%s)", exc)
        return False


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ok_r = fetch_ferreira_ea()
    ok_s = fetch_ecb_spf()
    print(f"ferreira_rstar_ea: {'ok' if ok_r else 'kept committed file'}")
    print(f"ecb_spf_infl_lt  : {'ok' if ok_s else 'kept committed file'}")


if __name__ == "__main__":
    _main()
