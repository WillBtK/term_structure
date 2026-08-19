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
  - ECB SPF longer-term     : the same survey's longer-term INTEREST-RATE assumption
    interest rate             (quarterly, 2025+). The only public, machine-readable
                              long-horizon euro-area short-rate expectation there is,
                              and therefore the carry-forward for the frozen Consensus
                              rung rather than merely another fallback.
                              -> germany/data/anchors/ecb_spf_rate_lt.csv

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
_ECB_BASE = "https://data-api.ecb.europa.eu/service/data/SPF"
ECB_SPF_URL = os.environ.get("ECB_SPF_URL", f"{_ECB_BASE}/Q.U2.HICP.POINT.LT.Q.AVG?format=csvdata")
# The SPF also publishes forecasters' longer-term INTEREST-RATE assumption. It is the
# only public, machine-readable long-horizon euro-area short-rate expectation there is,
# which makes it the carry-forward for the (proprietary, frozen) Consensus rung. The
# variable is a 3-month market rate, not the policy rate — its near-term counterpart
# (Q.U2.ASSU.IR.P3M.Q.AVG) tracks 3m EURIBOR — so it is the same concept as the
# Consensus 3m-interbank series and the two splice like for like. Horizons differ (SPF
# is 4-5 calendar years ahead, Consensus 6-10), which the level-bridge absorbs.
ECB_SPF_RATE_URL = os.environ.get("ECB_SPF_RATE_URL",
                                  f"{_ECB_BASE}/Q.U2.ASSU.IR.LT.Q.AVG?format=csvdata")


def fetch_ferreira_ea() -> bool:
    """Euro-area longer-run real neutral rate -> ferreira_rstar_ea.csv."""
    return fetch_ferreira_economy("EA", ("euro area", "euro", "ea"),
                                  "ferreira_rstar_ea.csv", out_dir=ANCHOR_DIR)


def _fetch_spf(url: str, out_name: str, what: str) -> bool:
    """Pull one SPF aggregate series -> a date,value anchor CSV. Never raises."""
    try:
        r = requests.get(url, headers=_UA, timeout=_TIMEOUT); r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.content.decode("utf-8", "replace")))
        if "TIME_PERIOD" not in df.columns or "OBS_VALUE" not in df.columns:
            logger.warning("fetch_neutral[DE]: unexpected SPF columns %s", list(df.columns)[:8])
            return False
        # TIME_PERIOD is a survey ROUND ("2026-Q3"); date it to the quarter end, which is
        # when the round is published — the anchor treats these as release-dated reads.
        idx = pd.PeriodIndex(df["TIME_PERIOD"].astype(str), freq="Q").to_timestamp(how="end").normalize()
        s = pd.Series(pd.to_numeric(df["OBS_VALUE"], errors="coerce").to_numpy(), index=idx)
        logger.info("fetch_neutral[DE]: ECB SPF %s — %d rounds %s..%s (latest %.3f)",
                    what, len(s), s.index.min().date(), s.index.max().date(), float(s.iloc[-1]))
        return _write_if_ok(out_name, s, out_dir=ANCHOR_DIR)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("fetch_neutral[DE]: ECB SPF %s fetch failed (%s)", what, exc)
        return False


def fetch_ecb_spf() -> bool:
    """Longer-term HICP expectation -> ecb_spf_infl_lt.csv (the inflation extender)."""
    return _fetch_spf(ECB_SPF_URL, "ecb_spf_infl_lt.csv", "longer-term HICP")


def fetch_ecb_spf_rate() -> bool:
    """Longer-term interest-rate assumption -> ecb_spf_rate_lt.csv (the short-rate
    extender: the public carry-forward for the frozen Consensus rung)."""
    return _fetch_spf(ECB_SPF_RATE_URL, "ecb_spf_rate_lt.csv", "longer-term interest rate")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ok_r = fetch_ferreira_ea()
    ok_s = fetch_ecb_spf()
    ok_i = fetch_ecb_spf_rate()
    print(f"ferreira_rstar_ea: {'ok' if ok_r else 'kept committed file'}")
    print(f"ecb_spf_infl_lt  : {'ok' if ok_s else 'kept committed file'}")
    print(f"ecb_spf_rate_lt  : {'ok' if ok_i else 'kept committed file'}")


if __name__ == "__main__":
    _main()
