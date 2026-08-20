"""Fetch the KOF Consensus Forecast medians relevant to the Swiss endpoint.

Access needs NO account: the KOF Time Series Database API v2 serves its public tier
anonymously with ``access_type=public`` (the login wall only guards non-public series
and the Swagger console). The consensus medians live in the public collection
``kof_consensus_forecast_median``.

What KOF asks — and does not ask. The survey's 5-year-horizon block covers CPI, real
GDP and unemployment; its INTEREST-RATE questions reach only 3 and 12 months ahead.
So KOF supplies a living long-horizon INFLATION expectation (``prices_5y``, quarterly
2015+, currently ~0.9% — 7bp from the frozen Consensus Economics 6-10y CPI), but no
rate object usable as a neutral. Whether that inflation pulse should carry the Swiss
endpoint (pi_5y + a survey-calibrated constant) or the endpoint should stay frozen is
a recorded design decision — this module only keeps the data current either way.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
_TS_URL = ("https://tsdb-api.kof.ethz.ch/v2/ts?keys={keys}&access_type=public")
SERIES = {
    "ch.kof.consensus.q_qn_prices_5y.median": ("kof_cpi_5y.csv", 0.0, 4.0),
    "ch.kof.consensus.q_qn_3minterest_12m.median": ("kof_3m_12m.csv", -2.0, 6.0),
}


def fetch_kof(out_dir: "Path | None" = None) -> bool:
    """All configured KOF medians -> committed date,value CSVs (percent). Best-effort:
    never raises, never overwrites a good committed file with an implausible parse."""
    import requests
    out = Path(out_dir) if out_dir is not None else ANCHOR_DIR
    try:
        r = requests.get(_TS_URL.format(keys=",".join(SERIES)), timeout=60,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        ok = 0
        for row in r.json():
            key = row.get("ts_key")
            if key not in SERIES:
                continue
            name, lo, hi = SERIES[key]
            s = pd.Series(row["value"], index=pd.to_datetime(row["time"])).dropna()
            if len(s) < 8 or not (lo <= s.median() <= hi):
                logger.warning("KOF %s: implausible (n=%d median=%.2f) — skipped",
                               key, len(s), s.median() if len(s) else float("nan"))
                continue
            out.mkdir(parents=True, exist_ok=True)
            s.rename("value").rename_axis("date").round(4).to_csv(out / name)
            logger.info("KOF %s: %d quarterly obs %s..%s (latest %.2f%%)",
                        key.split(".")[-2], len(s), s.index.min().date(),
                        s.index.max().date(), s.iloc[-1])
            ok += 1
        return ok == len(SERIES)
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("KOF fetch failed (%s) — keeping committed files", exc)
        return False


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fetch_kof()


if __name__ == "__main__":
    _main()
