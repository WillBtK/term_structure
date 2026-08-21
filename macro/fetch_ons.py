"""ONS time-series access, for UK macro series FRED does not carry or keeps current.

The ONS retired ``api.ons.gov.uk`` on 25/11/2024. The website itself still serves every
published time series as JSON by appending ``/data`` to the series page, which is what
this module uses::

    https://www.ons.gov.uk/<topic-path>/timeseries/<cdid>/<dataset>/data

The payload carries ``months``, ``quarters`` and ``years`` arrays of
``{date, value, ...}``, so the publication frequency can be chosen at read time rather
than resampled afterwards.
"""
from __future__ import annotations

import io
import json
import logging
import subprocess
import time

import pandas as pd

logger = logging.getLogger(__name__)

ONS_BASE = "https://www.ons.gov.uk"
_TIMEOUT = 60

# Topic path per dataset. The website URL embeds where the series is published, not just
# its identifier, so the path has to be known per source dataset.
DATASET_PATHS = {
    "diop": "economy/economicoutputandproductivity/output",
    "ukea": "economy/grossdomesticproductgdp",
    "pusf": "economy/governmentpublicsectorandtaxes/publicsectorfinance",
}

_QUARTER_MONTH = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}


def _get(url: str, retries: int = 4) -> dict:
    """Fetch JSON via curl — the same path the FRED loader uses, for the same reason
    (urllib is less reliable than curl through the egress proxy for these hosts)."""
    last = None
    for attempt in range(retries):
        try:
            out = subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", str(_TIMEOUT),
                 "-A", "Mozilla/5.0", url],
                capture_output=True, timeout=_TIMEOUT + 10)
            if out.returncode != 0:
                raise RuntimeError(f"curl {out.returncode}: {out.stderr[:160]!r}")
            return json.load(io.BytesIO(out.stdout))
        except Exception as exc:                                    # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ONS fetch failed for {url}: {last}")


def _parse_date(row: dict) -> pd.Timestamp | None:
    """ONS dates are '1948 JAN', '1948 Q1' or '1948'. Map each to its period start."""
    date = str(row.get("date", "")).strip()
    parts = date.split()
    if len(parts) == 1 and parts[0].isdigit():
        return pd.Timestamp(f"{parts[0]}-01-01")
    if len(parts) != 2:
        return None
    year, tail = parts
    if tail.upper() in _QUARTER_MONTH:
        return pd.Timestamp(f"{year}-{_QUARTER_MONTH[tail.upper()]}-01")
    try:
        return pd.Timestamp(f"{year} {tail} 01")
    except ValueError:
        return None


def ons_series(cdid: str, dataset: str, freq: str = "quarters") -> pd.Series:
    """One ONS time series as a float Series indexed by period start.

    ``freq`` selects which published frequency to read — ``months``, ``quarters`` or
    ``years``. Reading the published quarterly figures beats resampling the monthly ones
    ourselves: it is the ONS's own aggregation, so it needs no assumption about how to
    collapse the months.
    """
    path = DATASET_PATHS.get(dataset.lower())
    if path is None:
        raise KeyError(f"unknown ONS dataset {dataset!r}; add it to DATASET_PATHS")
    url = f"{ONS_BASE}/{path}/timeseries/{cdid.lower()}/{dataset.lower()}/data"
    payload = _get(url)
    rows = payload.get(freq) or []
    if not rows:
        raise RuntimeError(f"ONS {cdid}/{dataset}: no '{freq}' observations")
    out = {}
    for r in rows:
        d = _parse_date(r)
        try:
            v = float(r.get("value"))
        except (TypeError, ValueError):
            continue
        if d is not None:
            out[d] = v
    s = pd.Series(out).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    desc = payload.get("description") or {}
    logger.info("ONS %s (%s): %d %s obs, %s..%s", cdid.upper(),
                desc.get("title", ""), len(s), freq,
                s.index.min().date(), s.index.max().date())
    return s.rename(cdid.upper())
