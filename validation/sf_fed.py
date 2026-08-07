"""Fetch and normalize the SF Fed "Treasury Yield Premiums" benchmark series.

Produces the canonical tidy frame consumed by ``compare_afns`` (columns:
``date, maturity, term_premium, expected_rate, fitted_yield`` in DECIMAL).

⚠️ SCHEMA NOT YET VERIFIED AGAINST THE LIVE FILE.
``frbsf.org`` is egress-blocked in some environments, so the exact download URL
and column layout of the published file could not be confirmed here. The parser
below is driven by an explicit ``ColumnSpec`` so it can be adapted to the real
file *without touching the comparison code*: set ``SF_FED_URL`` and, if needed,
pass a ``ColumnSpec`` matching the file's actual headers. Verify against the
real download before trusting validation numbers (see top-level README open
items).

The SF Fed series is itself a plain-AFNS (Christensen-Diebold-Rudebusch) fit to
the same GSW data, which is exactly why it is the right baseline: our numbers
should track it closely.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import (
    DATA_DIR,
    SF_FED_PARQUET_PATH,
    SF_FED_URL,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 60


@dataclass
class ColumnSpec:
    """How to map the published file's columns onto the canonical schema.

    The published file is wide (one set of columns per maturity). For each
    maturity we need the term premium, the expected/risk-neutral rate, and the
    fitted yield. Provide, per canonical quantity, a mapping ``{maturity_years:
    source_column_name}``.

    ``date_column`` is the header of the date column; ``values_in_percent``
    controls the ÷100 conversion to decimal (the SF Fed file is expected to be
    in percent).

    Fill this in once the real headers are known. The defaults are intentionally
    empty so mis-parsing fails loudly rather than silently.
    """

    date_column: str = "Date"
    values_in_percent: bool = True
    term_premium: Dict[float, str] = field(default_factory=dict)
    expected_rate: Dict[float, str] = field(default_factory=dict)
    fitted_yield: Dict[float, str] = field(default_factory=dict)

    def maturities(self):
        return sorted(
            set(self.term_premium) | set(self.expected_rate) | set(self.fitted_yield)
        )


def _download_bytes(url: str) -> bytes:
    try:
        import requests

        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.content
    except ImportError:
        from urllib.request import urlopen

        with urlopen(url, timeout=_TIMEOUT) as r:  # noqa: S310
            return r.read()


def _read_table(content: bytes, url: str) -> pd.DataFrame:
    """Read the raw file into a DataFrame (Excel or CSV, by extension)."""
    if url.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


def normalize(raw: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    """Reshape the wide published table into the canonical tidy frame."""
    if spec.date_column not in raw.columns:
        raise ValueError(
            f"date column {spec.date_column!r} not found; columns present: "
            f"{list(raw.columns)[:15]}"
        )
    if not spec.maturities():
        raise ValueError(
            "ColumnSpec has no maturity mappings — fill in term_premium / "
            "expected_rate / fitted_yield against the real file's headers."
        )
    scale = 100.0 if spec.values_in_percent else 1.0
    dates = pd.to_datetime(raw[spec.date_column], errors="coerce")

    frames = []
    for mat in spec.maturities():
        cols = {}
        for q, mapping in (
            ("term_premium", spec.term_premium),
            ("expected_rate", spec.expected_rate),
            ("fitted_yield", spec.fitted_yield),
        ):
            src = mapping.get(mat)
            if src is not None:
                if src not in raw.columns:
                    raise ValueError(f"column {src!r} (for {q} {mat}y) not in file")
                cols[q] = pd.to_numeric(raw[src], errors="coerce") / scale
        part = pd.DataFrame({"date": dates, "maturity": float(mat), **cols})
        frames.append(part)

    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    # Ensure all canonical quantity columns exist (NaN where unmapped).
    for q in ("term_premium", "expected_rate", "fitted_yield"):
        if q not in out.columns:
            out[q] = float("nan")
    return out[["date", "maturity", "term_premium", "expected_rate", "fitted_yield"]]


def load_sf_fed_series(
    spec: Optional[ColumnSpec] = None,
    refresh: bool = False,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """Load the SF Fed series as the canonical tidy frame.

    Uses the cached parquet when present unless ``refresh``. On a failed
    download, falls back to cache if available, else raises with guidance.
    A ``ColumnSpec`` is required to (re)parse a freshly downloaded raw file.
    """
    url = url or SF_FED_URL
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and SF_FED_PARQUET_PATH.exists():
        return pd.read_parquet(SF_FED_PARQUET_PATH)

    if not url:
        raise RuntimeError(
            "SF_FED_URL is not set and no cached series exists. Set SF_FED_URL "
            "to the SF Fed Treasury Yield Premiums download link (frbsf.org may "
            "be egress-blocked here) and pass a ColumnSpec matching its headers."
        )
    if spec is None:
        raise ValueError(
            "A ColumnSpec is required to parse a freshly downloaded SF Fed file."
        )
    try:
        content = _download_bytes(url)
        raw = _read_table(content, url)
        (DATA_DIR / "sf_fed_raw.bin").write_bytes(content)
        series = normalize(raw, spec)
        series.to_parquet(SF_FED_PARQUET_PATH)
        logger.info("Cached SF Fed series: %d rows to %s", len(series), SF_FED_PARQUET_PATH)
        return series
    except Exception as exc:  # noqa: BLE001
        if SF_FED_PARQUET_PATH.exists():
            logger.warning("SF Fed download failed (%s); using cache.", exc)
            return pd.read_parquet(SF_FED_PARQUET_PATH)
        raise RuntimeError(
            f"Could not download the SF Fed series from {url!r} and no cache "
            f"exists at {SF_FED_PARQUET_PATH}. frbsf.org may be egress-blocked; "
            f"run where egress is open or point SF_FED_URL at a mirror."
        ) from exc
