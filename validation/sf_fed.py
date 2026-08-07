"""Fetch and normalize the SF Fed "Treasury Yield Premiums" benchmark series.

Produces the canonical tidy frame consumed by ``compare_afns`` (columns:
``date, maturity, term_premium, expected_rate, fitted_yield`` in DECIMAL).

Schema verified against the live file (``FRBSF_Term_Model_Data.xlsx``,
2026-08). The workbook has one sheet per quantity; we use the model **Fit**
sheets:

    "Fit Term Premium"      columns FTERMP01YR .. FTERMP10YR  -> term_premium
    "Fit Avg Expected Rate" columns FEXPTR01YR .. FEXPTR10YR  -> expected_rate
    "Fit Treasury Yields"   columns FYIELD01YR .. FYIELD10YR  -> fitted_yield

Values are already in DECIMAL (e.g. 0.055 = 5.5%), daily from 1998-01-02, at
integer maturities 1..10 years. Overlap with our COMMON_GRID is {1,2,3,5,7,10}y.

The SF Fed series is itself a plain-AFNS (Christensen-Diebold-Rudebusch) fit to
the same GSW data, which is exactly why it is the right baseline: our numbers
should track it closely.

A generic single-table path (``ColumnSpec`` + ``normalize``) is retained for
other/future layouts, but ``load_sf_fed_series`` uses the concrete multi-sheet
parser by default.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from .config import DATA_DIR, SF_FED_PARQUET_PATH, SF_FED_URL

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # the model-data workbook is ~4 MB

# Default source: the full model-output workbook (not the small web-chart file).
FRBSF_MODEL_DATA_URL = (
    "https://www.frbsf.org/wp-content/uploads/FRBSF_Term_Model_Data.xlsx"
)

# Fit (model output) sheets -> (sheet name, column prefix) per canonical quantity.
_FIT_SHEETS = {
    "term_premium": ("Fit Term Premium", "FTERMP"),
    "expected_rate": ("Fit Avg Expected Rate", "FEXPTR"),
    "fitted_yield": ("Fit Treasury Yields", "FYIELD"),
}

_CANONICAL = ["date", "maturity", "term_premium", "expected_rate", "fitted_yield"]

# Matches e.g. FTERMP01YR -> maturity 1, FYIELD10YR -> maturity 10.
_COL_RE = re.compile(r"^(?P<prefix>[A-Z]+?)(?P<yy>\d{1,2})YR$")


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


def _melt_fit_sheet(df: pd.DataFrame, prefix: str, value_name: str) -> pd.DataFrame:
    """Melt a wide Fit sheet (DATE + <prefix>NNYR columns) to long tidy form.

    Values are already decimal; returns columns ``date, maturity, <value_name>``.
    """
    date_col = next((c for c in df.columns if str(c).strip().upper() == "DATE"), None)
    if date_col is None:
        raise ValueError(f"Fit sheet for {value_name!r} has no DATE column.")
    dates = pd.to_datetime(df[date_col], errors="coerce")

    parts = []
    for col in df.columns:
        m = _COL_RE.match(str(col))
        if not m or m.group("prefix") != prefix:
            continue
        maturity = float(int(m.group("yy")))
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "maturity": maturity,
                    value_name: pd.to_numeric(df[col], errors="coerce"),
                }
            )
        )
    if not parts:
        raise ValueError(f"No {prefix}NNYR columns found for {value_name!r}.")
    return pd.concat(parts, ignore_index=True)


def parse_frbsf_model_data(content: bytes) -> pd.DataFrame:
    """Parse FRBSF_Term_Model_Data.xlsx bytes into the canonical tidy frame."""
    xl = pd.ExcelFile(io.BytesIO(content))
    merged: Optional[pd.DataFrame] = None
    for quantity, (sheet, prefix) in _FIT_SHEETS.items():
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"expected sheet {sheet!r} not found; sheets: {xl.sheet_names}"
            )
        part = _melt_fit_sheet(pd.read_excel(xl, sheet_name=sheet), prefix, quantity)
        merged = part if merged is None else merged.merge(
            part, on=["date", "maturity"], how="outer"
        )
    out = merged.dropna(subset=["date"]).sort_values(["date", "maturity"])
    return out.reset_index(drop=True)[_CANONICAL]


def load_sf_fed_series(refresh: bool = False, url: Optional[str] = None) -> pd.DataFrame:
    """Load the SF Fed benchmark as the canonical tidy frame.

    Uses the cached parquet when present unless ``refresh``. On a failed
    download, falls back to cache if available, else raises with guidance.
    """
    url = url or SF_FED_URL or FRBSF_MODEL_DATA_URL
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and SF_FED_PARQUET_PATH.exists():
        return pd.read_parquet(SF_FED_PARQUET_PATH)

    try:
        content = _download_bytes(url)
        (DATA_DIR / "FRBSF_Term_Model_Data.xlsx").write_bytes(content)
        series = parse_frbsf_model_data(content)
        series.to_parquet(SF_FED_PARQUET_PATH)
        logger.info(
            "Cached SF Fed series: %d rows (%s .. %s), maturities %s",
            len(series),
            series["date"].min().date(),
            series["date"].max().date(),
            sorted(series["maturity"].unique().tolist()),
        )
        return series
    except Exception as exc:  # noqa: BLE001
        if SF_FED_PARQUET_PATH.exists():
            logger.warning("SF Fed download failed (%s); using cache.", exc)
            return pd.read_parquet(SF_FED_PARQUET_PATH)
        raise RuntimeError(
            f"Could not download the SF Fed series from {url!r} and no cache "
            f"exists at {SF_FED_PARQUET_PATH}. Ensure www.frbsf.org is reachable "
            f"(egress allowlist) or point SF_FED_URL at the file."
        ) from exc


# ---------------------------------------------------------------------------
# Generic single-table path — retained for alternative layouts. Not used by the
# concrete FRBSF loader above, but unit-tested and available if the published
# format ever changes to a single wide table.
# ---------------------------------------------------------------------------
@dataclass
class ColumnSpec:
    """Map a single wide table's columns onto the canonical schema.

    For each canonical quantity, provide ``{maturity_years: source_column}``.
    ``values_in_percent`` controls the ÷100 conversion to decimal.
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


def normalize(raw: pd.DataFrame, spec: ColumnSpec) -> pd.DataFrame:
    """Reshape a single wide table into the canonical tidy frame."""
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
        frames.append(pd.DataFrame({"date": dates, "maturity": float(mat), **cols}))

    out = pd.concat(frames, ignore_index=True).dropna(subset=["date"])
    for q in ("term_premium", "expected_rate", "fitted_yield"):
        if q not in out.columns:
            out[q] = float("nan")
    return out[_CANONICAL]
