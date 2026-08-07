"""Fetch and cache daily GSW Svensson parameters (feds200628).

Pulls the Federal Reserve Board's GSW file, keeps the six Svensson parameters
(BETA0-3, TAU1-2) plus the published fitted zero yields (SVENYNN, for the
reconstruction cross-check), and caches both the raw CSV and a cleaned parquet
under ``us/data``.

Egress note: ``www.federalreserve.gov`` is blocked by the proxy in some
environments. If a live download fails and a cached parquet exists, we fall
back to the cache with a warning; otherwise we raise with clear guidance. Run
this where egress to federalreserve.gov is open (or set ``GSW_URL`` to a
mirror) to refresh the cache.

Usage
-----
    python -m us.fetch_gsw            # use cache if present, else download
    python -m us.fetch_gsw --refresh # force a fresh download
"""
from __future__ import annotations

import argparse
import io
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import (
    DATA_DIR,
    GSW_URL,
    PARAM_COLUMNS,
    PARAMS_PARQUET_PATH,
    PUBLISHED_YIELD_PREFIX,
    RAW_CSV_PATH,
)

logger = logging.getLogger(__name__)

# Network timeout for the download, seconds.
_TIMEOUT = 60


def _download(url: str) -> str:
    """Download the GSW CSV as text. Uses requests if available, else urllib.

    Both paths honour the environment's HTTP(S) proxy settings.
    """
    try:
        import requests  # optional dependency; nice error handling

        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except ImportError:
        from urllib.request import urlopen

        with urlopen(url, timeout=_TIMEOUT) as r:  # noqa: S310 (trusted Fed URL)
            return r.read().decode("utf-8", errors="replace")


def _find_header_row(text: str) -> int:
    """Return the 0-based line index of the CSV header (the row starting 'Date').

    The feds200628 file is prefixed with several lines of description before the
    real header, so we cannot use a fixed skiprows.
    """
    for i, line in enumerate(text.splitlines()):
        first = line.split(",", 1)[0].strip().strip('"').lower()
        if first == "date":
            return i
    raise ValueError("Could not locate the 'Date' header row in the GSW file.")


def parse_gsw_csv(text: str) -> pd.DataFrame:
    """Parse GSW CSV text into a date-indexed DataFrame of the columns we keep.

    Keeps the six Svensson parameters and any published ``SVENYNN`` yield
    columns (for the cross-check). Values are coerced to float; the index is a
    sorted DatetimeIndex.
    """
    skip = _find_header_row(text)
    df = pd.read_csv(io.StringIO(text), skiprows=skip)

    # Normalise the date column (named 'Date') to the index.
    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if date_col is None:
        raise ValueError("Parsed GSW file has no 'Date' column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"

    keep = [c for c in PARAM_COLUMNS if c in df.columns]
    missing = [c for c in PARAM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"GSW file is missing expected parameter columns {missing}; "
            f"columns present: {list(df.columns)[:12]}..."
        )
    sveny = [
        c
        for c in df.columns
        if c.startswith(PUBLISHED_YIELD_PREFIX)
        and c[len(PUBLISHED_YIELD_PREFIX):].isdigit()
    ]
    out = df[keep + sveny].apply(pd.to_numeric, errors="coerce")
    # Drop rows with no parameters at all (early history / blank tails).
    return out.dropna(how="all", subset=keep)


def load_gsw_params(
    refresh: bool = False,
    url: Optional[str] = None,
) -> pd.DataFrame:
    """Load GSW Svensson parameters, downloading + caching as needed.

    Parameters
    ----------
    refresh
        If True, always attempt a fresh download.
    url
        Override the source URL (defaults to ``config.GSW_URL``).

    Returns a date-indexed DataFrame with the parameter columns (and any
    ``SVENYNN`` columns present).
    """
    url = url or GSW_URL
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    need_download = refresh or not PARAMS_PARQUET_PATH.exists()
    if need_download:
        try:
            logger.info("Downloading GSW parameters from %s", url)
            text = _download(url)
            df = parse_gsw_csv(text)
            RAW_CSV_PATH.write_text(text)
            df.to_parquet(PARAMS_PARQUET_PATH)
            logger.info(
                "Cached %d rows (%s .. %s) to %s",
                len(df),
                df.index.min().date(),
                df.index.max().date(),
                PARAMS_PARQUET_PATH,
            )
            return df
        except Exception as exc:  # noqa: BLE001 - want a clear fallback message
            if PARAMS_PARQUET_PATH.exists():
                logger.warning(
                    "GSW download failed (%s); falling back to cached %s. "
                    "Refresh where egress to the Fed Board is open.",
                    exc,
                    PARAMS_PARQUET_PATH,
                )
            else:
                raise RuntimeError(
                    f"Could not download GSW parameters from {url} and no cache "
                    f"exists at {PARAMS_PARQUET_PATH}. This environment may block "
                    f"federalreserve.gov; run the fetch where egress is open, or "
                    f"set GSW_URL to a mirror."
                ) from exc

    return pd.read_parquet(PARAMS_PARQUET_PATH)


def build_yield_panel(refresh: bool = False, grid=None) -> pd.DataFrame:
    """Convenience: load params and reconstruct the engine-ready yield panel.

    Returns a date x maturity DataFrame of zero yields in DECIMAL on ``grid``
    (defaults to the common 8-point grid) — the ``/model`` data contract.
    """
    from .config import COMMON_GRID
    from .reconstruct import reconstruct_grid

    params = load_gsw_params(refresh=refresh)
    return reconstruct_grid(params, COMMON_GRID if grid is None else grid)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Fetch/cache GSW Svensson params")
    parser.add_argument("--refresh", action="store_true", help="force fresh download")
    args = parser.parse_args()

    params = load_gsw_params(refresh=args.refresh)
    print(f"Loaded {len(params)} daily observations")
    print(f"  date range : {params.index.min().date()} .. {params.index.max().date()}")
    print(f"  columns    : {list(params.columns)}")

    # Report the reconstruction cross-check if published yields are available.
    from .reconstruct import cross_check_against_published

    check = cross_check_against_published(params)
    if not check.empty:
        worst = check.sort_values("max_abs_diff_bp", ascending=False).head(5)
        print("Reconstruction vs published SVENY (worst 5 maturities, bp):")
        print(worst.to_string(index=False))
        if not bool(check["within_tol"].all()):
            print("WARNING: some maturities exceed the cross-check tolerance.")
    else:
        print("(No published SVENY columns available for cross-check.)")


if __name__ == "__main__":
    _main()
