"""Fetch + cache MoF JGB spot yields and build the (ragged) zero-yield panel.

The MoF publishes the full history as a single Shift-JIS CSV (`jgbcm_all.csv`)
with Japanese-era dates and yields (percent) at maturities 1-10y + 15/20/25/30/40y;
missing values are '-'. Yields are published directly, so there is no Svensson
reconstruction — we just parse, convert dates and units, and subset to the grid.
The long end is genuinely ragged (10y from 1986-07, 20y 1986-12, 30y 1999-09),
handled downstream by the EKF's per-date maturity masking.
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd

from japan import config as cfg

logger = logging.getLogger(__name__)


def _download(timeout: int = 90) -> str:
    """Fetch jgbcm_all.csv and decode Shift-JIS. Falls back to the raw cache."""
    try:
        import requests
        content = requests.get(cfg.MOF_JGB_URL, timeout=timeout).content
        cfg.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg.RAW_CSV_PATH.write_bytes(content)
        return content.decode("shift_jis", errors="replace")
    except Exception as exc:
        if cfg.RAW_CSV_PATH.exists():
            logger.warning("download failed (%s); using cached raw CSV", exc)
            return cfg.RAW_CSV_PATH.read_bytes().decode("shift_jis", errors="replace")
        raise


def parse_jgb_csv(csv_text: str) -> pd.DataFrame:
    """Parse the MoF CSV into a date x tenor(years) frame of yields in DECIMAL.

    Row 0 is a title, row 1 the header (基準日,1年,...); data rows follow. '-' and
    blanks become NaN.
    """
    rows, index = [], []
    for line in io.StringIO(csv_text):
        parts = line.rstrip("\r\n").split(",")
        if not parts or not parts[0]:
            continue
        d0 = parts[0].strip()
        if not d0 or d0[0] not in cfg._ERA_BASE:   # skip title/header/blank lines
            continue
        try:
            dt = cfg.parse_jp_date(d0)
        except Exception:
            continue
        vals = []
        for v in parts[1:1 + len(cfg.MOF_TENORS)]:
            v = v.strip()
            vals.append(np.nan if v in ("", "-", "*", "***") else float(v))
        if len(vals) < len(cfg.MOF_TENORS):
            vals += [np.nan] * (len(cfg.MOF_TENORS) - len(vals))
        index.append(dt)
        rows.append(vals)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=[float(t) for t in cfg.MOF_TENORS])
    return (df.sort_index() / 100.0)   # percent -> decimal


def load_yields(refresh: bool = True) -> pd.DataFrame:
    """Full MoF JGB yield frame (all tenors, decimal). Cached to parquet."""
    if not refresh and cfg.YIELDS_PARQUET_PATH.exists():
        return pd.read_parquet(cfg.YIELDS_PARQUET_PATH)
    df = parse_jgb_csv(_download())
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.YIELDS_PARQUET_PATH)
    logger.info("MoF JGB yields: %d rows %s..%s", len(df),
                df.index.min().date(), df.index.max().date())
    return df


def build_yield_panel(grid: np.ndarray = cfg.COMMON_GRID, freq: str = "M",
                      start: str = cfg.SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Yields on the estimation ``grid`` (decimal), monthly by default, from
    ``start``. Ragged: a tenor is NaN before its first-available date.
    """
    df = load_yields(refresh=refresh)
    cols = [c for c in np.asarray(grid, dtype=float) if c in df.columns]
    panel = df[cols].copy()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if freq == "monthly" or freq == "M":
        panel = panel.groupby([panel.index.year, panel.index.month]).tail(1)
    elif freq not in ("daily", "D"):
        raise ValueError(f"unknown freq {freq!r}")
    return panel.dropna(how="all")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_yields(refresh=True)
    print(f"MoF JGB yields {df.index.min().date()}..{df.index.max().date()} rows={len(df)}")
    print("first non-null per tenor:")
    for c in df.columns:
        fnn = df[c].first_valid_index()
        print(f"  {c:>4.0f}y: {fnn.date() if fnn is not None else 'never'}  n={df[c].notna().sum()}")


if __name__ == "__main__":
    _main()
