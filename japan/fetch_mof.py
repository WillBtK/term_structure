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


def _download_one(url: str, cache, timeout: int = 90) -> "str | None":
    """Fetch a MoF CSV, cache it (Shift-JIS bytes), and return decoded text. On a fetch
    failure, fall back to the cache; return None if neither is available."""
    try:
        import requests
        content = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout).content
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(content)
        return content.decode("shift_jis", errors="replace")
    except Exception as exc:                                        # noqa: BLE001
        if cache.exists():
            logger.warning("MoF download failed (%s); using cached %s", exc, cache.name)
            return cache.read_bytes().decode("shift_jis", errors="replace")
        logger.warning("MoF download failed (%s) and no cache for %s", exc, cache.name)
        return None


def _download(timeout: int = 90) -> str:
    """Fetch the full-history jgbcm_all.csv (falls back to the raw cache)."""
    txt = _download_one(cfg.MOF_JGB_URL, cfg.RAW_CSV_PATH, timeout)
    if txt is None:
        raise RuntimeError("MoF jgbcm_all.csv unavailable and no cache")
    return txt


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
    """Full MoF JGB yield frame (all tenors, decimal). Cached to parquet.

    Merges the ~monthly full-history file (jgbcm_all.csv) with the daily-updated
    current-year file (jgbcm.csv), preferring the current-year rows on overlap, so the
    panel reaches the latest business day rather than the prior month-end."""
    if not refresh and cfg.YIELDS_PARQUET_PATH.exists():
        return pd.read_parquet(cfg.YIELDS_PARQUET_PATH)
    hist = parse_jgb_csv(_download())
    frames = [hist]
    cur_txt = _download_one(cfg.MOF_JGB_CURRENT_URL, cfg.RAW_CURRENT_CSV_PATH)
    if cur_txt:
        try:
            cur = parse_jgb_csv(cur_txt)
            if len(cur):
                frames.append(cur)
        except Exception as exc:                                   # noqa: BLE001
            logger.warning("MoF current-year parse failed (%s); history only", exc)
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()        # current-year wins on overlap
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
