"""Fetch + parse the Bank of Canada fitted zero-coupon curve.

The BoC "Yield curves for zero-coupon bonds" data is served as a plain CSV from the
legacy stats endpoint (see ``canada.config``). The CSV is wide:
``Date, ZC025YR, ZC050YR, ... , ZC3000YR`` where ``ZC<n>YR`` is the zero-coupon yield
at ``n/100`` years and values are ALREADY IN DECIMAL. Direct fitted spot rates → no
reconstruction (like the SNB / MoF / BoE curves). BoC publishes daily only; monthly is
a month-end resample.
"""
from __future__ import annotations

import argparse
import io
import logging
import re

import numpy as np
import pandas as pd

from canada import config as cfg

logger = logging.getLogger(__name__)
_TIMEOUT = 120
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
_ZC_RE = re.compile(r"^ZC(\d+)YR$")     # ZC025YR -> 0.25y, ZC1000YR -> 10y, ZC3000YR -> 30y


def _download() -> str:
    """Download the full-history yield-curve CSV; fall back to the cached copy."""
    params = {"lookupPage": cfg.BOC_YC_LOOKUP, "startRange": cfg.BOC_YC_START,
              "searchRange": "all"}    # 'all' = full history; any INVALID value 302s to maintenance
    try:
        import requests
        r = requests.get(cfg.BOC_YC_URL, params=params, headers=_BROWSER_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        text = r.text
        if "ZC025YR" not in text[:2000]:        # maintenance/HTML page, not the CSV
            raise ValueError("unexpected BoC response (not the yield-curve CSV)")
        cfg.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg.RAW_CSV_PATH.write_text(text, encoding="utf-8")
        return text
    except Exception as exc:                                        # noqa: BLE001
        if cfg.RAW_CSV_PATH.exists():
            logger.warning("BoC download failed (%s); using cached CSV", exc)
            return cfg.RAW_CSV_PATH.read_text(encoding="utf-8")
        raise


def parse_boc_csv(text: str) -> pd.DataFrame:
    """Parse the wide BoC ZC CSV into a date x maturity(years) frame (DECIMAL).

    Columns ``ZC<n>YR`` map to ``n/100`` years; values are already decimal (a trailing
    empty column from the line-ending comma is dropped)."""
    df = pd.read_csv(io.StringIO(text), skipinitialspace=True, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    date_col = df.columns[0]                                        # "Date"
    ren, mats = {}, []
    for c in df.columns:
        m = _ZC_RE.match(c)
        if m:
            yr = int(m.group(1)) / 100.0
            ren[c] = yr
            mats.append(c)
    out = df[[date_col] + mats].rename(columns=ren)
    out = out[out[date_col].str.strip() != ""].copy()
    out["date"] = pd.to_datetime(out[date_col].str.strip(), errors="coerce")
    out = out.dropna(subset=["date"]).set_index("date").drop(columns=[date_col])
    out = out.apply(pd.to_numeric, errors="coerce")                # already decimal
    out.columns = [float(c) for c in out.columns]
    return out.sort_index()


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Month-end resample: last observation in each calendar month."""
    m = daily.copy()
    m.index = pd.to_datetime(m.index)
    return m.resample("M").last()


def load_yields(refresh: bool = True, freq: str = "D") -> pd.DataFrame:
    """Fitted ZC yield frame (DECIMAL), daily or month-end. Cached to parquet."""
    pq = cfg.YIELDS_PARQUET_PATH
    if not refresh and pq.exists():
        daily = pd.read_parquet(pq)
    else:
        daily = parse_boc_csv(_download())
        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(pq)
        logger.info("BoC ZC yields: %d daily rows %s..%s", len(daily),
                    daily.index.min().date(), daily.index.max().date())
    return daily if freq in ("D", "daily") else _to_monthly(daily)


def build_yield_panel(grid: np.ndarray = cfg.COMMON_GRID, freq: str = "M",
                      start: str = cfg.SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Engine-ready CAD zero-coupon panel (DECIMAL) on ``grid`` from ``start``.
    ``freq='D'`` daily, ``'M'`` month-end. 30y is NaN before 1991 (masked per date)."""
    df = load_yields(refresh=refresh, freq=freq)
    cols = [c for c in np.asarray(grid, dtype=float) if c in df.columns]
    panel = df[cols].copy()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    return panel.dropna(how="all")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch/inspect the BoC zero-coupon curve")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = load_yields(refresh=args.refresh, freq="D")
    print(f"BoC ZC {df.index.min().date()}..{df.index.max().date()} rows={len(df)} "
          f"maturities={len(df.columns)} ({df.columns[0]:.2f}..{df.columns[-1]:.0f}y)")
    panel = build_yield_panel()
    print(f"panel {panel.shape} from {panel.index.min().date()}")
    for c in panel.columns:
        fv = panel[c].first_valid_index()
        print(f"  {c:>5.2f}y first {fv.date() if fv is not None else None} n={panel[c].notna().sum()}")
    print("latest (%):", (panel.tail(1) * 100).round(2).to_dict("records"))


if __name__ == "__main__":
    _main()
