"""Fetch + parse SNB Swiss Confederation spot rates (cube ``rendoblim``).

The SNB data portal serves the estimated yield curve as a semicolon-delimited CSV
in long form: a few metadata lines, then ``Date;D0;Value`` with ``Date`` = month
(YYYY-MM), ``D0`` = maturity code (``1J`` .. ``30J``, J = year), ``Value`` in
percent. We pivot to a date x maturity(years) panel in decimal. Direct spot rates
→ no reconstruction (like Japan's MoF / the BoE GLC).
"""
from __future__ import annotations

import argparse
import io
import logging

import numpy as np
import pandas as pd

from switzerland import config as cfg

logger = logging.getLogger(__name__)
_TIMEOUT = 90
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}


def _download() -> str:
    try:
        import requests
        r = requests.get(cfg.SNB_CUBE_URL, headers=_BROWSER_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        cfg.RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg.RAW_CSV_PATH.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception as exc:                                       # noqa: BLE001
        if cfg.RAW_CSV_PATH.exists():
            logger.warning("SNB download failed (%s); using cached CSV", exc)
            return cfg.RAW_CSV_PATH.read_text(encoding="utf-8")
        raise


def parse_snb_csv(text: str) -> pd.DataFrame:
    """Parse the rendoblim long CSV into a date x maturity(years) frame (DECIMAL)."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.replace('"', "").strip().startswith("Date;"))
    body = "\n".join(lines[hdr:])
    df = pd.read_csv(io.StringIO(body), sep=";", quotechar='"',
                     dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df[df["Date"].str.strip() != ""].copy()
    df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y-%m") + pd.offsets.MonthEnd(0)
    df["mat"] = df["D0"].str.strip().str.rstrip("Jj").astype(float)
    df["val"] = pd.to_numeric(df["Value"].str.strip().replace("", np.nan), errors="coerce")
    wide = df.pivot_table(index="date", columns="mat", values="val")
    return (wide.sort_index() / 100.0)                            # percent -> decimal


def load_yields(refresh: bool = True) -> pd.DataFrame:
    if not refresh and cfg.YIELDS_PARQUET_PATH.exists():
        return pd.read_parquet(cfg.YIELDS_PARQUET_PATH)
    df = parse_snb_csv(_download())
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.YIELDS_PARQUET_PATH)
    logger.info("SNB yields: %d rows %s..%s", len(df),
                df.index.min().date(), df.index.max().date())
    return df


def build_yield_panel(grid: np.ndarray = cfg.COMMON_GRID, freq: str = "M",
                      start: str = cfg.SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Engine-ready CHF spot panel (DECIMAL) on ``grid``, monthly, from ``start``.
    Ragged: 30y is NaN before 1997-12 (masked per date downstream)."""
    df = load_yields(refresh=refresh)
    cols = [c for c in np.asarray(grid, dtype=float) if c in df.columns]
    panel = df[cols].copy()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if freq not in ("M", "monthly"):
        raise ValueError("SNB rendoblim is monthly; freq must be 'M'")
    return panel.dropna(how="all")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch/inspect SNB CHF spot curve")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = load_yields(refresh=args.refresh)
    print(f"SNB {df.index.min().date()}..{df.index.max().date()} rows={len(df)} maturities {list(df.columns)}")
    panel = build_yield_panel()
    print(f"panel {panel.shape}")
    for c in panel.columns:
        fv = panel[c].first_valid_index()
        print(f"  {c:>4.0f}y first {fv.date() if fv is not None else None} n={panel[c].notna().sum()}")
    print("latest (%):", (panel.tail(1) * 100).round(2).to_dict("records"))


if __name__ == "__main__":
    _main()
