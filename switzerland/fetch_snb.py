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
import re

import numpy as np
import pandas as pd

from switzerland import config as cfg

logger = logging.getLogger(__name__)
_TIMEOUT = 90
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}


_MATURITY_RE = re.compile(r"^\d+J$")   # keep 1J..30J, drop rating buckets (AAA/AA/A)


def _is_daily(freq: str) -> bool:
    return freq not in ("M", "monthly")


def _download(freq: str = "M") -> str:
    url = cfg.SNB_CUBE_URL_D if _is_daily(freq) else cfg.SNB_CUBE_URL
    raw_path = cfg.RAW_CSV_PATH_D if _is_daily(freq) else cfg.RAW_CSV_PATH
    try:
        import requests
        r = requests.get(url, headers=_BROWSER_HEADERS, timeout=_TIMEOUT)
        r.raise_for_status()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(r.text, encoding="utf-8")
        return r.text
    except Exception as exc:                                       # noqa: BLE001
        if raw_path.exists():
            logger.warning("SNB download failed (%s); using cached CSV", exc)
            return raw_path.read_text(encoding="utf-8")
        raise


def parse_snb_csv(text: str, freq: str = "M") -> pd.DataFrame:
    """Parse a rendoblim (monthly) / rendoblid (daily) long CSV into a
    date x maturity(years) frame (DECIMAL). The daily cube's ``D0`` also carries
    rating buckets (AAA/AA/A) — we keep only the ``<n>J`` maturity codes."""
    lines = text.splitlines()
    hdr = next(i for i, l in enumerate(lines)
               if l.replace('"', "").strip().startswith("Date;"))
    body = "\n".join(lines[hdr:])
    df = pd.read_csv(io.StringIO(body), sep=";", quotechar='"',
                     dtype=str, keep_default_na=False)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df[df["Date"].str.strip() != ""].copy()
    df["d0"] = df["D0"].str.strip()
    df = df[df["d0"].str.match(_MATURITY_RE)].copy()              # drop rating rows
    if _is_daily(freq):
        df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y-%m-%d")
    else:
        df["date"] = pd.to_datetime(df["Date"].str.strip(), format="%Y-%m") + pd.offsets.MonthEnd(0)
    df["mat"] = df["d0"].str.rstrip("Jj").astype(float)
    df["val"] = pd.to_numeric(df["Value"].str.strip().replace("", np.nan), errors="coerce")
    wide = df.pivot_table(index="date", columns="mat", values="val")
    return (wide.sort_index() / 100.0)                            # percent -> decimal


def load_yields(refresh: bool = True, freq: str = "M") -> pd.DataFrame:
    pq = cfg.YIELDS_PARQUET_PATH_D if _is_daily(freq) else cfg.YIELDS_PARQUET_PATH
    if not refresh and pq.exists():
        return pd.read_parquet(pq)
    df = parse_snb_csv(_download(freq), freq)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)
    logger.info("SNB yields (%s): %d rows %s..%s", freq, len(df),
                df.index.min().date(), df.index.max().date())
    return df


def build_yield_panel(grid: np.ndarray = cfg.COMMON_GRID, freq: str = "M",
                      start: str = cfg.SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Engine-ready CHF spot panel (DECIMAL) on ``grid`` from ``start``.
    ``freq='D'`` uses the daily ``rendoblid`` cube (to ~2025-07); monthly uses
    ``rendoblim``. Ragged: 30y is NaN before 1997-12 (masked per date downstream)."""
    df = load_yields(refresh=refresh, freq=freq)
    cols = [c for c in np.asarray(grid, dtype=float) if c in df.columns]
    panel = df[cols].copy()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
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
