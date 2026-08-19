"""Fetch + parse Bank of England GLC month-end curves (nominal / real / inflation).

Each curve is a zip of Excel workbooks split by era (e.g. 1970-2015, 2016-2024,
2025-present). Within each workbook the ``4. spot curve`` sheet has a maturity
header row labelled ``years:`` and month-end dated rows of spot rates in percent.
We parse every workbook, concatenate across eras, convert to decimal, and subset
to the estimation grid. Direct spot rates -> no parametric reconstruction.

Self-healing: the download URL is discovered from the BoE yield-curves page
(falling back to the stable media path) so a site reorg doesn't silently break the
fetch (mirrors the Krippner self-healing link).
"""
from __future__ import annotations

import argparse
import io
import logging
import re
import zipfile

import numpy as np
import pandas as pd

from uk import config as cfg

logger = logging.getLogger(__name__)
_TIMEOUT = 120
_BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}

_ZIP = {"nominal": cfg.BOE_NOMINAL_ZIP, "real": cfg.BOE_REAL_ZIP,
        "inflation": cfg.BOE_INFLATION_ZIP}
_ZIP_D = {"nominal": cfg.BOE_NOMINAL_ZIP_D, "real": cfg.BOE_REAL_ZIP_D,
          "inflation": cfg.BOE_INFLATION_ZIP_D}
_PARQUET = {"nominal": cfg.NOMINAL_PARQUET, "real": cfg.REAL_PARQUET,
            "inflation": cfg.INFLATION_PARQUET}


def _is_daily(freq: str) -> bool:
    return freq not in ("M", "monthly")


def _zip_name(kind: str, freq: str) -> str:
    return _ZIP_D[kind] if _is_daily(freq) else _ZIP[kind]


def _parquet_path(kind: str, freq: str):
    pq = _PARQUET[kind]
    return pq.with_name(pq.stem + "_daily" + pq.suffix) if _is_daily(freq) else pq


def discover_zip_url(zip_name: str) -> str:
    """Return the download URL for ``zip_name``, discovered from the yield-curves
    page when possible, else the stable media path."""
    try:
        import requests
        html = requests.get(cfg.BOE_YIELD_PAGE, headers=_BROWSER_HEADERS,
                            timeout=_TIMEOUT).text
        m = re.search(r'href="([^"]*' + re.escape(zip_name) + r')"', html)
        if m:
            href = m.group(1)
            return href if href.startswith("http") else "https://www.bankofengland.co.uk" + href
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("BoE URL discovery failed (%s); using default path", exc)
    return f"{cfg.BOE_MEDIA_BASE}/{zip_name}"


def _download_zip(kind: str, freq: str = "M") -> bytes:
    import requests
    zip_name = _zip_name(kind, freq)
    url = discover_zip_url(zip_name)
    logger.info("downloading BoE %s curve (%s): %s", kind, freq, url)
    content = requests.get(url, headers=_BROWSER_HEADERS, timeout=_TIMEOUT).content
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (cfg.RAW_DIR / zip_name).write_bytes(content)
    return content


def parse_spot_workbook(xls_bytes: bytes) -> pd.DataFrame:
    """Parse one GLC workbook's ``4. spot curve`` sheet into a date x maturity(years)
    frame of spot rates in DECIMAL. Returns empty if the sheet is absent."""
    xl = pd.ExcelFile(io.BytesIO(xls_bytes))
    sheet = next((s for s in xl.sheet_names if s.strip().lower() == cfg.SPOT_SHEET), None)
    if sheet is None:
        sheet = next((s for s in xl.sheet_names if "spot curve" in s.lower()), None)
    if sheet is None:
        return pd.DataFrame()
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    # locate the 'years:' maturity header row
    yr = None
    for i in range(min(10, len(raw))):
        if str(raw.iloc[i, 0]).strip().lower() == cfg.YEARS_LABEL:
            yr = i
            break
    if yr is None:
        return pd.DataFrame()
    mats = pd.to_numeric(raw.iloc[yr, 1:], errors="coerce")
    valid = mats.notna().to_numpy()
    mat_cols = mats[valid].to_numpy(dtype=float)
    body = raw.iloc[yr + 1:].copy()
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    keep = dates.notna().to_numpy()
    body = body.loc[keep]
    dates = dates[keep]
    vals = body.iloc[:, 1:].to_numpy(dtype=float)[:, valid]        # (T, m) percent
    df = pd.DataFrame(vals / 100.0, index=pd.DatetimeIndex(dates.values, name="date"),
                      columns=mat_cols)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _latest_current_month(kind: str) -> list:
    """Fetch the BoE 'latest' zip and parse the current-month daily workbook for ``kind``
    ('GLC <Kind> daily data current month.xlsx'). Best-effort: returns [] on any failure
    so a hiccup falls back to the archive alone."""
    try:
        import requests
        url = discover_zip_url(cfg.BOE_LATEST_ZIP)
        content = requests.get(url, headers=_BROWSER_HEADERS, timeout=_TIMEOUT).content
        cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (cfg.RAW_DIR / cfg.BOE_LATEST_ZIP).write_bytes(content)
        out = []
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            for name in z.namelist():
                low = name.lower()
                if kind in low and "current month" in low and low.endswith((".xlsx", ".xls")):
                    fr = parse_spot_workbook(z.read(name))
                    if not fr.empty:
                        out.append(fr)
                        logger.info("BoE %s current-month file -> %s (last %s)", kind, name,
                                    fr.index.max().date())
        return out
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("BoE %s current-month fetch failed (%s); archive only", kind, exc)
        return []


def load_curve(kind: str = "nominal", refresh: bool = False,
               freq: str = "M") -> pd.DataFrame:
    """Full BoE curve (all maturities, decimal), concatenated across era workbooks.
    Cached to parquet. ``freq='D'`` pulls the daily GLC archive instead of
    month-end (same ``4. spot curve`` workbook layout)."""
    pq = _parquet_path(kind, freq)
    if not refresh and pq.exists():
        return pd.read_parquet(pq)
    content = _download_zip(kind, freq)
    frames = []
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if name.lower().endswith((".xlsx", ".xls")):
                fr = parse_spot_workbook(z.read(name))
                if not fr.empty:
                    frames.append(fr)
    if not frames:
        raise RuntimeError(f"No spot-curve sheets parsed from BoE {kind} zip.")
    if _is_daily(freq):
        frames += _latest_current_month(kind)          # append the intra-month daily file
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(sorted(df.columns), axis=1)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pq)
    logger.info("BoE %s: %d rows %s..%s, maturities %.1f..%.1f", kind, len(df),
                df.index.min().date(), df.index.max().date(),
                df.columns.min(), df.columns.max())
    return df


def build_yield_panel(kind: str = "nominal", grid=None, freq: str = "M",
                      start: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Engine-ready panel (DECIMAL) for a BoE curve on ``grid``, month-end, from
    ``start``. Maturities absent for a date are NaN (ragged)."""
    if grid is None:
        grid = cfg.REAL_GRID if kind == "real" else cfg.COMMON_GRID
    df = load_curve(kind, refresh=refresh, freq=freq)
    grid = np.asarray(grid, dtype=float)
    # BoE columns are exact 0.5-steps; select the grid maturities that exist.
    cols = [c for c in grid if c in df.columns]
    missing = [c for c in grid if c not in df.columns]
    if missing:
        logger.warning("grid maturities not in BoE %s columns: %s", kind, missing)
    panel = df[cols].copy()
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    return panel.dropna(how="all")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch/inspect BoE GLC curves")
    ap.add_argument("--kind", default="nominal", choices=["nominal", "real", "inflation"])
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    df = load_curve(args.kind, refresh=args.refresh)
    print(f"BoE {args.kind}: {df.index.min().date()}..{df.index.max().date()} "
          f"rows={len(df)} maturities {df.columns.min():.1f}..{df.columns.max():.1f}")
    grid = cfg.REAL_GRID if args.kind == "real" else cfg.COMMON_GRID
    panel = build_yield_panel(args.kind, grid=grid)
    print(f"panel on grid {list(grid)}: {panel.shape}")
    for c in panel.columns:
        fv = panel[c].first_valid_index()
        print(f"  {c:>4.1f}y: {fv.date() if fv is not None else 'never'}  n={panel[c].notna().sum()}")


if __name__ == "__main__":
    _main()
