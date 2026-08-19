"""Fetch + cache the Deutsche Bundesbank estimated term-structure (Svensson)
parameters for German listed Federal securities, and reconstruct the zero-yield
grid.

Data flows over the Bundesbank SDMX REST API (flow ``BBSIS``) as one CSV per
series. We pull the six Svensson parameters (BETA0-3 in percent, TAU1-2 in years
— the same convention as GSW), assemble a date x param frame, and reconstruct the
common maturity grid with the shared ``us.reconstruct`` Svensson math (verified to
match the Bundesbank's own published spot yields to < 1 bp). Monthly (from 1972-09)
is the default; daily (from 1997-08) is available via ``freq='D'``.
"""
from __future__ import annotations

import io
import logging
import re

import numpy as np
import pandas as pd

from germany import config as cfg
from us.reconstruct import reconstruct_grid

logger = logging.getLogger(__name__)

_DATE_RE = re.compile(r"^(\d{4}-\d{2}(?:-\d{2})?),([-0-9.]*)")


def _has_observations(text: str) -> bool:
    """True when the payload actually carries dated rows (not an error page)."""
    return any(_DATE_RE.match(line) for line in io.StringIO(text))


def _download_tsdb(series_key: str, timeout: int = 60) -> str:
    """GET one series from the public time-series database (the un-challenged
    mirror of the SDMX flow). Same ``YYYY-MM[-DD],value,flag`` body."""
    import urllib.parse
    import urllib.request
    qs = urllib.parse.urlencode({
        "tsId": f"{cfg.BUNDESBANK_FLOW}.{series_key}",
        "its_csvFormat": "en", "its_fileFormat": "csv", "mode": "its",
    })
    req = urllib.request.Request(f"{cfg.BUNDESBANK_TSDB}?{qs}",
                                 headers={"Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8-sig")


def _download_csv(series_key: str, timeout: int = 60) -> str:
    """GET one BBSIS series as CSV (English).

    Tries the SDMX API first, then the time-series-database mirror. The API host
    is fronted by a bot challenge that answers automated clients with a 303 to
    ``/.enodia/challenge`` — a *successful* HTTP response carrying no data — so a
    clean status code is not enough: the payload must contain dated rows before
    we accept it. Raises only when both routes fail.
    """
    url = f"{cfg.BUNDESBANK_API}/{series_key}"
    params = {"format": "csv", "lang": "en"}
    headers = {"Accept": "text/csv"}
    try:
        import requests
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.content.decode("utf-8-sig")
        if _has_observations(text):
            return text
        logger.warning("SDMX API returned no observations for %s "
                       "(bot challenge?); falling back to the TSDB mirror",
                       series_key)
    except Exception as exc:
        logger.warning("SDMX API failed for %s (%s); falling back to the "
                       "TSDB mirror", series_key, exc)
    return _download_tsdb(series_key, timeout=timeout)


def parse_series(csv_text: str, freq: str = "M") -> pd.Series:
    """Parse a Bundesbank BBSIS CSV into a date-indexed float Series.

    The file carries a short metadata preamble (series id, description,
    ``BBK_UNIT_ENG``, ``Decimals`` ...) followed by ``YYYY-MM[-DD],value,flag``
    rows; empty values become NaN.
    """
    dates, vals = [], []
    for line in io.StringIO(csv_text):
        m = _DATE_RE.match(line)
        if not m:
            continue
        dates.append(m.group(1))
        raw = m.group(2)
        # Bundesbank marks non-trading days (weekends/holidays) with a lone "."
        # in the daily series; empty and "." both mean "no value" -> NaN.
        try:
            vals.append(float(raw))
        except (TypeError, ValueError):
            vals.append(np.nan)
    idx = pd.PeriodIndex(dates, freq=("M" if freq == "M" else "D")).to_timestamp(
        how="end" if freq == "M" else "start").normalize()
    return pd.Series(vals, index=idx, dtype=float).sort_index()


def _params_parquet(freq: str):
    """Frequency-specific cache path so a daily fetch never clobbers the monthly
    cache (monthly reaches 1972; daily only 1997-08)."""
    pq = cfg.PARAMS_PARQUET_PATH
    return pq.with_name(pq.stem + "_daily" + pq.suffix) if freq not in ("M", "monthly") else pq


def load_params(freq: str = "M", refresh: bool = True) -> pd.DataFrame:
    """Return a date x ``PARAM_COLUMNS`` frame of Svensson parameters (percent /
    years). Caches to a per-frequency parquet; falls back to it on network failure."""
    pq = _params_parquet(freq)
    if not refresh and pq.exists():
        return pd.read_parquet(pq)
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    series = {}
    try:
        for name in cfg.PARAM_COLUMNS:
            key = cfg.param_series_key(name, freq)
            text = _download_csv(key)
            (cfg.RAW_DIR / f"{name}_{freq}.csv").write_text(text, encoding="utf-8")
            series[name] = parse_series(text, freq)
        params = pd.DataFrame(series)[cfg.PARAM_COLUMNS]
        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        params.to_parquet(pq)
        logger.info("Bundesbank params %s: %d rows %s..%s", freq, len(params),
                    params.index.min().date(), params.index.max().date())
        return params
    except Exception as exc:
        if pq.exists():
            logger.warning("download failed (%s); using cached params", exc)
            return pd.read_parquet(pq)
        raise


def build_yield_panel(grid: np.ndarray = cfg.COMMON_GRID, freq: str = "M",
                      refresh: bool = False) -> pd.DataFrame:
    """Reconstruct the zero-yield grid (decimal) from the Svensson parameters.

    The German curve is out to 10y and dense from 1972-09, so no availability
    masking is needed (``mask_by_availability=False``)."""
    params = load_params(freq=freq, refresh=refresh)
    return reconstruct_grid(params, grid, mask_by_availability=False)


def cross_check_against_published(freq: str = "M", tol_bp: float = 2.0) -> pd.DataFrame:
    """Compare reconstruction to the Bundesbank's own published spot yields.

    Returns per-maturity ``[maturity, max_abs_diff_bp, mean_abs_diff_bp,
    within_tol]``. A correct reconstruction agrees to well within a basis point.
    """
    params = load_params(freq=freq, refresh=False)
    rows = []
    for mat, _code in sorted(cfg.SPOT_CODES.items()):
        recon = reconstruct_grid(params, np.array([mat]),
                                 mask_by_availability=False).iloc[:, 0] * 100.0
        published = parse_series(_download_csv(cfg.spot_series_key(mat, freq)), freq)
        common = recon.index.intersection(published.index)
        diff_bp = (recon.loc[common] - published.loc[common]).abs() * 100.0
        rows.append({
            "maturity": mat,
            "max_abs_diff_bp": float(diff_bp.max()),
            "mean_abs_diff_bp": float(diff_bp.mean()),
            "within_tol": bool(diff_bp.max() <= tol_bp),
        })
    return pd.DataFrame(rows)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="Fetch Bundesbank term-structure params")
    ap.add_argument("--freq", default="M", choices=["M", "D"])
    ap.add_argument("--check", action="store_true", help="cross-check vs published spots")
    args = ap.parse_args()
    panel = build_yield_panel(freq=args.freq, refresh=True)
    print(f"panel {panel.index.min().date()}..{panel.index.max().date()} "
          f"rows={len(panel)} cols={list(panel.columns)}")
    if args.check:
        print(cross_check_against_published(freq=args.freq).to_string(index=False))


if __name__ == "__main__":
    _main()
