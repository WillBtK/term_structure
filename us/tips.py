"""US TIPS real-yield curve + breakeven inflation (GSW feds200805).

The TIPS file is the real-curve analog of the nominal feds200628: same Svensson
parametric form (BETA0-3, TAU1-2), plus published fitted real zeros ``TIPSYNN``
and breakeven inflation ``BKEVENNN`` (percent). We reconstruct the real zero
curve with the shared Svensson math (`us.reconstruct.svensson_yield`) and also
expose the published breakevens directly.

Two things differ from the nominal leg (both by design):
- **No ELB.** Real short rates have no lower bound (no "real cash"), so the real
  term premium is a *plain-AFNS* decomposition — we run the shadow pipeline with
  the bound pushed to `config.ELB_REAL` (-100%), where it never binds.
- **2y-20y only, thin before ~2004.** TIPS do not trade at the short end, so the
  real grid starts at 2y (`config.REAL_GRID`); the reliable sample starts ~2004
  (`config.REAL_SAMPLE_START`) — GSW caution against the illiquid 1999-2003 fits.

The GSW header itself notes breakevens "incorporate inflation risk premiums and
the effects of the differential liquidity of TIPS and nominal securities" — the
liquidity caveat we carry into the inflation-risk-premium decomposition.
"""
from __future__ import annotations

import argparse
import io
import logging

import numpy as np
import pandas as pd

from us import config as cfg
from us.fetch_gsw import _download, _find_header_row
from us.reconstruct import svensson_yield

logger = logging.getLogger(__name__)

_PARAM_COLUMNS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def parse_tips_csv(text: str) -> pd.DataFrame:
    """Parse feds200805 into a date-indexed frame of BETA/TAU params + TIPSYNN +
    BKEVENNN columns (all coerced to float, percent)."""
    skip = _find_header_row(text)
    df = pd.read_csv(io.StringIO(text), skiprows=skip,
                     na_values=["NA", "ND", "#N/A", "N/A", "."], keep_default_na=True)
    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if date_col is None:
        raise ValueError("TIPS file has no 'Date' column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    miss = [c for c in _PARAM_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"TIPS file missing params {miss}; have {list(df.columns)[:10]}...")

    def _pref(prefix):
        return [c for c in df.columns
                if c.startswith(prefix) and c[len(prefix):].isdigit()]
    keep = (_PARAM_COLUMNS + _pref(cfg.TIPS_PUBLISHED_YIELD_PREFIX)
            + _pref(cfg.TIPS_BREAKEVEN_PREFIX))
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    return out.dropna(how="all", subset=_PARAM_COLUMNS)


def load_tips_params(refresh: bool = False) -> pd.DataFrame:
    """Load TIPS params (+TIPSY/BKEVEN), downloading + caching feds200805."""
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if refresh or not cfg.TIPS_PARQUET_PATH.exists():
        try:
            logger.info("Downloading TIPS params from %s", cfg.TIPS_URL)
            text = _download(cfg.TIPS_URL)
            df = parse_tips_csv(text)
            cfg.TIPS_RAW_CSV_PATH.write_text(text)
            df.to_parquet(cfg.TIPS_PARQUET_PATH)
            logger.info("Cached %d TIPS rows (%s..%s)", len(df),
                        df.index.min().date(), df.index.max().date())
            return df
        except Exception as exc:                                   # noqa: BLE001
            if cfg.TIPS_PARQUET_PATH.exists():
                logger.warning("TIPS download failed (%s); using cache", exc)
            else:
                raise RuntimeError(
                    f"Could not download TIPS from {cfg.TIPS_URL} and no cache "
                    f"exists. Run where egress to federalreserve.gov is open.") from exc
    return pd.read_parquet(cfg.TIPS_PARQUET_PATH)


def _availability_cap(params: pd.DataFrame) -> "pd.Series | None":
    """Per-date longest published TIPSY maturity (years), for ragged masking."""
    pref = cfg.TIPS_PUBLISHED_YIELD_PREFIX
    cols = [c for c in params.columns if c.startswith(pref) and c[len(pref):].isdigit()]
    if not cols:
        return None
    mats = np.array([int(c[len(pref):]) for c in cols], dtype=float)
    present = params[cols].notna().to_numpy()
    return pd.Series((present * mats[np.newaxis, :]).max(axis=1), index=params.index)


def _floor_cap(params: pd.DataFrame) -> "pd.Series | None":
    """Per-date SHORTEST published TIPSY maturity (years): TIPS do not trade at the
    short end, so we must not extrapolate the Svensson fit below the shortest
    available real bond (masked to NaN, ragged like the long end)."""
    pref = cfg.TIPS_PUBLISHED_YIELD_PREFIX
    cols = [c for c in params.columns if c.startswith(pref) and c[len(pref):].isdigit()]
    if not cols:
        return None
    mats = np.array([int(c[len(pref):]) for c in cols], dtype=float)
    present = params[cols].notna().to_numpy()
    big = np.where(present, mats[np.newaxis, :], np.inf)
    floor = big.min(axis=1)
    floor[~np.isfinite(floor)] = np.inf
    return pd.Series(floor, index=params.index)


def reconstruct_real(params: pd.DataFrame, grid: np.ndarray = cfg.REAL_GRID,
                     mask_by_availability: bool = True) -> pd.DataFrame:
    """Real zero curve (DECIMAL) on ``grid`` from TIPS Svensson params.

    Reuses the shared Svensson formula. With ``mask_by_availability`` each maturity
    is NaN outside the published TIPSY range for that date (ragged both ends: no
    short TIPS, and the long end extends over time)."""
    valid = params[["BETA0", "BETA1", "BETA2", "TAU1"]].notna().all(axis=1) & (params["TAU1"] > 0)
    has_sv = params["BETA3"].notna() & params["TAU2"].notna() & (params["TAU2"] > 0)
    p = params.loc[valid].copy()
    ns_only = ~has_sv.loc[p.index].to_numpy()
    beta3 = np.where(ns_only, 0.0, p["BETA3"].to_numpy(dtype=float))
    tau2 = np.where(ns_only, 1.0, p["TAU2"].to_numpy(dtype=float))
    grid = np.asarray(grid, dtype=float)
    yy = svensson_yield(grid[np.newaxis, :],
                        p["BETA0"].to_numpy()[:, None], p["BETA1"].to_numpy()[:, None],
                        p["BETA2"].to_numpy()[:, None], beta3[:, None],
                        p["TAU1"].to_numpy()[:, None], tau2[:, None])
    out = pd.DataFrame(yy / 100.0, index=p.index, columns=grid)
    if mask_by_availability:
        cap = _availability_cap(params)
        flo = _floor_cap(params)
        if cap is not None:
            c = cap.loc[out.index].to_numpy()[:, None]
            out = out.mask(grid[np.newaxis, :] > c)
        if flo is not None:
            f = flo.loc[out.index].to_numpy()[:, None]
            out = out.mask(grid[np.newaxis, :] < f)
    return out


def build_real_panel(grid: np.ndarray = cfg.REAL_GRID, freq: str = "M",
                     start: str = cfg.REAL_SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Engine-ready real zero panel (DECIMAL), monthly by default, from ``start``."""
    params = load_tips_params(refresh=refresh)
    panel = reconstruct_real(params, grid)
    if start is not None:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if freq in ("M", "monthly"):
        panel = panel.groupby([panel.index.year, panel.index.month]).tail(1)
    elif freq not in ("D", "daily"):
        raise ValueError(f"unknown freq {freq!r}")
    return panel.dropna(how="all")


def build_breakeven_panel(grid: np.ndarray = cfg.REAL_GRID, freq: str = "M",
                          start: str = cfg.REAL_SAMPLE_START, refresh: bool = False) -> pd.DataFrame:
    """Published breakeven inflation (DECIMAL) at integer maturities on ``grid``.

    Breakeven = nominal - real yield = expected inflation + inflation risk premium
    (+ liquidity differential). Uses the file's own BKEVENNN columns where present.
    """
    params = load_tips_params(refresh=refresh)
    pref = cfg.TIPS_BREAKEVEN_PREFIX
    cols = {int(c[len(pref):]): c for c in params.columns
            if c.startswith(pref) and c[len(pref):].isdigit()}
    want = [m for m in np.asarray(grid, dtype=float) if int(m) in cols]
    be = pd.DataFrame({m: params[cols[int(m)]] for m in want}) / 100.0
    if start is not None:
        be = be[be.index >= pd.Timestamp(start)]
    if freq in ("M", "monthly"):
        be = be.groupby([be.index.year, be.index.month]).tail(1)
    return be.dropna(how="all")


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Fetch/inspect GSW TIPS real curve")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    params = load_tips_params(refresh=args.refresh)
    print(f"TIPS params {params.index.min().date()}..{params.index.max().date()} rows={len(params)}")
    panel = build_real_panel()
    print(f"real panel {panel.shape} {panel.index.min().date()}..{panel.index.max().date()}")
    print("first-valid per real tenor:")
    for c in panel.columns:
        fv = panel[c].first_valid_index()
        print(f"  {c:>4.0f}y: {fv.date() if fv is not None else 'never'}  n={panel[c].notna().sum()}")
    print("latest real yields (%):", (panel.tail(1) * 100).round(2).to_dict("records"))


if __name__ == "__main__":
    _main()
