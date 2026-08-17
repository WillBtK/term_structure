"""Price-based TIPS liquidity proxy — the Hu-Pan-Wang (2013) "noise" measure applied
to the TIPS cross-section, built from the individual TIPS securities on FRED.

Motivation. The joint nominal/real model's inflation-risk premium (IRP) is the small
residual of two large components (nominal TP, clean real TP). Its *level* is anchored
by the ATSIX survey (see `FINDINGS_joint_us.md`), but its *dynamics* need the TIPS
liquidity premium to be identified separately from the real term premium — a **Q-side
(pricing)** object that expectations data cannot pin. DKW publish such a series, but
using it to identify our model and then validating against DKW would be circular.

This module builds an **independent, non-circular, price-based** liquidity proxy:
per date, fit a smooth real curve to the cross-section of individual outstanding TIPS
yields and take the RMS fitting residual. When TIPS-specific liquidity is scarce
(2008-09, 2020), individual issues scatter further from any smooth curve, so the
"noise" rises — exactly the illiquidity signal (Hu, Pan & Wang, 2013, *J. Finance*;
their market-wide measure, here restricted to the TIPS cross-section).

Data: FRED individual TIPS series (`DTP*`, category 82), via the official FRED API
(key from ``FRED_API_KEY`` — far more reliable than the keyless fredgraph CSV, and it
exposes each security's maturity date in the series title). No paid data, no DKW.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from validation import config as cfg
from validation.fred_api import fred_get, fred_series

logger = logging.getLogger(__name__)

_TIPS_CATEGORY = 82                       # FRED "Treasury Inflation-Indexed Securities"
_NS_LAMBDA = 0.7                          # fixed Nelson-Siegel decay (Diebold-Li) -> linear fit
_MIN_RESID_MAT = 1.5                      # weight -> 0 below here (near-maturity TIPS blow up)
_TAPER_HI = 2.5                           # weight -> 1 above here; smooth taper across [1.5, 2.5]
_MAX_RESID_MAT = 25.0
_MIN_BONDS = 5                            # need > n_params(=3) bonds for a meaningful RMS
_MIN_EFF_BONDS = 4.0                      # min effective (weighted) bond count for a fit


def _bond_weight(tau: np.ndarray) -> np.ndarray:
    """Smooth taper on residual maturity so a near-maturity outlier TIPS crossing the
    exclusion floor fades out CONTINUOUSLY instead of dropping the basket by one bond
    (which stepped the gauge on 2026-01-14). Hermite smoothstep 0->1 across
    [_MIN_RESID_MAT, _TAPER_HI]; flat 1 in the interior; 0 at/above _MAX_RESID_MAT."""
    u = np.clip((tau - _MIN_RESID_MAT) / (_TAPER_HI - _MIN_RESID_MAT), 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)
    return np.where(tau >= _MAX_RESID_MAT, 0.0, w)


# --------------------------------------------------------------------------- io
_DUE_RE = re.compile(r"Due\s+(\d{1,2})/(\d{1,2})/(\d{4})")


def list_tips_series() -> pd.DataFrame:
    """All individual TIPS (`DTP*`) on FRED with their maturity date (parsed from the
    series title 'Due M/D/YYYY') and observation span."""
    js = fred_get("/category/series", category_id=_TIPS_CATEGORY, limit=1000)
    rows = []
    for s in js["seriess"]:
        if not s["id"].startswith("DTP"):
            continue
        m = _DUE_RE.search(s["title"])
        if not m:
            continue
        mo, dy, yr = (int(x) for x in m.groups())
        rows.append({"series": s["id"], "maturity": pd.Timestamp(yr, mo, dy),
                     "obs_start": pd.Timestamp(s["observation_start"]),
                     "obs_end": pd.Timestamp(s["observation_end"])})
    df = pd.DataFrame(rows).sort_values("maturity").reset_index(drop=True)
    logger.info("FRED TIPS universe: %d individual securities, maturities %s..%s",
                len(df), df["maturity"].min().date(), df["maturity"].max().date())
    return df


def _fetch_one(series_id: str) -> pd.Series:
    return fred_series(series_id)


def fetch_tips_yields(refresh: bool = False) -> pd.DataFrame:
    """Long tidy frame (date, series, maturity, yield_decimal) of every individual TIPS
    security's daily yield. Cached to parquet; on any fetch failure falls back to cache."""
    path = cfg.TIPS_NOISE_RAW_PARQUET
    if not refresh and path.exists():
        return pd.read_parquet(path)
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        universe = list_tips_series()
        frames = []
        for _, row in universe.iterrows():
            s = _fetch_one(row["series"])
            if s.empty:
                continue
            frames.append(pd.DataFrame({"date": s.index, "series": row["series"],
                                        "maturity": row["maturity"],
                                        "yield": s.to_numpy() / 100.0}))   # pct -> decimal
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(path)
        logger.info("TIPS yields: %d obs, %d securities, %s..%s", len(df),
                    df["series"].nunique(), df["date"].min().date(), df["date"].max().date())
        return df
    except Exception as exc:                                       # noqa: BLE001
        if path.exists():
            logger.warning("TIPS yield fetch failed (%s); using cached parquet", exc)
            return pd.read_parquet(path)
        raise


# ------------------------------------------------------------------- noise measure
def _ns_design(tau: np.ndarray, lam: float = _NS_LAMBDA) -> np.ndarray:
    """Nelson-Siegel (Diebold-Li) regressors [1, slope-load, curvature-load] at tenors
    ``tau`` for a fixed decay ``lam`` (so the curve fit is ordinary least squares)."""
    lt = lam * tau
    s = (1.0 - np.exp(-lt)) / lt
    c = s - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), s, c])


def _rms_residual(tau: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted RMS residual of a fixed-λ Nelson-Siegel WLS fit to (tau, y) — the
    cross-sectional 'noise'. Weights ``w`` (default from ``_bond_weight``) taper
    near-maturity bonds smoothly, so basket-composition changes don't step the series;
    the fit and the RMS use the SAME weights so a down-weighted outlier distorts neither.
    NaN if too few (effective) bonds."""
    if tau.size < _MIN_BONDS:
        return np.nan
    if w is None:
        w = _bond_weight(tau)
    if w.sum() < _MIN_EFF_BONDS:
        return np.nan
    X = _ns_design(tau)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)   # WLS
    resid = y - X @ beta
    return float(np.sqrt(np.sum(w * resid ** 2) / w.sum()))


def build_noise(refresh: bool = False, freq: str = "M") -> pd.DataFrame:
    """The TIPS 'noise' liquidity proxy: per date, RMS deviation of individual TIPS
    yields from a smooth (fixed-λ NS) fit across the cross-section, in DECIMAL. Returns
    (date, noise, n_bonds); ``freq='M'`` gives month-end, 'D' daily. Cached to parquet."""
    path = cfg.TIPS_NOISE_PARQUET
    if not refresh and path.exists():
        out = pd.read_parquet(path)
        return out if freq == "D" else _to_monthly(out)
    raw = fetch_tips_yields(refresh=refresh)
    raw = raw.dropna(subset=["yield"])
    recs = []
    for dt, g in raw.groupby("date"):
        resid_mat = (g["maturity"] - dt).dt.days.to_numpy() / 365.25
        keep = (resid_mat >= _MIN_RESID_MAT) & (resid_mat <= _MAX_RESID_MAT)
        tau = resid_mat[keep]; y = g["yield"].to_numpy()[keep]
        if tau.size < _MIN_BONDS:
            continue
        w = _bond_weight(tau)
        # effective (weighted) bond count — an integer n_bonds would step whenever a
        # tapering bond crosses the floor; the effective count moves continuously.
        recs.append({"date": dt, "noise": _rms_residual(tau, y, w),
                     "n_bonds": round(float(w.sum()), 2)})
    daily = pd.DataFrame(recs).dropna().sort_values("date").reset_index(drop=True)
    daily.to_parquet(path)
    logger.info("TIPS noise: %d daily obs %s..%s, latest %.1f bp (%.1f eff. bonds)",
                len(daily), daily["date"].min().date(), daily["date"].max().date(),
                daily["noise"].iloc[-1] * 1e4, daily["n_bonds"].iloc[-1])
    return daily if freq == "D" else _to_monthly(daily)


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["ym"] = d["date"].dt.to_period("M")
    m = d.groupby("ym").agg(noise=("noise", "mean"), n_bonds=("n_bonds", "last")).reset_index()
    m["date"] = m["ym"].dt.to_timestamp("M")
    return m[["date", "noise", "n_bonds"]]


def noise_panel(dates, freq: str = "M", refresh: bool = False) -> pd.Series:
    """The noise proxy aligned (by month) to ``dates`` — a Series indexed by ``dates``."""
    m = build_noise(refresh=refresh, freq="M")
    s = m.set_index(m["date"].dt.to_period("M"))["noise"]
    dates = pd.DatetimeIndex(dates)
    out = s.reindex(dates.to_period("M"))
    out.index = dates
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import argparse
    ap = argparse.ArgumentParser(description="TIPS liquidity 'noise' proxy (FRED)")
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    m = build_noise(refresh=args.refresh, freq="M")
    print(f"TIPS noise (monthly): {len(m)} obs {m['date'].min().date()}..{m['date'].max().date()}")
    print("  recent:", (m.tail(3).assign(noise_bp=lambda d: (d["noise"] * 1e4).round(1))
                        [["date", "noise_bp", "n_bonds"]].to_string(index=False)))
    # spikes: top months
    top = m.assign(bp=(m["noise"] * 1e4).round(1)).nlargest(6, "noise")[["date", "bp", "n_bonds"]]
    print("  largest noise (illiquidity) months:\n", top.to_string(index=False))


if __name__ == "__main__":
    _main()
