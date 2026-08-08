"""Validation for the US shadow-rate AFNS (Christensen-Rudebusch).

Per the top-level README, the shadow-rate model should:
- collapse to the plain AFNS **away from the bound** (the ELB "wedge" -> 0), and
- **diverge** during the ZLB/negative-rate window (2008-2015, 2020-2021) — that
  divergence is expected, not a bug.

External shadow-rate benchmarks (Wu-Xia @ atlantafed.org, Krippner @ ljkmfa.com)
are egress-blocked, so the external anchor here is the effective federal funds
rate from FRED (reachable): the shadow short rate should track the policy rate
when it is comfortably positive and fall **below** it (negative) during the ZLB.
If a proper external shadow-rate cross-check is wanted, allowlist
`www.atlantafed.org` and add a Wu-Xia loader.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

FEDFUNDS_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"

# ZLB windows for the US (effective-lower-bound episodes).
ZLB_WINDOWS = [("2008-12-01", "2015-12-01"), ("2020-03-01", "2021-12-01")]


def load_fed_funds() -> pd.Series:
    """Monthly effective federal funds rate (decimal) from FRED."""
    import requests

    txt = requests.get(FEDFUNDS_CSV, timeout=60).text
    df = pd.read_csv(io.StringIO(txt))
    df.columns = ["date", "ff"]
    df["date"] = pd.to_datetime(df["date"])
    df["ff"] = pd.to_numeric(df["ff"], errors="coerce") / 100.0
    return df.dropna().set_index("date")["ff"]


def _in_windows(idx: pd.DatetimeIndex, windows) -> np.ndarray:
    mask = np.zeros(len(idx), dtype=bool)
    for lo, hi in windows:
        mask |= (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
    return mask


def internal_checks(shadow_df: pd.DataFrame, wedge_df: pd.DataFrame) -> dict:
    """ELB wedge should be ~0 away from the bound and large during the ZLB.

    Returns mean |wedge| (bp, averaged over maturities) inside vs outside the ZLB
    windows, and the shadow short-rate minimum / its date.
    """
    idx = pd.DatetimeIndex(shadow_df["date"])
    wedge_bp = wedge_df.abs().mean(axis=1) * 1e4
    zlb = _in_windows(idx, ZLB_WINDOWS)
    s = shadow_df.set_index("date")["shadow_short_rate"]
    return {
        "mean_wedge_bp_zlb": float(np.nanmean(wedge_bp.to_numpy()[zlb])),
        "mean_wedge_bp_normal": float(np.nanmean(wedge_bp.to_numpy()[~zlb])),
        "shadow_min": float(s.min()),
        "shadow_min_date": str(s.idxmin().date()),
        "shadow_negative_months": int((s < 0).sum()),
    }


def compare_to_fed_funds(shadow_df: pd.DataFrame, ff: pd.Series) -> dict:
    """Directional check vs the effective fed funds rate.

    Away from the bound the shadow rate should track the policy rate; during the
    ZLB it should fall below it (and go negative).
    """
    s = shadow_df.set_index("date")["shadow_short_rate"]
    idx = s.index.intersection(ff.index)
    s, f = s.loc[idx], ff.loc[idx]
    zlb = _in_windows(pd.DatetimeIndex(idx), ZLB_WINDOWS)
    normal = ~zlb
    out = {
        "corr_normal": float(np.corrcoef(s[normal], f[normal])[0, 1]) if normal.sum() > 2 else np.nan,
        "mean_gap_bp_normal": float(np.mean(s[normal] - f[normal]) * 1e4),
        "mean_gap_bp_zlb": float(np.mean(s[zlb] - f[zlb]) * 1e4) if zlb.sum() else np.nan,
        "shadow_below_ff_in_zlb_frac": float(np.mean((s[zlb] < f[zlb]))) if zlb.sum() else np.nan,
    }
    return out
