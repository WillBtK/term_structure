"""Validation for the US shadow-rate AFNS (Christensen-Rudebusch).

Per the top-level README, the shadow-rate model should:
- collapse to the plain AFNS **away from the bound** (the ELB "wedge" -> 0), and
- **diverge** during the ZLB/negative-rate window (2008-2015, 2020-2021) — that
  divergence is expected, not a bug.

External anchors: (1) the effective federal funds rate from FRED — the shadow
short rate should track the policy rate when positive and fall **below** it
(negative) during the ZLB; (2) the Wu-Xia shadow federal funds rate from the
Atlanta Fed (`load_wu_xia`), a directional plausibility check on level and
turning points over its ZLB coverage (different model, so not a target to match
exactly — per the README). Krippner (ljkmfa.com) remains egress-blocked.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd

FEDFUNDS_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
WUXIA_XLSX = (
    "https://www.atlantafed.org/-/media/Project/Atlanta/FRBA/Documents/datafiles/"
    "cqer/research/wu-xia-shadow-federal-funds-rate/WuXiaShadowRate.xlsx"
)
# Krippner SSR estimates. The filename carries the update date; override via
# KRIPPNER_XLSX if it moves (discoverable from https://www.ljkmfa.com/visitors/).
import os as _os
KRIPPNER_XLSX = _os.environ.get(
    "KRIPPNER_XLSX",
    "https://www.ljkmfa.com/wp-content/uploads/2021/08/SSR_Estimates_20260727.xlsx",
)

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


def load_wu_xia() -> pd.Series:
    """Wu-Xia shadow federal funds rate (decimal, monthly) from the Atlanta Fed.

    Populated only during the ZLB episodes (NaN otherwise). Requires
    `www.atlantafed.org` on the egress allowlist. Suspended by the Atlanta Fed in
    2022 (covers through ~Aug 2022).
    """
    import requests

    content = requests.get(WUXIA_XLSX, timeout=90).content
    df = pd.read_excel(io.BytesIO(content), sheet_name="Data")
    df = df.rename(columns={df.columns[0]: "date", df.columns[2]: "wuxia"})
    df["date"] = pd.to_datetime(df["date"])
    df["wuxia"] = pd.to_numeric(df["wuxia"], errors="coerce") / 100.0
    return df.dropna(subset=["wuxia"]).set_index("date")["wuxia"]


def compare_to_wu_xia(shadow_df: pd.DataFrame, wuxia: pd.Series) -> dict:
    """Directional cross-check vs the Wu-Xia shadow rate over its (ZLB) coverage.

    Per the README, methodology differs (different model), so this is a
    plausibility check on **level and turning points**, not a target to match.
    """
    s = shadow_df.set_index("date")["shadow_short_rate"]
    # align to month to be robust to day-of-month differences
    s.index = s.index.to_period("M"); wuxia.index = wuxia.index.to_period("M")
    idx = s.index.intersection(wuxia.index)
    if len(idx) < 3:
        return {"n": len(idx), "note": "insufficient overlap"}
    a, b = s.loc[idx].to_numpy(), wuxia.loc[idx].to_numpy()
    return {
        "n": int(len(idx)),
        "overlap": [str(idx.min()), str(idx.max())],
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "mean_gap_bp": float(np.mean(a - b) * 1e4),   # ours - Wu-Xia
        "ours_min": float(a.min()), "wuxia_min": float(b.min()),
    }


def load_krippner_ssr() -> pd.Series:
    """Krippner US shadow short rate (decimal, month-end) from ljkmfa.com.

    Full series (positive when rates are, negative at the ZLB), still maintained.
    Requires `www.ljkmfa.com` on the egress allowlist.
    """
    import requests

    # ljkmfa.com's WAF returns 406 without browser-like headers.
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    content = requests.get(KRIPPNER_XLSX, timeout=120, headers=headers).content
    raw = pd.read_excel(io.BytesIO(content), sheet_name="D. Month-end SSR series",
                        header=None, engine="openpyxl")
    hdr = raw.index[raw.iloc[:, 2].astype(str).str.strip() == "US SSR"]
    if len(hdr) == 0:
        raise ValueError("Could not find the 'US SSR' header in the Krippner sheet.")
    h = hdr[0]
    data = raw.iloc[h + 1:, [1, 2]].copy()
    data.columns = ["date", "us_ssr"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["us_ssr"] = pd.to_numeric(data["us_ssr"], errors="coerce") / 100.0
    return data.dropna().set_index("date")["us_ssr"]


def compare_to_krippner(shadow_df: pd.DataFrame, krippner: pd.Series) -> dict:
    """Directional cross-check vs Krippner's US SSR (full sample).

    Krippner's is a different (K-ANSM) model; per the README this is a
    plausibility check on level and turning points, not a target to match.
    """
    s = shadow_df.set_index("date")["shadow_short_rate"].copy()
    s.index = s.index.to_period("M")
    k = krippner.copy(); k.index = k.index.to_period("M")
    idx = s.index.intersection(k.index)
    if len(idx) < 3:
        return {"n": len(idx), "note": "insufficient overlap"}
    a, b = s.loc[idx].to_numpy(), k.loc[idx].to_numpy()
    return {
        "n": int(len(idx)),
        "overlap": [str(idx.min()), str(idx.max())],
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "mean_gap_bp": float(np.mean(a - b) * 1e4),
        "ours_min": float(a.min()), "krippner_min": float(b.min()),
    }


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
    s = shadow_df.set_index("date")["shadow_short_rate"].copy()
    f = ff.copy()
    # Align on calendar month (our dates are month-end, FRED's are month-start).
    s.index = s.index.to_period("M")
    f.index = f.index.to_period("M")
    idx = s.index.intersection(f.index)
    s, f = s.loc[idx], f.loc[idx]
    zlb = _in_windows(idx.to_timestamp(), ZLB_WINDOWS)
    normal = ~zlb
    out = {
        "corr_normal": float(np.corrcoef(s[normal], f[normal])[0, 1]) if normal.sum() > 2 else np.nan,
        "mean_gap_bp_normal": float(np.mean(s[normal] - f[normal]) * 1e4),
        "mean_gap_bp_zlb": float(np.mean(s[zlb] - f[zlb]) * 1e4) if zlb.sum() else np.nan,
        "shadow_below_ff_in_zlb_frac": float(np.mean((s[zlb] < f[zlb]))) if zlb.sum() else np.nan,
    }
    return out
