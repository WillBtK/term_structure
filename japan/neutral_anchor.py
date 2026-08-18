"""Japan expected-rate NEUTRAL anchor — a per-date long-run nominal short rate n(t)
used as a *shifting endpoint* for the expected-rate path (see
``model.term_premium.expected_floored_short_rate_yields(..., endpoint=)``).

Why this exists
---------------
Japan's modern (2000+) sample is ~24 years of deflation/ZIRP, so the yields-only
sample-mean anchor implies a long-run short rate of ~0% (theta_L+theta_S). The expected
path then reverts DOWN to ~0%, which balloons the term premium (published 10y expected
~0.65% / 30y ~0.45% vs a BoJ/survey neutral near ~2%). A structural break defeats
yields-only estimation; the standard remedy (Kim-Wright/DKW, Kim-Orphanides) is to anchor
the expectations endpoint to a survey/neutral. Applied at DECOMPOSITION time, so it needs
no re-estimation — it only re-targets where the path mean-reverts, leaving the fitted
curve and the whole Q-side untouched.

Degradation ladder (graceful; inert when no files present)
----------------------------------------------------------
  1. Consensus Economics **3m-interbank 6-10y** forecast (2017+) -> used directly, within
     a staleness window of its last release.
  2. Public bridge  n = r*(Ferreira longer-run real neutral) + pi^e, where
       pi^e = Consensus CPI 6-10y (1990+), extended past its last release by Tankan 5y
              (level-bridged), else Tankan alone, else the BoJ 2% target;
     the bridge is level-calibrated to the Consensus short-rate over their overlap.
  3. Carry-forward (LOCF) between the intermittent, irregular survey releases.
  4. No usable file -> NaN everywhere -> the model falls back to constant-theta (today's
     yields-only behaviour). The pipeline never breaks if the data is unavailable.

All inputs are COMMITTED CSVs (``date,value`` in percent) under ``japan/data/anchors/`` so
the daily CI and a fresh clone reproduce the anchor with no live/proprietary access.
Consensus is proprietary: commit privately (this repo is private) and publish only the
derived premium, never the raw survey.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
BOJ_TARGET_PCT = 2.0            # last-resort inflation anchor (BoJ's 2% target)
STALENESS_DAYS = 550           # carry a Consensus short-rate release this long before -> bridge
# Endpoint smoothing (random-walk local-level state-space on the intermittent survey reads).
# The surveys release ~quarterly, so a plain carry-forward makes the endpoint a staircase
# (e.g. the +42bp step on the 2024-07-08 Consensus release). SMOOTH replaces that with a
# Kalman smoother over history and the causal filter at the current edge: the interior path
# is interpolated between reads (no artificial step; single-release noise damped) while the
# right edge uses only past reads (no look-ahead), so today's value still tracks the latest
# survey. OBS_STD/DRIFT_STD are in percent; DRIFT_STD is the RW innovation per YEAR (the
# latent neutral is slow-moving) and OBS_STD the noise attributed to a single survey read.
SMOOTH = True
SMOOTH_OBS_STD = 0.15          # % — measurement noise on one survey release
SMOOTH_DRIFT_STD = 0.28        # %/yr — random-walk drift of the latent neutral
FILES = {
    "shortrate": "consensus_shortrate_6_10y.csv",   # primary nominal short-rate survey, 2017+
    "cpi":       "consensus_cpi_6_10y.csv",          # inflation survey, 1990+
    "rstar":     "ferreira_rstar_jp.csv",            # Ferreira real longer-run neutral, semi-annual
    "tankan":    "tankan_infl_5y.csv",               # BoJ Tankan 5y inflation expectation, 2014+
}


def _load(name: str, d: Path) -> "pd.Series | None":
    """Read a committed ``date,value`` CSV (value in percent) -> sorted Series, or None."""
    f = d / FILES[name]
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
    except Exception as e:                                          # noqa: BLE001
        logger.warning("neutral-anchor: %s unreadable (%s)", f.name, e)
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    dc, vc = cols.get("date"), cols.get("value")
    if dc is None or vc is None:                                   # fall back to first two cols
        dc, vc = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[vc], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[dc], errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    return s if len(s) else None


def _locf(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Carry the last observation forward onto ``index`` (releases are intermittent).
    Dates before the first release stay NaN (the model uses constant-theta there)."""
    return s.reindex(s.index.union(index)).sort_index().ffill().reindex(index)


def _rw_smooth(s: pd.Series, index: pd.DatetimeIndex,
               obs_std: float = SMOOTH_OBS_STD, drift_std: float = SMOOTH_DRIFT_STD) -> pd.Series:
    """Random-walk local-level Kalman filter + RTS smoother of an intermittent survey series
    onto ``index``. The latent neutral follows ``x_t = x_{t-1} + w`` (``Var(w)=drift_std**2``
    per YEAR); each release is a noisy read ``y = x + v`` (``Var(v)=obs_std**2``). Returns the
    SMOOTHED mean over the interior (interpolating between reads, damping single-release noise)
    and, because the RTS recursion adds no information beyond the last read, the CAUSAL FILTER
    value at/after the last release — so the current edge uses only past reads (no look-ahead)
    and still tracks the latest survey. Dates before the first release stay NaN (constant-theta).

    Falls back to plain carry-forward for a degenerate series (<2 reads)."""
    obs = s.dropna().sort_index()
    if len(obs) < 2:
        return _locf(s, index)
    grid = obs.index.union(pd.DatetimeIndex(index)).sort_values().unique()
    grid = pd.DatetimeIndex(grid)
    y = pd.Series(np.nan, index=grid)
    y.loc[obs.index] = obs.to_numpy()
    yv = y.to_numpy()
    t_years = grid.view("int64") / 8.64e13 / 365.25          # ns -> days -> years
    q, r = float(drift_std) ** 2, float(obs_std) ** 2
    n = len(grid)
    xp = np.empty(n); Pp = np.empty(n)                        # one-step-ahead prediction
    xf = np.empty(n); Pf = np.empty(n)                        # filtered
    x, P = float(obs.iloc[0]), r                             # init at the first read
    prev = t_years[0]
    for i in range(n):
        dt = t_years[i] - prev if i else 0.0
        x_pred, P_pred = x, P + q * max(dt, 0.0)             # RW predict
        xp[i], Pp[i] = x_pred, P_pred
        yi = yv[i]
        if np.isfinite(yi):                                  # update on a release
            K = P_pred / (P_pred + r)
            x, P = x_pred + K * (yi - x_pred), (1.0 - K) * P_pred
        else:
            x, P = x_pred, P_pred
        xf[i], Pf[i] = x, P
        prev = t_years[i]
    xs = xf.copy()                                           # RTS backward smoother
    for i in range(n - 2, -1, -1):
        C = Pf[i] / Pp[i + 1] if Pp[i + 1] > 0 else 0.0
        xs[i] = xf[i] + C * (xs[i + 1] - xp[i + 1])
    out = pd.Series(xs, index=grid).reindex(pd.DatetimeIndex(index))
    out[pd.DatetimeIndex(index) < obs.index[0]] = np.nan     # keep pre-first-read NaN
    return out


def _align(s: "pd.Series | None", index: pd.DatetimeIndex) -> pd.Series:
    """Align an intermittent survey series onto ``index``: RW smoother when ``SMOOTH`` (removes
    the release-date staircase), else plain carry-forward. Inert on ``None`` (all-NaN)."""
    if s is None:
        return pd.Series(np.nan, index=index)
    return _rw_smooth(s, index) if SMOOTH else _locf(s, index)


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR):
    """Per-date nominal neutral short rate (DECIMAL) aligned to ``index``.

    Returns ``(values, meta)``: ``values`` is an (T,) float array (NaN where undefined,
    which the model treats as no shift); ``meta`` records the active source/rung, the
    inflation leg used, the calibration offset, the as-of date, and coverage. When no
    anchor files exist, returns all-NaN with ``source='none'`` (fully inert)."""
    index = pd.DatetimeIndex(pd.to_datetime(index))
    sr = _load("shortrate", dir)
    cpi = _load("cpi", dir)
    rstar = _load("rstar", dir)
    tankan = _load("tankan", dir)

    if sr is None and rstar is None:
        return np.full(len(index), np.nan), {"source": "none", "asof": None,
                                             "coverage": 0, "n": len(index)}

    # --- inflation leg pi^e (percent): Consensus CPI, extended past its last release by
    #     Tankan (level-bridged); else Tankan; else the BoJ target. ---
    if cpi is not None and tankan is not None:
        # level offset over the overlap SPAN (surveys release on different days, so an
        # exact-date intersection is usually empty) — align both by LOCF on a monthly grid.
        span = pd.date_range(max(cpi.index.min(), tankan.index.min()),
                             min(cpi.index.max(), tankan.index.max()), freq="M")
        off = float((_locf(cpi, span) - _locf(tankan, span)).mean()) if len(span) else 0.0
        tail = tankan[tankan.index > cpi.index.max()] + off
        pie = pd.concat([cpi, tail]).sort_index()
        pie_src = "consensus_cpi+tankan_tail"
    elif cpi is not None:
        pie, pie_src = cpi, "consensus_cpi"
    elif tankan is not None:
        pie, pie_src = tankan, "tankan"
    else:
        pie, pie_src = pd.Series([BOJ_TARGET_PCT], index=[index.min()]), "boj_target"

    # --- real leg r* (percent): Ferreira if present; else, with no Ferreira but both a
    #     short-rate and a CPI survey, back out a CONSTANT real rate from their overlap
    #     (r ~= 3m-interbank 6-10y - CPI 6-10y) and hold it through history. ---
    if rstar is not None:
        rr_i, rr_src = _align(rstar, index), "ferreira"
    elif sr is not None and cpi is not None:
        rspan = pd.date_range(max(sr.index.min(), cpi.index.min()),
                              min(sr.index.max(), cpi.index.max()), freq="M")
        rr_const = float((_locf(sr, rspan) - _locf(cpi, rspan)).mean()) if len(rspan) else np.nan
        rr_i, rr_src = pd.Series(rr_const, index=index), "implied_from_overlap"
    else:
        rr_i, rr_src = pd.Series(np.nan, index=index), "none"
    bridge = rr_i + _align(pie, index)

    src = pd.Series("bridge", index=index)
    c = 0.0
    if sr is not None:
        sr_i = _align(sr, index)
        ov = sr_i.dropna().index.intersection(bridge.dropna().index)
        c = float((sr_i[ov] - bridge[ov]).mean()) if len(ov) else 0.0
        bridge = bridge + c                                         # level-calibrate bridge to survey
        n = bridge.copy()
        stale_cut = sr.index.max() + pd.Timedelta(days=STALENESS_DAYS)
        use_direct = sr_i.notna() & (index <= stale_cut)           # direct survey, within staleness
        n[use_direct] = sr_i[use_direct]
        src[use_direct] = "consensus_shortrate"
        source = "consensus_shortrate+bridge"
    else:
        n = bridge
        source = "bridge"

    vals = (n / 100.0).to_numpy(dtype=float)                        # percent -> decimal
    # "as of" = last release of the point-in-time surveys that pin today's endpoint
    # (short-rate / CPI / Tankan). Ferreira r* is a slow semi-annual longer-run neutral
    # published with FORWARD-dated half-year labels (e.g. a 2026H2 row), so it must not
    # drive the as-of date; use it only when it is the sole source.
    obs = [s for s in (sr, cpi, tankan) if s is not None]
    asof = max([s.index.max() for s in (obs or [rstar])])
    meta = {"source": source, "pi_e": pie_src, "r_star": rr_src, "calib_offset_pct": round(c, 3),
            "asof": str(pd.Timestamp(asof).date()), "coverage": int(np.isfinite(vals).sum()),
            "n": len(index), "src_by_date": src}
    logger.info("JP neutral anchor: source=%s pi_e=%s r*=%s offset=%.2f%% asof=%s coverage=%d/%d",
                source, pie_src, rr_src, c, meta["asof"], meta["coverage"], len(index))
    return vals, meta


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    """True if at least one usable anchor file (short-rate survey or Ferreira r*) exists."""
    return (dir / FILES["shortrate"]).exists() or (dir / FILES["rstar"]).exists()
