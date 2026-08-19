"""Shared machinery for a survey-anchored **shifting endpoint** — the per-date long-run
nominal short rate ``n(t)`` the expected-rate path reverts to (Kozicki-Tinsley,
Kim-Orphanides), applied at DECOMPOSITION time via
``model.term_premium.expected_floored_short_rate_yields(..., endpoint=)``.

Why a market needs one
----------------------
A yields-only sample mean anchors the expected path at whatever the sample average
short rate happened to be. That is wrong in two distinguishable ways, and this module
addresses the second:

* a **regime break inside the sample** — fixed by re-cutting the estimation window
  (see ``validation/FINDINGS_regime_dating.md``);
* **secular drift in r\\* within one regime** — not fixed by re-cutting, because the
  sample can be regime-clean and its mean still stale. Japan's 2000+ deflation sample
  implies a ~0% long-run short rate against a survey neutral near 2%; Germany's
  regime-clean 1984+ sample implies 2.84% against a euro-area neutral nearer 2.5%.

Anchoring the endpoint to surveys re-targets only where the path mean-reverts. The
fitted curve and the entire Q-side are untouched, so no re-estimation is required.

The degradation ladder
----------------------
Built so a market can be anchored with a **one-off, unmaintainable** proprietary
survey and stay live indefinitely afterwards:

  1. a direct **6-10y-ahead short-rate survey** — used as-is, within a staleness window
     of its last release;
  2. a **public bridge** ``n = r* + pi^e``, where ``pi^e`` is a long-horizon inflation
     survey extended past its last release by a public expectations series
     (level-bridged), else that series alone, else the central bank's target. The
     bridge is level-calibrated to the direct survey over their overlap, so the two
     rungs join without a step;
  3. carry-forward between the intermittent, irregular releases;
  4. nothing usable -> NaN everywhere -> the model falls back to constant-theta, i.e.
     today's yields-only behaviour. The pipeline never breaks on missing data.

Rung 1 is typically the proprietary, short-history series; rungs 2-4 are public and
auto-refreshable. That asymmetry is the point: the expensive input can be a frozen
snapshot, and the anchor still tracks after it lapses.

All inputs are committed ``date,value`` CSVs (values in PERCENT). Proprietary surveys
belong in a private repo — publish the derived premium, never the raw survey.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorSpec:
    """One market's anchor configuration: where the files live, what they are called,
    and the constants governing staleness and endpoint smoothing."""
    market: str
    dir: Path
    files: dict                      # rung name -> filename
    target_pct: float                # last-resort inflation anchor (the CB's target)
    staleness_days: int = 550        # carry a direct-survey release this long, then bridge
    smooth: bool = True              # RW smoother vs plain carry-forward
    obs_std: float = 0.15            # % — measurement noise on one survey release
    drift_std: float = 0.28          # %/yr — RW drift of the latent neutral
    # Rung labels reported in meta. They surface in the front end (the survey-anchor
    # badge) and are asserted by tests, so each market names its own rungs.
    survey_label: str = "survey"
    shortrate_ext_label: str = "rate_ext"
    rstar_label: str = "rstar"
    target_label: str = "cb_target"
    extender_label: str = "extender"


def load_series(spec: AnchorSpec, name: str, dir: "Path | None" = None) -> "pd.Series | None":
    """Read a committed ``date,value`` CSV (percent) -> sorted Series, or None."""
    d = Path(dir) if dir is not None else spec.dir
    fn = spec.files.get(name)
    if not fn:
        return None
    f = d / fn
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
    except Exception as e:                                          # noqa: BLE001
        logger.warning("neutral-anchor[%s]: %s unreadable (%s)", spec.market, f.name, e)
        return None
    cols = {c.strip().lower(): c for c in df.columns}
    dc, vc = cols.get("date"), cols.get("value")
    if dc is None or vc is None:                                    # fall back to first two cols
        dc, vc = df.columns[0], df.columns[1]
    s = pd.Series(pd.to_numeric(df[vc], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[dc], errors="coerce"))
    s = s[s.index.notna()].dropna().sort_index()
    return s if len(s) else None


def locf(s: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Carry the last observation forward onto ``index`` (releases are intermittent).
    Dates before the first release stay NaN (the model uses constant-theta there)."""
    return s.reindex(s.index.union(index)).sort_index().ffill().reindex(index)


def rw_smooth(s: pd.Series, index: pd.DatetimeIndex,
              obs_std: float = 0.15, drift_std: float = 0.28) -> pd.Series:
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
        return locf(s, index)
    grid = obs.index.union(pd.DatetimeIndex(index)).sort_values().unique()
    grid = pd.DatetimeIndex(grid)
    y = pd.Series(np.nan, index=grid)
    y.loc[obs.index] = obs.to_numpy()
    yv = y.to_numpy()
    t_years = grid.view("int64") / 8.64e13 / 365.25                 # ns -> days -> years
    q, r = float(drift_std) ** 2, float(obs_std) ** 2
    n = len(grid)
    xp = np.empty(n); Pp = np.empty(n)                              # one-step-ahead prediction
    xf = np.empty(n); Pf = np.empty(n)                              # filtered
    x, P = float(obs.iloc[0]), r                                    # init at the first read
    prev = t_years[0]
    for i in range(n):
        dt = t_years[i] - prev if i else 0.0
        x_pred, P_pred = x, P + q * max(dt, 0.0)                    # RW predict
        xp[i], Pp[i] = x_pred, P_pred
        yi = yv[i]
        if np.isfinite(yi):                                         # update on a release
            K = P_pred / (P_pred + r)
            x, P = x_pred + K * (yi - x_pred), (1.0 - K) * P_pred
        else:
            x, P = x_pred, P_pred
        xf[i], Pf[i] = x, P
        prev = t_years[i]
    xs = xf.copy()                                                  # RTS backward smoother
    for i in range(n - 2, -1, -1):
        C = Pf[i] / Pp[i + 1] if Pp[i + 1] > 0 else 0.0
        xs[i] = xf[i] + C * (xs[i + 1] - xp[i + 1])
    out = pd.Series(xs, index=grid).reindex(pd.DatetimeIndex(index))
    out[pd.DatetimeIndex(index) < obs.index[0]] = np.nan            # keep pre-first-read NaN
    return out


def align(s: "pd.Series | None", index: pd.DatetimeIndex, spec: AnchorSpec) -> pd.Series:
    """Align an intermittent survey series onto ``index``: RW smoother when the spec asks for
    it (removes the release-date staircase), else plain carry-forward. Inert on ``None``."""
    if s is None:
        return pd.Series(np.nan, index=index)
    return (rw_smooth(s, index, spec.obs_std, spec.drift_std) if spec.smooth
            else locf(s, index))


def build_nominal_neutral(index, spec: AnchorSpec, dir: "Path | None" = None):
    """Per-date nominal neutral short rate (DECIMAL) aligned to ``index``.

    Returns ``(values, meta)``: ``values`` is an (T,) float array (NaN where undefined,
    which the model treats as no shift); ``meta`` records the active source/rung, the
    inflation leg used, the calibration offset, the as-of date, and coverage. With no
    anchor files present, returns all-NaN with ``source='none'`` (fully inert)."""
    index = pd.DatetimeIndex(pd.to_datetime(index))
    sr = load_series(spec, "shortrate", dir)
    cpi = load_series(spec, "cpi", dir)
    rstar = load_series(spec, "rstar", dir)
    ext = load_series(spec, "extender", dir)
    sr_ext = load_series(spec, "shortrate_ext", dir)

    # --- extend the DIRECT rung past its last release with a public long-horizon rate
    #     survey, level-bridged over their overlap. Same mechanism as the inflation leg,
    #     and it matters more: a proprietary short-rate survey is the rung most likely to
    #     be a frozen one-off, and without this the anchor drops to r* + pi^e the moment
    #     it lapses. The horizons usually differ (e.g. 6-10y vs 4-5y ahead), which is what
    #     the offset absorbs — so an overlap is REQUIRED, and without one the extension is
    #     refused rather than splicing two different horizons at face value.
    sr_ext_src = None
    if sr is not None and sr_ext is not None:
        span = pd.date_range(max(sr.index.min(), sr_ext.index.min()),
                             min(sr.index.max(), sr_ext.index.max()), freq="M")
        if len(span) >= 6:
            off = float((locf(sr, span) - locf(sr_ext, span)).mean())
            tail = sr_ext[sr_ext.index > sr.index.max()] + off
            if len(tail):
                sr = pd.concat([sr, tail]).sort_index()
                sr_ext_src = spec.shortrate_ext_label
        else:
            logger.warning("%s neutral anchor: %s overlaps the direct survey by %d months "
                           "(<6) — not extending; horizons differ, so the level offset "
                           "cannot be identified", spec.market, spec.shortrate_ext_label,
                           len(span))
    elif sr is None and sr_ext is not None:
        sr, sr_ext_src = sr_ext, spec.shortrate_ext_label     # public rung alone

    if sr is None and rstar is None:
        return np.full(len(index), np.nan), {"source": "none", "asof": None,
                                             "coverage": 0, "n": len(index)}

    # --- inflation leg pi^e (percent): the long survey, extended past its last release by
    #     the public extender (level-bridged); else the extender; else the CB target. ---
    if cpi is not None and ext is not None:
        # level offset over the overlap SPAN (surveys release on different days, so an
        # exact-date intersection is usually empty) — align both by LOCF on a monthly grid.
        span = pd.date_range(max(cpi.index.min(), ext.index.min()),
                             min(cpi.index.max(), ext.index.max()), freq="M")
        off = float((locf(cpi, span) - locf(ext, span)).mean()) if len(span) else 0.0
        tail = ext[ext.index > cpi.index.max()] + off
        pie = pd.concat([cpi, tail]).sort_index()
        pie_src = f"{spec.survey_label}_cpi+{spec.extender_label}_tail"
    elif cpi is not None:
        pie, pie_src = cpi, f"{spec.survey_label}_cpi"
    elif ext is not None:
        pie, pie_src = ext, spec.extender_label
    else:
        pie, pie_src = pd.Series([spec.target_pct], index=[index.min()]), spec.target_label

    # --- real leg r* (percent): the published estimate if present; else, with no r* but both
    #     a short-rate and a CPI survey, back out a CONSTANT real rate from their overlap
    #     (r ~= short-rate 6-10y - CPI 6-10y) and hold it through history. ---
    if rstar is not None:
        rr_i, rr_src = align(rstar, index, spec), spec.rstar_label
    elif sr is not None and cpi is not None:
        rspan = pd.date_range(max(sr.index.min(), cpi.index.min()),
                              min(sr.index.max(), cpi.index.max()), freq="M")
        rr_const = float((locf(sr, rspan) - locf(cpi, rspan)).mean()) if len(rspan) else np.nan
        rr_i, rr_src = pd.Series(rr_const, index=index), "implied_from_overlap"
    else:
        rr_i, rr_src = pd.Series(np.nan, index=index), "none"
    bridge = rr_i + align(pie, index, spec)

    src = pd.Series("bridge", index=index)
    c = 0.0
    if sr is not None:
        sr_i = align(sr, index, spec)
        ov = sr_i.dropna().index.intersection(bridge.dropna().index)
        c = float((sr_i[ov] - bridge[ov]).mean()) if len(ov) else 0.0
        bridge = bridge + c                                         # level-calibrate to survey
        n = bridge.copy()
        stale_cut = sr.index.max() + pd.Timedelta(days=spec.staleness_days)
        use_direct = sr_i.notna() & (index <= stale_cut)            # direct survey, in-window
        n[use_direct] = sr_i[use_direct]
        src[use_direct] = f"{spec.survey_label}_shortrate"
        source = f"{spec.survey_label}_shortrate+bridge"
    else:
        n = bridge
        source = "bridge"

    vals = (n / 100.0).to_numpy(dtype=float)                        # percent -> decimal
    # "as of" = last release of the point-in-time surveys that pin today's endpoint
    # (short-rate / CPI / extender). A published r* is a slow semi-annual longer-run
    # estimate carrying FORWARD-dated period labels, so it must not drive the as-of date;
    # use it only when it is the sole source.
    obs = [s for s in (sr, cpi, ext) if s is not None]
    asof = max([s.index.max() for s in (obs or [rstar])])
    meta = {"source": source, "pi_e": pie_src, "r_star": rr_src, "shortrate_ext": sr_ext_src,
            "calib_offset_pct": round(c, 3),
            "asof": str(pd.Timestamp(asof).date()), "coverage": int(np.isfinite(vals).sum()),
            "n": len(index), "src_by_date": src}
    logger.info("%s neutral anchor: source=%s pi_e=%s r*=%s offset=%.2f%% asof=%s coverage=%d/%d",
                spec.market, source, pie_src, rr_src, c, meta["asof"], meta["coverage"], len(index))
    return vals, meta


def has_anchor(spec: AnchorSpec, dir: "Path | None" = None) -> bool:
    """True if at least one usable anchor file (direct short-rate survey or r*) exists."""
    d = Path(dir) if dir is not None else spec.dir
    return ((d / spec.files["shortrate"]).exists()
            or (d / spec.files["rstar"]).exists())


def bridge_fidelity(spec: AnchorSpec, dir: "Path | None" = None) -> dict:
    """How well the PUBLIC bridge reproduces the direct survey over their overlap.

    This is the number that decides whether a one-off proprietary survey is safe to rely
    on. Rung 1 will inevitably go stale; when it does, everything falls through to
    ``r* + pi^e``. If the bridge tracks the survey closely over the years they overlap,
    that hand-off is trustworthy and the anchor keeps meaning what it meant. If it does
    not, the anchor silently changes character on the day the survey lapses — which is
    the failure this measures rather than assumes.

    Reports the gap AFTER the level calibration the ladder already applies, so it
    isolates co-movement (a constant offset is removed by construction, a drifting one is
    not). Returns ``{}`` when the two do not overlap.
    """
    sr = load_series(spec, "shortrate", dir)
    if sr is None:
        return {}
    idx = pd.date_range(sr.index.min(), sr.index.max(), freq="M")
    if not len(idx):
        return {}
    # rebuild the bridge alone, i.e. the spec with the direct survey withheld
    bare = AnchorSpec(**{**spec.__dict__, "files": {k: v for k, v in spec.files.items()
                                                   if k != "shortrate"}})
    vals, meta = build_nominal_neutral(idx, bare, dir=dir)
    bridge = pd.Series(vals * 100.0, index=idx)
    direct = align(sr, idx, spec)
    ov = bridge.dropna().index.intersection(direct.dropna().index)
    if len(ov) < 6:
        return {"overlap_months": int(len(ov))}
    gap = (direct[ov] - bridge[ov])
    resid = gap - gap.mean()                       # after the ladder's level calibration
    return {
        "overlap_months": int(len(ov)),
        "overlap": [str(ov.min().date()), str(ov.max().date())],
        "calib_offset_pct": round(float(gap.mean()), 3),
        "corr": (round(float(np.corrcoef(direct[ov], bridge[ov])[0, 1]), 4)
                 if float(np.std(bridge[ov])) > 1e-9 else None),
        "mean_abs_gap_bp": round(float(resid.abs().mean()) * 100.0, 1),
        "max_abs_gap_bp": round(float(resid.abs().max()) * 100.0, 1),
        "pi_e": meta.get("pi_e"), "r_star": meta.get("r_star"),
    }
