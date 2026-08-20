"""US expected-rate anchors — SPF **average-matched** shifting endpoints.

Every other market anchors on a 6-10y-ahead survey, which is (approximately) the
terminal object ``endpoint=`` wants. The US surveys are different objects: SPF
``BILL10``/``CPI10`` are **0-10y averages**, so substituting them for theta would
double-count the near-term path. Instead the endpoint is solved so that the model's
implied 10-year AVERAGE expected rate equals the survey's:

    avg10(t) = w * endpoint + (1 - w) * near-path(t),   w = 1 - (1 - e^{-k T})/(k T)

with ``k`` the (bias-corrected, floored) level-factor mean reversion and T = 10. The
near-path term is read off the model's own UNANCHORED decomposition, giving the closed
form  ``endpoint(t) = theta_LR + (survey(t) - avg10_unanchored(t)) / w``.  This is the
DKW/Kim-Wright use of these exact inputs: the survey disciplines the average, the
model supplies the shape. NaN before the first SPF release (1992) -> constant-theta,
so pre-survey history is untouched.

Both anchors are gated, per the pre-committed rule, in ``viz/export_daily``:
  * REAL  (BILL10 - CPI10): ships only if it improves BOTH the level gap and the
    correlation of the expected real rate against DKW's.
  * NOMINAL (BILL10): ships only if it improves both against Kim-Wright's 10y TP —
    the survey-anchored external benchmark.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model.neutral_anchor import rw_smooth

logger = logging.getLogger(__name__)

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
SMOOTH_OBS_STD = 0.15
SMOOTH_DRIFT_STD = 0.28
# Gate switches — MEASURED twice (2026-08-20, full overlaps), both rounds failed:
#
# Round 1, 0-10y average objects (BILL10 / BILL10-CPI10, avg-matched):
#   REAL vs DKW expected real:    corr 0.786 -> 0.634, |gap| 0.32 -> 0.53pp. Fail.
#   NOMINAL vs Kim-Wright TP10:   gap +0.45 -> +0.05pp but corr 0.980 -> 0.907. Fail.
#
# Round 2, terminal-consistent 2-10y FORWARD objects ((10*BILL10 - TBILLA - TBILLB)/8,
# real leg net of the same-window ATSIX forward; window-matched, w = 0.662):
#   REAL vs DKW expected real:    corr 0.786 -> 0.605, |gap| 0.32 -> 0.60pp. Fail —
#         the SPF/ATSIX real forward sits ~0.4pp ABOVE DKW's expected real path;
#         the disagreement is between the surveys and DKW, not an artifact of the
#         object or the transform.
#   NOMINAL vs Kim-Wright TP10:   gap 0.62 -> 0.45pp, mean gap +0.02 (dead on KW's
#         level), corr 0.919 — better than round 1's 0.907, still below the
#         unanchored 0.980. Fail on the corr leg.
#
# A structural observation, recorded as a hypothesis and NOT used to relax the gate:
# Kim-Wright's high-frequency variation is yields-driven, like our unanchored series,
# so the correlation leg may inherently favour "unanchored" regardless of anchor
# quality. If the gate criteria are ever changed, that is a recorded design decision
# taken in review — never a post-hoc edit to let a failed result through.
NOMINAL_ACTIVATED = False
REAL_ACTIVATED = False


def _load(name: str) -> "pd.Series | None":
    f = ANCHOR_DIR / name
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["date"])
    s = pd.Series(pd.to_numeric(df["value"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(df["date"])).dropna().sort_index()
    return s if len(s) else None


FWD_START = 2.0                          # strip the surveyed near years (TBILLA/B)
FWD_END = 10.0


def survey_forward(instrument: str, index: pd.DatetimeIndex) -> "pd.Series | None":
    """The SPF-implied **2-10y forward average** expectation in PERCENT (RW-smoothed).

    BILL10 is a 0-10y average, and matching on it forces the model's near path onto
    survey-release timing — the mechanism that failed the first gates. The SPF also
    surveys the near path itself (TBILLA/TBILLB: current- and next-year T-bill), so
    the cyclical front strips out by pure arithmetic:

        fwd_2_10 = (10*BILL10 - y1 - y2) / 8

    No model, no convergence assumption — every term is a surveyed median from the
    same round. For the real curve the inflation leg comes off the ATSIX curve
    (Aruoba, Philadelphia Fed — already the joint model's anchor): the same-window
    forward  (10*pi10 - 2*pi2)/8,  CPI-basis like TIPS, monthly 1998+."""
    bill = _load("spf_bill10.csv")
    y1, y2 = _load("spf_tbill_y1.csv"), _load("spf_tbill_y2.csv")
    if bill is None or y1 is None or y2 is None:
        return None
    grid = bill.index                                   # annual (Q1) survey rounds
    a = y1.reindex(grid).ffill(); b = y2.reindex(grid).ffill()
    fwd = ((FWD_END * bill - a - b) / (FWD_END - FWD_START)).dropna()
    if instrument == "real":
        try:
            from validation.benchmarks import load_atsix
            at = load_atsix(refresh=False)
        except Exception:                               # noqa: BLE001
            return None
        p10 = at[at.maturity == 10.0].set_index("date")["expected_inflation"]
        p2 = at[at.maturity == 2.0].set_index("date")["expected_inflation"]
        pi_fwd = ((FWD_END * p10 - FWD_START * p2) / (FWD_END - FWD_START)) * 100.0             if p10.max() < 1.0 else (FWD_END * p10 - FWD_START * p2) / (FWD_END - FWD_START)
        pi_on_grid = pi_fwd.reindex(pi_fwd.index.union(grid)).sort_index().ffill().reindex(grid)
        fwd = (fwd - pi_on_grid).dropna()
    return rw_smooth(fwd, index, SMOOTH_OBS_STD, SMOOTH_DRIFT_STD)


def survey_avg(instrument: str, index: pd.DatetimeIndex) -> "pd.Series | None":
    """The SPF 10y-average expectation in PERCENT on ``index`` (RW-smoothed):
    BILL10 for the nominal curve, BILL10 - CPI10 for the real one."""
    bill = _load("spf_bill10.csv")
    if bill is None:
        return None
    if instrument == "real":
        cpi = _load("spf_cpi10.csv")
        if cpi is None:
            return None
        # difference on the union of release dates (CPI10 is quarterly, BILL10 annual)
        grid = bill.index.union(cpi.index)
        b = bill.reindex(grid).ffill()
        c = cpi.reindex(grid).ffill()
        s = (b - c).dropna()
        s = s[s.index >= max(bill.index.min(), cpi.index.min())]
    else:
        s = bill
    return rw_smooth(s, index, SMOOTH_OBS_STD, SMOOTH_DRIFT_STD)


def endpoint_weight(K_bc: np.ndarray, a: float = 0.0, b: float = 10.0) -> float:
    """Weight of the endpoint in the model's [a,b]-year average expected rate. A level
    shift d decays at the smallest K^P eigenvalue k, so the window average moves by
    w*d with ``w = 1 - (e^{-ka} - e^{-kb})/(k(b-a))``. For [0,10] at the floored
    k=0.198 this is 0.565; for the [2,10] forward window 0.662 — the window sheds the
    most transitory stretch of the path, which is the point of matching on it."""
    k = float(np.min(np.linalg.eigvals(np.asarray(K_bc)).real))
    return 1.0 - (np.exp(-k * a) - np.exp(-k * b)) / (k * (b - a))


def avg_matched_endpoint(prod, frame_unanchored: pd.DataFrame, instrument: str,
                         index: pd.DatetimeIndex, T: float = 10.0):
    """(endpoint decimal array on ``index``, meta) or (None, None) when no survey."""
    svy = survey_forward(instrument, index)
    if svy is None or not svy.notna().any():
        return None, None
    fr = frame_unanchored
    e10 = pd.Series(fr[fr["maturity"] == FWD_END]["expected_rate"].to_numpy(),
                    index=pd.DatetimeIndex(fr[fr["maturity"] == FWD_END]["date"])).reindex(index)
    e2 = pd.Series(fr[fr["maturity"] == FWD_START]["expected_rate"].to_numpy(),
                   index=pd.DatetimeIndex(fr[fr["maturity"] == FWD_START]["date"])).reindex(index)
    model_fwd = (FWD_END * e10 - FWD_START * e2) / (FWD_END - FWD_START)
    w = endpoint_weight(prod.K_bc, FWD_START, FWD_END)
    th = prod.theta_for_expectations()
    theta_lr = float(th[0] + th[1])                                 # decimal
    ep = theta_lr + (svy / 100.0 - model_fwd) / w                   # decimal
    meta = {"source": "spf_fwd_2_10" + ("-atsix" if instrument == "real" else ""),
            "object": "2-10y forward average (window-matched)", "w": round(float(w), 4),
            "first": str(svy.dropna().index.min().date()),
            "asof": None, "coverage": int(ep.notna().sum()), "n": len(index)}
    return ep.to_numpy(dtype=float), meta
