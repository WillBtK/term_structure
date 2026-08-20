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
# Gate switches — set from the MEASURED gate results (2026-08-20, full overlap):
#
#   REAL  vs DKW expected real (10y):  corr 0.786 -> 0.634, mean|gap| 0.32 -> 0.53pp.
#         Worse on both legs. FAILED — off.
#   NOMINAL vs Kim-Wright 10y TP:      mean gap +0.45 -> +0.05pp (level lands on KW),
#         but corr 0.980 -> 0.907. The pre-committed rule was "ships only if it
#         improves BOTH", and a split result is a fail; the correlation loss says the
#         avg-matching transform injects survey-release timing into a daily series.
#         FAILED — off.
#
# The machinery stays: these are one-flag switches so a future gate re-run (e.g. with
# a terminal-consistent 6-10y survey object instead of the 0-10y averages) is a
# measurement, not a rebuild. Until one passes, the US publishes yields-only on both
# curves, with the SPF cross-check reported in the FINDINGS rather than imposed.
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


def endpoint_weight(K_bc: np.ndarray, T: float = 10.0) -> float:
    """Weight of the endpoint in the model's T-year average expected rate. Under the
    (diagonal) independent-factor AFNS the level shift decays at the smallest K^P
    eigenvalue, so a shift d moves the T-average by w*d with
    ``w = 1 - (1 - e^{-kT})/(kT)``."""
    k = float(np.min(np.linalg.eigvals(np.asarray(K_bc)).real))
    return 1.0 - (1.0 - np.exp(-k * T)) / (k * T)


def avg_matched_endpoint(prod, frame_unanchored: pd.DataFrame, instrument: str,
                         index: pd.DatetimeIndex, T: float = 10.0):
    """(endpoint decimal array on ``index``, meta) or (None, None) when no survey."""
    svy = survey_avg(instrument, index)
    if svy is None or not svy.notna().any():
        return None, None
    f10 = frame_unanchored[frame_unanchored["maturity"] == T]
    avg10 = pd.Series(f10["expected_rate"].to_numpy(),
                      index=pd.DatetimeIndex(f10["date"])).reindex(index)
    w = endpoint_weight(prod.K_bc, T)
    th = prod.theta_for_expectations()
    theta_lr = float(th[0] + th[1])                                 # decimal
    ep = theta_lr + (svy / 100.0 - avg10) / w                       # decimal
    meta = {"source": "spf_bill10" + ("-cpi10" if instrument == "real" else ""),
            "object": "10y-average (avg-matched)", "w": round(float(w), 4),
            "first": str(svy.dropna().index.min().date()),
            "asof": None, "coverage": int(ep.notna().sum()), "n": len(index)}
    return ep.to_numpy(dtype=float), meta
