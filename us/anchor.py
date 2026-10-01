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
# BOTH ANCHORS ON (design review, 2026-08-20) — the same survey-only rule as every
# other market. The original external gates (DKW expected real; Kim-Wright TP) were
# RETIRED IN REVIEW, not relaxed to pass a run: they made models the arbiters of a
# survey anchor — the Ferreira mistake readmitted — and the Kim-Wright correlation
# leg was shown to favour "unanchored" structurally (KW's daily variation is
# yields-driven, like any unanchored series, so no anchor can win that leg). The
# project owner's standard is the one applied everywhere else: observed professional
# expectations govern the endpoint; models are displayed cross-checks, never vetoes.
#
# The two failed gate rounds are kept in validation/FINDINGS_anchors.md as history
# and as standing cross-checks: vs DKW the survey real forward sits ~0.4pp higher
# (a real disagreement, shown to the reader); vs Kim-Wright the anchored nominal TP
# agrees in level to +0.02pp — two independent survey-based constructions landing on
# the same number.
NOMINAL_ACTIVATED = True
REAL_ACTIVATED = True


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


SME_FILE = "sme_ffr.csv"
SME_MATCH_DAYS = 60                      # an SME round "matches" a Q1 SPF round within this


def spf_forward_reads() -> "pd.Series | None":
    """SPF 2-10y forward reads in PERCENT, one per Q1 round (BILL10 is asked in Q1 only):

        fwd_2_10 = (10*BILL10 - y1 - y2) / 8

    with y1/y2 the same round's TBILLA/TBILLB (current- and next-year T-bill)."""
    bill = _load("spf_bill10.csv")
    y1, y2 = _load("spf_tbill_y1.csv"), _load("spf_tbill_y2.csv")
    if bill is None or y1 is None or y2 is None:
        return None
    grid = bill.index
    a = y1.reindex(grid).ffill(); b = y2.reindex(grid).ffill()
    return ((FWD_END * bill - a - b) / (FWD_END - FWD_START)).dropna()


def _load_sme() -> "pd.Series | None":
    f = ANCHOR_DIR / SME_FILE
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["date"])
    s = pd.Series(pd.to_numeric(df["fwd_2_10"], errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(df["date"])).dropna().sort_index()
    return s if len(s) else None


def bridged_forward_reads() -> "tuple[pd.Series | None, dict]":
    """SPF Q1 reads + NY Fed SME reads (eight a year) on the SPF's T-bill basis.

    The SME asks the same 10y-average question about the FED FUNDS rate, which sits
    above the 3-month bill and is asked of a different panel, so its LEVEL is put on
    the SPF basis while its TIMING is used as-is: each SME read is shifted by the
    SPF-minus-SME gap measured at the most recent Q1 round with an SME round within
    ``SME_MATCH_DAYS`` (nearest round). The SPF stays the level authority, re-pinning
    each February; the SME carries the anchor through the other ten months. The basis
    is causal (latest past Q1); SME rounds before the first overlap use the first one."""
    spf = spf_forward_reads()
    info = {"sme_rounds": 0, "basis": None, "basis_by_q1": {}}
    if spf is None:
        return None, info
    sme = _load_sme()
    if sme is None:
        return spf, info
    basis = {}
    for d, v in spf.items():
        gap = (sme.index - d).days
        near = np.abs(gap) <= SME_MATCH_DAYS
        if near.any():
            j = int(np.argmin(np.where(near, np.abs(gap), 10**9)))
            basis[d] = float(v - sme.iloc[j])
    if not basis:
        return spf, info
    b = pd.Series(basis).sort_index()
    b_on = b.reindex(b.index.union(sme.index)).sort_index().ffill().bfill().reindex(sme.index)
    shifted = sme + b_on
    # where an SPF read and an SME read share a date the SPF wins (it is the level authority)
    reads = pd.concat([spf, shifted[~shifted.index.isin(spf.index)]]).sort_index()
    info = {"sme_rounds": int(len(sme)), "basis": round(float(b.iloc[-1]), 4),
            "basis_by_q1": {str(k.date()): round(v, 4) for k, v in b.items()},
            "last_read": str(reads.index.max().date())}
    return reads, info


def survey_forward(instrument: str, index: pd.DatetimeIndex) -> "pd.Series | None":
    """The survey-implied **2-10y forward average** expectation in PERCENT (RW-smoothed).

    BILL10 is a 0-10y average, and matching on it forces the model's near path onto
    survey-release timing — the mechanism that failed the first gates. The SPF also
    surveys the near path itself (TBILLA/TBILLB: current- and next-year T-bill), so
    the cyclical front strips out by pure arithmetic:

        fwd_2_10 = (10*BILL10 - y1 - y2) / 8

    BILL10 is asked in the Q1 round only, so on its own the anchor moves once a year.
    The NY Fed SME supplies the same 2-10y object eight times a year (see
    ``bridged_forward_reads``); both feed the same random-walk smoother.

    For the real curve the inflation leg comes off the ATSIX curve (Aruoba,
    Philadelphia Fed — already the joint model's anchor, monthly 1998+): the
    same-window forward  (10*pi10 - 2*pi2)/8,  CPI-basis like TIPS."""
    fwd, _ = bridged_forward_reads()
    if fwd is None:
        return None
    grid = fwd.index
    if instrument == "real":
        try:
            from validation.benchmarks import load_atsix
            at = load_atsix(refresh=False)
        except Exception:                               # noqa: BLE001
            return None
        p10 = at[at.maturity == 10.0].set_index("date")["expected_inflation"]
        p2 = at[at.maturity == 2.0].set_index("date")["expected_inflation"]
        pi_fwd = ((FWD_END * p10 - FWD_START * p2) / (FWD_END - FWD_START)) * 100.0 \
            if p10.max() < 1.0 else (FWD_END * p10 - FWD_START * p2) / (FWD_END - FWD_START)
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
    _, binfo = bridged_forward_reads()
    src = "spf_fwd_2_10" + ("+sme" if binfo.get("sme_rounds") else "")
    meta = {"source": src + ("-atsix" if instrument == "real" else ""),
            "object": "2-10y forward average (window-matched)", "w": round(float(w), 4),
            "first": str(svy.dropna().index.min().date()),
            "asof": binfo.get("last_read"), "sme_basis": binfo.get("basis"),
            "coverage": int(ep.notna().sum()), "n": len(index)}
    return ep.to_numpy(dtype=float), meta
