"""Switzerland expected-rate NEUTRAL anchor — survey-era shifting endpoint.

Same one-rule construction: Consensus 3m-rate 6-10y (2017+, frozen one-off snapshot),
RW-smoothed, NaN before the survey era.

The carry-forward (decision recorded 2026-08-20): **live KOF inflation + a constant
real component.** KOF's Consensus Forecast has no long-horizon RATE question (its
5-year block covers CPI/GDP/unemployment only), so the living series here is the KOF
CPI 5y-ahead median (public API, no account needed, quarterly 2015+, auto-refreshed
weekly by ``switzerland/fetch_neutral.py``). It rides through the standard
``shortrate_ext`` level bridge: the offset estimated over the ~9-year overlap with
the Consensus rate survey IS the constant real component (plus the 5y-vs-6-10y
horizon wedge), measured from surveys only — no model r* anywhere. Once KOF releases
past the frozen snapshot, the endpoint follows ``KOF pi_5y + const``; the known
residual risk, documented rather than modelled, is drift in the true real component
away from the calibration decade. Cross-check at the 2026 handover: KOF pi_5y 0.90%
vs Consensus 6-10y CPI 0.97%.

Consensus is proprietary: commit privately, publish only the derived premium.
"""
from __future__ import annotations

from pathlib import Path

from model.neutral_anchor import AnchorSpec
from model.neutral_anchor import build_survey_neutral as _build

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
STALENESS_DAYS = 550
SMOOTH = True
SMOOTH_OBS_STD = 0.15
SMOOTH_DRIFT_STD = 0.28
FILES = {
    "shortrate":     "consensus_shortrate_ch_6_10y.csv",  # 2017-01+, frozen snapshot
    "shortrate_ext": "kof_cpi_5y.csv",                    # KOF pi_5y, living (see above)
}


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="CH", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=1.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus",
                      shortrate_ext_label="kof_cpi5y+const_real")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR, driver=None):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era. ``driver`` is
    the market 5-10y forward (percent) for the survey pass-through."""
    return _build(index, _spec(dir), dir=dir, driver=driver)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    d = Path(dir)
    return any((d / f).exists() for f in FILES.values())
