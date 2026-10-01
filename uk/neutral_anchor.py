"""UK expected-rate NEUTRAL anchor — survey-era shifting endpoint.

Same construction as every other market (one rule): the endpoint exists exactly where
long-horizon rate SURVEYS exist — Consensus 6-10y (2017 to Jul-2026, a frozen one-off
snapshot that is never refreshed) carried forward by the BoE Market Participants
Survey's neutral-rate question (interquartile mean; public, refreshed weekly),
level-bridged over their overlap (Jun-2023 onward). NaN before the
survey era, so pre-2017 history keeps the yields-only constant-theta decomposition.
No model r* anywhere in the construction. See model.neutral_anchor.build_survey_neutral.

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
    "shortrate":     "consensus_shortrate_uk_6_10y.csv",
    "shortrate_ext": "boe_mps_neutral.csv",
}


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="UK", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=2.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus", shortrate_ext_label="boe_mps")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR, driver=None):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era. ``driver`` is
    the market 5-10y forward (percent) for the survey pass-through."""
    return _build(index, _spec(dir), dir=dir, driver=driver)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    d = Path(dir)
    return any((d / f).exists() for f in FILES.values())
