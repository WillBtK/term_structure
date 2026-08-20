"""Switzerland expected-rate NEUTRAL anchor — survey-era shifting endpoint.

Same one-rule construction: Consensus 3m-rate 6-10y (2017+, frozen one-off snapshot),
RW-smoothed, NaN before the survey era. Switzerland currently has NO public living
carry-forward — the KOF Consensus Forecast carries the right 5y-horizon question but
its Time Series Database API v2 requires a KOF account — so past the staleness window
the endpoint FREEZES at the last smoothed value, flagged ``stale`` in the metadata.
A slow 6-10y object stays approximately right for years; the flag and the displayed
provenance are what keep the freeze honest. Wire ``shortrate_ext`` to a KOF series the
day an account exists.

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
FILES = {"shortrate": "consensus_shortrate_ch_6_10y.csv"}


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="CH", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=1.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era."""
    return _build(index, _spec(dir), dir=dir)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    return (Path(dir) / FILES["shortrate"]).exists()
