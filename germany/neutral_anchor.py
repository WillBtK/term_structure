"""Germany expected-rate NEUTRAL anchor — survey-era shifting endpoint.

Germany's regime-clean 1984+ sample mean implies a 2.84% long-run short rate against
a surveyed euro-area neutral near 2.1-2.2%. Re-cutting the sample fixed the regime
break inside it; it cannot fix r* drifting secularly WITHIN the regime — that is what
the shifting endpoint is for, applied at decomposition time (no re-estimation).

Construction (survey-only — see ``model.neutral_anchor.build_survey_neutral``)
------------------------------------------------------------------------------
  * 2016-10 .. last release: Consensus Economics euro-area 3m-rate 6-10y survey
    (frozen one-off snapshot, RW-smoothed). NaN before 2016 -> constant-theta:
    pre-survey history is NOT rewritten.
  * past the snapshot: the **ECB SPF longer-term rate assumption** (2025-Q1+, public,
    auto-refreshed weekly) carries the endpoint forward, level-bridged over the
    overlap — measured at a -0.01pp mean gap, so the hand-off is near-seamless.

Retired from construction (activation review, 2026-08-20): the Ferreira r* bridge.
It read -0.87% for the euro area in mid-2008, implying a ~0.7% nominal neutral while
the ECB hiked to 4.25% — worth +1.16pp of 2008 term premium on its own. Files stay on
disk for research; no published series uses them.

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
    "shortrate":     "consensus_shortrate_ea_6_10y.csv",  # EA 3m 6-10y, 2016-10+, frozen
    "shortrate_ext": "ecb_spf_rate_lt.csv",               # ECB SPF longer-term rate, 2025+
}

# The activation decision (survey-era-only construction) was taken 2026-08-20 after
# the Ferreira review; the flag stays so a future re-design is again a decision, not
# a data side effect.
ACTIVATED = True


# Survey 3m rate -> this curve's short rate: the average gap between the model's shortest
# fitted tenor and the survey's instrument over 2017-26 (validation/survey_basis.py).
BASIS_PCT = -0.242


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="DE", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=2.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      basis_pct=BASIS_PCT, survey_label="consensus", shortrate_ext_label="ecb_spf_rate")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR, driver=None):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era. ``driver`` is
    the market 5-10y forward (percent) for the survey pass-through."""
    return _build(index, _spec(dir), dir=dir, driver=driver)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    d = Path(dir)
    return any((d / f).exists() for f in FILES.values())


def is_enabled(dir: Path = ANCHOR_DIR) -> bool:
    """Whether the pipeline should APPLY this anchor."""
    return ACTIVATED and (Path(dir) / FILES["shortrate"]).exists()
