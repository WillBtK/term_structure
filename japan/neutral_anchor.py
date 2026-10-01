"""Japan expected-rate NEUTRAL anchor — survey-era shifting endpoint.

Japan's modern (2000+) sample is ~24 years of deflation/ZIRP, so the yields-only
sample mean implies a long-run short rate near 0% and the expected path reverts DOWN
to it, ballooning the term premium. The fix is a shifting endpoint applied at
DECOMPOSITION time (Kim-Orphanides; ``model.term_premium`` ``endpoint=``): no
re-estimation, Q-side untouched.

Construction (survey-only — see ``model.neutral_anchor.build_survey_neutral``)
------------------------------------------------------------------------------
  * 2017-01 .. last release: Consensus Economics 3m-rate 6-10y survey (frozen one-off
    snapshot, RW-smoothed). NaN before 2017 -> constant-theta, so published history
    before the survey era is NOT rewritten.
  * after the last release (2026-07): carried forward by the survey PASS-THROUGH rule
    shared by every market: the endpoint moves by 0.16 of the change in the JGB 5-10y
    forward (the share forecasters' long-horizon expectations absorb, pooled across
    seven surveys). Japan has no public long-horizon rate survey (ESP Forecast is
    membership-gated; the BoJ Bond Market Survey measures the 2-5y path, see
    validation/FINDINGS_anchors.md), so this replaces the earlier glide to the
    BoJ-communicated 1.75-2.25% corridor, which stays as a cross-check and as the
    fallback when no forward is supplied.

Retired from construction (files kept on disk for research): the Ferreira r* rung and
the CPI/Tankan bridge. A model-filtered r* rewrote pre-survey history with paths no
market participant priced — the same review that held Germany's activation — and the
one-rule outcome is that ALL markets anchor on surveys only, where surveys exist.

Consensus is proprietary: commit privately, publish only the derived premium.
"""
from __future__ import annotations

from pathlib import Path

from model.neutral_anchor import AnchorSpec
from model.neutral_anchor import build_survey_neutral as _build

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
STALENESS_DAYS = 550
SMOOTH = True
SMOOTH_OBS_STD = 0.15          # % — measurement noise on one survey release
SMOOTH_DRIFT_STD = 0.28        # %/yr — random-walk drift of the latent neutral
# BoJ-communicated nominal neutral corridor (user-supplied policy range; midpoint is
# the carry-forward terminal once the frozen Consensus snapshot goes stale).
TERMINAL_PCT = 2.0
TERMINAL_BAND = (1.75, 2.25)
FILES = {"shortrate": "consensus_shortrate_6_10y.csv"}   # 2017-01+, frozen snapshot


# Survey 3m rate -> this curve's short rate: the average gap between the model's shortest
# fitted tenor and the survey's instrument over 2017-26 (validation/survey_basis.py).
BASIS_PCT = -0.137


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="JP", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=2.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      basis_pct=BASIS_PCT, survey_label="consensus")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR, driver=None):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era.

    With ``driver`` (the JGB 5-10y forward, percent) the endpoint follows the same rule
    as every market: the last Consensus read carried forward by the survey pass-through
    (``model.neutral_anchor.PASS_THROUGH_BETA`` of the forward's moves). The BoJ corridor
    is then a cross-check, not an input. Without a driver it falls back to the glide."""
    if driver is not None:
        return _build(index, _spec(dir), dir=dir, driver=driver)
    return _build(index, _spec(dir), dir=dir,
                  terminal_pct=TERMINAL_PCT, terminal_band=TERMINAL_BAND)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    return (Path(dir) / FILES["shortrate"]).exists()
