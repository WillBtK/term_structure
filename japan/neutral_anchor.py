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
  * beyond the staleness window: the endpoint GLIDES to the **BoJ-communicated
    nominal neutral corridor, 1.75-2.25% (midpoint 2.0%)** — a policy-communication
    number, not a model estimate. Japan has no public long-horizon rate survey (the
    ESP Forecast is membership-gated), so this is the honest carry-forward until one
    exists; the glide is implemented by handing the smoother the terminal as one more
    observation at the staleness cut, so nothing steps on the day the survey lapses.

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


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    return AnchorSpec(market="JP", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=2.0, staleness_days=STALENESS_DAYS,
                      smooth=SMOOTH, obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus")


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR):
    """Per-date neutral short rate (DECIMAL); NaN before the survey era."""
    return _build(index, _spec(dir), dir=dir,
                  terminal_pct=TERMINAL_PCT, terminal_band=TERMINAL_BAND)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    return (Path(dir) / FILES["shortrate"]).exists()
