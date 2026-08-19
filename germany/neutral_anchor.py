"""Germany expected-rate NEUTRAL anchor — the euro-area long-run nominal short rate used
as a *shifting endpoint* for the German expected-rate path.

Why Germany needs one
---------------------
Germany's production sample was re-cut to the regime-clean 1984+ window (a 2-regime Kim
filter dates the permanent level break at 1983-11; see
``validation/FINDINGS_regime_dating.md``). That fixed a regime break inside the sample,
but it cannot fix the other problem: **r\\* has drifted down secularly WITHIN the
post-1984 regime**. The 1984+ sample mean implies a long-run short rate of **2.84%**,
against a euro-area neutral nearer **2.5%** (Ferreira's EA real longer-run rate is
~0.50%, plus a ~2% inflation anchor). A sample can be regime-clean and its mean still
stale, which is exactly what a shifting endpoint is for.

The ladder (implemented in ``model.neutral_anchor``; identical to Japan's)
-------------------------------------------------------------------------
  1. Consensus Economics **3m-interbank (euro area) 6-10y** forecast, 2016-10+ — used
     directly, within a staleness window of its last release.
  2. Public bridge  n = r*(Ferreira EA longer-run real neutral) + pi^e, where
       pi^e = Consensus CPI 6-10y for Germany (1990+), extended past its last release by
              the ECB Survey of Professional Forecasters' longer-term expectation
              (level-bridged), else the SPF alone, else the ECB's 2% target;
     level-calibrated to the Consensus short-rate over their overlap.
  3. Carry-forward between the intermittent, irregular releases.
  4. No usable file -> NaN everywhere -> constant-theta (today's yields-only behaviour).

**On maintainability.** The Consensus inputs are a one-off snapshot that cannot be
refreshed. That is the case the ladder was built for: rung 1 and the CPI leg of rung 2
are frozen committed CSVs, while the Ferreira r\\* and the ECB SPF beneath them are
public and auto-refreshed weekly by ``.github/workflows/refresh-anchors.yml``. When the
Consensus short-rate series passes its staleness window the anchor falls through to a
bridge that keeps tracking indefinitely, with no manual step and no silent freeze.

Ferreira publishes one file covering many economies (``EA`` sits beside ``JA``), so the
euro-area real leg costs nothing extra to maintain — it is the same weekly fetch Japan
already depends on.

Consensus is proprietary: commit privately (this repo is private) and publish only the
derived premium, never the raw survey.
"""
from __future__ import annotations

from pathlib import Path

from model.neutral_anchor import AnchorSpec
from model.neutral_anchor import align as _align_generic
from model.neutral_anchor import build_nominal_neutral as _build
from model.neutral_anchor import has_anchor as _has

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
ECB_TARGET_PCT = 2.0            # last-resort inflation anchor (the ECB's 2% target)
STALENESS_DAYS = 550            # carry a Consensus short-rate release this long -> bridge
# Endpoint smoothing — see model.neutral_anchor.rw_smooth. Same constants as Japan: both
# ladders read ~quarterly surveys, so the staircase problem and its cure are identical.
SMOOTH = True
SMOOTH_OBS_STD = 0.15          # % — measurement noise on one survey release
SMOOTH_DRIFT_STD = 0.28        # %/yr — random-walk drift of the latent neutral
FILES = {
    "shortrate": "consensus_shortrate_ea_6_10y.csv",  # EA 3m interbank 6-10y, 2016-10+
    "cpi":       "consensus_cpi_de_6_10y.csv",        # German CPI 6-10y, 1990+
    "rstar":     "ferreira_rstar_ea.csv",             # Ferreira EA real longer-run neutral
    "extender":  "ecb_spf_infl_lt.csv",               # ECB SPF longer-term inflation expectation
}


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    """Build the spec from the CURRENT module-level constants, so a test that
    monkeypatches e.g. ``SMOOTH`` on this module still takes effect."""
    return AnchorSpec(market="DE", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=FILES, target_pct=ECB_TARGET_PCT,
                      staleness_days=STALENESS_DAYS, smooth=SMOOTH,
                      obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus", rstar_label="ferreira",
                      target_label="ecb_target", extender_label="ecb_spf")


def _align(s, index):
    return _align_generic(s, index, _spec())


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR):
    """Per-date euro-area nominal neutral short rate (DECIMAL) aligned to ``index``."""
    return _build(index, _spec(dir), dir=dir)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    """True if at least one usable anchor file exists."""
    return _has(_spec(dir), dir=dir)


# The public rungs alone already build a working bridge, but only from 1999-03 (the ECB
# SPF's first round) — and the production sample starts 1984. Switching the anchor on
# with that coverage would apply a shifting endpoint to the back two-thirds of the
# sample and constant-theta to the front, a method seam at 1999 dressed up as a result.
# The Consensus CPI file carries the inflation leg back to 1990, which is what makes the
# coverage defensible, so activation waits for it. Set REQUIRE_CONSENSUS = False to run
# the public-only bridge deliberately (it is fine for research, not for publication).
REQUIRE_CONSENSUS = True


def is_enabled(dir: Path = ANCHOR_DIR) -> bool:
    """Whether the pipeline should APPLY this anchor (as opposed to merely being able to
    build one). Guards the coverage seam described above."""
    d = Path(dir)
    if not has_anchor(d):
        return False
    if REQUIRE_CONSENSUS and not (d / FILES["cpi"]).exists():
        return False
    return True
