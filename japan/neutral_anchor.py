"""Japan expected-rate NEUTRAL anchor — a per-date long-run nominal short rate n(t)
used as a *shifting endpoint* for the expected-rate path (see
``model.term_premium.expected_floored_short_rate_yields(..., endpoint=)``).

Why this exists
---------------
Japan's modern (2000+) sample is ~24 years of deflation/ZIRP, so the yields-only
sample-mean anchor implies a long-run short rate of ~0% (theta_L+theta_S). The expected
path then reverts DOWN to ~0%, which balloons the term premium (published 10y expected
~0.65% / 30y ~0.45% vs a BoJ/survey neutral near ~2%). A structural break defeats
yields-only estimation; the standard remedy (Kim-Wright/DKW, Kim-Orphanides) is to anchor
the expectations endpoint to a survey/neutral. Applied at DECOMPOSITION time, so it needs
no re-estimation — it only re-targets where the path mean-reverts, leaving the fitted
curve and the whole Q-side untouched.

The ladder itself lives in ``model.neutral_anchor`` (shared with Germany); this module
supplies Japan's files and constants. Rung names below map onto the generic ones:
``extender`` = Tankan, ``target`` = the BoJ's 2%.

Degradation ladder (graceful; inert when no files present)
----------------------------------------------------------
  1. Consensus Economics **3m-interbank 6-10y** forecast (2017+) -> used directly, within
     a staleness window of its last release.
  2. Public bridge  n = r*(Ferreira longer-run real neutral) + pi^e, where
       pi^e = Consensus CPI 6-10y (1990+), extended past its last release by Tankan 5y
              (level-bridged), else Tankan alone, else the BoJ 2% target;
     the bridge is level-calibrated to the Consensus short-rate over their overlap.
  3. Carry-forward (LOCF) between the intermittent, irregular survey releases.
  4. No usable file -> NaN everywhere -> the model falls back to constant-theta (today's
     yields-only behaviour). The pipeline never breaks if the data is unavailable.

All inputs are COMMITTED CSVs (``date,value`` in percent) under ``japan/data/anchors/`` so
the daily CI and a fresh clone reproduce the anchor with no live/proprietary access.
Consensus is proprietary: commit privately (this repo is private) and publish only the
derived premium, never the raw survey.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.neutral_anchor import (AnchorSpec, align as _align_generic, has_anchor as _has,
                                  locf as _locf, rw_smooth as _rw_smooth)
from model.neutral_anchor import build_nominal_neutral as _build

ANCHOR_DIR = Path(__file__).resolve().parent / "data" / "anchors"
BOJ_TARGET_PCT = 2.0            # last-resort inflation anchor (BoJ's 2% target)
STALENESS_DAYS = 550            # carry a Consensus short-rate release this long -> bridge
# Endpoint smoothing (random-walk local-level state-space on the intermittent survey reads).
# The surveys release ~quarterly, so a plain carry-forward makes the endpoint a staircase
# (e.g. the +42bp step on the 2024-07-08 Consensus release). SMOOTH replaces that with a
# Kalman smoother over history and the causal filter at the current edge: the interior path
# is interpolated between reads (no artificial step; single-release noise damped) while the
# right edge uses only past reads (no look-ahead), so today's value still tracks the latest
# survey. OBS_STD/DRIFT_STD are in percent; DRIFT_STD is the RW innovation per YEAR (the
# latent neutral is slow-moving) and OBS_STD the noise attributed to a single survey read.
SMOOTH = True
SMOOTH_OBS_STD = 0.15          # % — measurement noise on one survey release
SMOOTH_DRIFT_STD = 0.28        # %/yr — random-walk drift of the latent neutral
FILES = {
    "shortrate": "consensus_shortrate_6_10y.csv",   # primary nominal short-rate survey, 2017+
    "cpi":       "consensus_cpi_6_10y.csv",          # inflation survey, 1990+
    "rstar":     "ferreira_rstar_jp.csv",            # Ferreira real longer-run neutral, semi-annual
    "tankan":    "tankan_infl_5y.csv",               # BoJ Tankan 5y inflation expectation, 2014+
}


def _spec(dir: "Path | None" = None) -> AnchorSpec:
    """Build the spec from the CURRENT module-level constants, so tests that monkeypatch
    e.g. ``SMOOTH`` on this module still take effect."""
    # FILES keeps Japan's own rung name ("tankan"); the shared engine asks for the
    # generic "extender", so translate here rather than renaming the public surface.
    files = dict(FILES)
    files["extender"] = files["tankan"]
    return AnchorSpec(market="JP", dir=Path(dir) if dir is not None else ANCHOR_DIR,
                      files=files, target_pct=BOJ_TARGET_PCT,
                      staleness_days=STALENESS_DAYS, smooth=SMOOTH,
                      obs_std=SMOOTH_OBS_STD, drift_std=SMOOTH_DRIFT_STD,
                      survey_label="consensus", rstar_label="ferreira",
                      target_label="boj_target", extender_label="tankan")


def _align(s, index):
    """Kept for callers/tests that reach for it directly."""
    return _align_generic(s, index, _spec())


def build_nominal_neutral(index, dir: Path = ANCHOR_DIR):
    """Per-date nominal neutral short rate (DECIMAL) aligned to ``index``. See
    ``model.neutral_anchor.build_nominal_neutral``."""
    return _build(index, _spec(dir), dir=dir)


def has_anchor(dir: Path = ANCHOR_DIR) -> bool:
    """True if at least one usable anchor file exists."""
    return _has(_spec(dir), dir=dir)
