"""Okun's "mirage of steady inflation": mapping the LEVEL of core inflation to its
UNCERTAINTY.

Okun (1971, *Brookings Papers on Economic Activity*) argued that steady inflation is a
mirage — as inflation rises, it also becomes more variable and less predictable. Ball
(1992, *JME*) supplied the mechanism: at high inflation, agents become uncertain
whether and when policymakers will attempt disinflation, so the level mechanically
breeds uncertainty about the future path.

Why this module exists. A scenario framework that forecasts only growth, inflation and
policy rates has no direct handle on inflation *uncertainty* — yet uncertainty, not
level, is what a term premium compensates. (Brixton et al., 2023, *JPM*: the
stock-bond correlation "depends not on the level of inflation, but on the relative
volatility of growth and inflation and the correlation between them".) Okun closes the
gap: uncertainty becomes a function of the level, which the scenario already fixes.

The estimated shape is a HOCKEY STICK, not a smooth convexity — see
``validation/FINDINGS_okun.md``. Below roughly 1-1.5% core inflation, volatility sits
on a floor and is insensitive to the level; above it, volatility rises roughly linearly.
That is economically coherent: while inflation is at or below target the anchor holds
and uncertainty is irreducible-but-small, and it is credibility erosion above target
that generates uncertainty. It also matters for scenario work, because it says moving
from 0% to 2% inflation earns almost no extra inflation risk premium while moving from
2% to 4% earns a lot.

Everything here is pure: no I/O, no network. The empirical driver is
``validation/okun_mirage.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Default trailing window for the "regime" level/vol pair, in months. Ten years is
# long enough that the pair describes a monetary regime rather than a business cycle,
# which is the object a terminal-state scenario needs. `validation/okun_mirage.py`
# reports the 5y window as a sensitivity; the shape survives, the floor sits lower.
DEFAULT_WINDOW = 120


def rolling_level_vol(inflation: pd.Series, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """Trailing mean (`level`) and standard deviation (`vol`) of year-on-year inflation.

    ``inflation`` is year-on-year core inflation in PERCENT. Both outputs are in
    percentage points, so a level of 3.0 and a vol of 1.5 read as "3% average inflation
    with a 1.5pp standard deviation around it".
    """
    s = pd.Series(inflation).dropna()
    out = pd.DataFrame({"level": s.rolling(window).mean(),
                        "vol": s.rolling(window).std()})
    return out.dropna()


@dataclass(frozen=True)
class OkunParams:
    """A fitted level -> volatility map: ``vol = floor + slope * max(level - kink, 0)``."""

    floor: float          # irreducible inflation uncertainty at/below the kink, in pp
    slope: float          # extra pp of vol per pp of inflation above the kink
    kink: float           # inflation level (%) at which uncertainty starts to scale
    r2: float = float("nan")
    n: int = 0

    def sigma(self, level):
        """Project inflation uncertainty for a forecast inflation ``level`` (percent).

        Accepts a scalar or an array, so a whole scenario path maps in one call.
        """
        lv = np.asarray(level, dtype=float)
        out = self.floor + self.slope * np.maximum(lv - self.kink, 0.0)
        return float(out) if np.isscalar(level) or out.ndim == 0 else out

    def anchored_to(self, observed_vol: float, at_level: float) -> "OkunParams":
        """Shift the floor so the map reproduces an OBSERVED uncertainty at today's level.

        The estimated map is a cross-market average and will not sit exactly on any one
        market today. Anchoring re-levels it to a directly observed reading (e.g. the
        current cross-sectional dispersion of long-horizon SPF inflation forecasts), so
        the projection starts from something the reader can verify, and the Okun slope
        governs only the DEVIATION as inflation moves away from where it is now.

        Only the floor moves; the slope — the part actually estimated from history — is
        untouched, so anchoring cannot flatter or steepen the level-uncertainty link.
        """
        shift = float(observed_vol) - self.sigma(at_level)
        return OkunParams(self.floor + shift, self.slope, self.kink, self.r2, self.n)


def _fit_at(level: np.ndarray, vol: np.ndarray, kink: float):
    X = np.column_stack([np.ones(len(level)), np.maximum(level - kink, 0.0)])
    beta, *_ = np.linalg.lstsq(X, vol, rcond=None)
    resid = vol - X @ beta
    denom = ((vol - vol.mean()) ** 2).sum()
    r2 = 1.0 - (resid @ resid) / denom if denom > 0 else float("nan")
    return beta, r2


def fit_hockey_stick(level, vol, kink: float | None = None,
                     grid=np.arange(0.5, 4.01, 0.25)) -> OkunParams:
    """Least-squares fit of ``vol = floor + slope * max(level - kink, 0)``.

    With ``kink=None`` the breakpoint is profiled over ``grid`` and the best-fitting
    value returned. The R2 surface is FLAT over roughly 0.5-1.5 (see FINDINGS), so treat
    the fitted breakpoint as "somewhere around 1-1.5%" rather than a precise estimate,
    and pass an explicit ``kink`` when you want a fixed, defensible number.

    Fit on ANNUAL observations. The level/vol pairs come from overlapping rolling
    windows, so consecutive months are nearly identical and a monthly fit would imply a
    sample size roughly twelve times the real one.
    """
    level = np.asarray(level, dtype=float)
    vol = np.asarray(vol, dtype=float)
    if len(level) < 8:
        raise ValueError(f"need at least 8 observations to fit, got {len(level)}")
    if kink is not None:
        beta, r2 = _fit_at(level, vol, kink)
        return OkunParams(beta[0], beta[1], float(kink), r2, len(level))
    best = None
    for k in grid:
        beta, r2 = _fit_at(level, vol, float(k))
        if best is None or r2 > best[1]:
            best = (beta, r2, float(k))
    beta, r2, k = best
    return OkunParams(beta[0], beta[1], k, r2, len(level))
