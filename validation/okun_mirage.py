"""Estimate Okun's level -> inflation-uncertainty map on the six-market core CPI panel.

Run: ``python -m validation.okun_mirage``

Reports, in order: the non-parametric relationship (buckets, no functional form
imposed), the fitted hockey stick, and four robustness cuts — breakpoint sensitivity,
per-market fits, leave-one-market-out, and era stability. Findings written up in
``validation/FINDINGS_okun.md``.

The fits are on ANNUAL observations. Level/vol pairs come from overlapping 10-year
rolling windows, so consecutive months are almost the same observation; fitting monthly
would imply roughly twelve times the sample size actually present.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from model.okun import DEFAULT_WINDOW, fit_hockey_stick, rolling_level_vol

logger = logging.getLogger(__name__)

BUCKETS = [-2, 0, 1, 2, 3, 4, 6, 8, 20]


def build_panel(window: int = DEFAULT_WINDOW, refresh: bool = False) -> pd.DataFrame:
    """Long-form (date, country, level, vol) panel of trailing core-inflation regimes."""
    from macro.fetch_core_cpi import core_inflation_yoy
    infl = core_inflation_yoy(refresh=refresh)
    frames = []
    for country in infl.columns:
        d = rolling_level_vol(infl[country], window=window)
        d["country"] = country
        frames.append(d)
    return (pd.concat(frames).reset_index().rename(columns={"index": "date"})
            .sort_values(["country", "date"]).reset_index(drop=True))


def annual(panel: pd.DataFrame) -> pd.DataFrame:
    """One observation per market-year (December), to respect the overlapping windows."""
    return panel[panel["date"].dt.month == 12].copy()


def nonparametric(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean volatility by inflation-level bucket — the result before any functional form."""
    b = pd.cut(panel["level"], BUCKETS)
    return (panel.groupby(b, observed=True)
            .agg(n=("vol", "size"), mean_level=("level", "mean"), mean_vol=("vol", "mean"))
            .round(2))


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    panel = build_panel()
    A = annual(panel)
    print(f"panel: {len(panel)} monthly obs -> {len(A)} annual, {A.country.nunique()} markets, "
          f"{A.date.min().date()}..{A.date.max().date()}\n")

    print("=== NON-PARAMETRIC: 10y core-inflation vol by 10y mean level ===")
    print(nonparametric(panel).to_string(), "\n")

    fit = fit_hockey_stick(A["level"], A["vol"])
    print("=== FITTED HOCKEY STICK (annual obs) ===")
    print(f"  vol = {fit.floor:.2f} + {fit.slope:.2f} * max(level - {fit.kink:.2f}, 0)"
          f"    R2={fit.r2:.3f}  n={fit.n}")
    lin = np.polyfit(A["level"], A["vol"], 1)
    r_lin = A["vol"] - np.polyval(lin, A["level"])
    ss = ((A["vol"] - A["vol"].mean()) ** 2).sum()
    print(f"  (linear comparator R2={1 - (r_lin @ r_lin) / ss:.3f})\n")

    print("=== BREAKPOINT SENSITIVITY (R2 at a fixed kink) ===")
    print("  " + "  ".join(f"{k:.2f}:{fit_hockey_stick(A['level'], A['vol'], kink=k).r2:.3f}"
                           for k in (0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)) + "\n")

    print("=== PER-MARKET (kink fixed at 1.25) ===")
    for c in sorted(A.country.unique()):
        q = A[A.country == c]
        f = fit_hockey_stick(q["level"], q["vol"], kink=1.25)
        print(f"  {c}: floor={f.floor:5.2f}  slope={f.slope:4.2f}  R2={f.r2:.3f}  n={f.n}")
    print()

    print("=== LEAVE-ONE-MARKET-OUT ===")
    for c in sorted(A.country.unique()):
        f = fit_hockey_stick(*[A[A.country != c][k] for k in ("level", "vol")])
        print(f"  ex-{c}: floor={f.floor:.2f}  slope={f.slope:.2f}  kink={f.kink:.2f}  R2={f.r2:.3f}")
    print()

    print("=== ERA STABILITY ===")
    for lab, q in (("<=1995", A[A.date.dt.year <= 1995]), ("1996+", A[A.date.dt.year > 1995])):
        f = fit_hockey_stick(q["level"], q["vol"])
        print(f"  {lab:7}: floor={f.floor:.2f}  slope={f.slope:.2f}  kink={f.kink:.2f}"
              f"  R2={f.r2:.3f}  n={f.n}")
    print()

    print("=== WINDOW SENSITIVITY (5y vs 10y regime window) ===")
    for w in (60, DEFAULT_WINDOW):
        f = fit_hockey_stick(*[annual(build_panel(window=w))[k] for k in ("level", "vol")])
        print(f"  {w:3}m: floor={f.floor:.2f}  slope={f.slope:.2f}  kink={f.kink:.2f}  R2={f.r2:.3f}")
    print()

    print("=== PROJECTION EXAMPLE (what a scenario terminal inflation implies) ===")
    for lv in (0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        print(f"  terminal core inflation {lv:4.1f}%  ->  inflation uncertainty {fit.sigma(lv):.2f}pp")


if __name__ == "__main__":
    _main()
