"""Where does the stock-bond correlation turn positive, as a level of inflation?

``validation/sbc_macro.py`` establishes the mechanism: the correlation depends on the
relative VOLATILITY of inflation and growth, not on the level of inflation. That is the
right model and it is what the scenario chain uses. But it is not the question an
investment committee asks, which is "at what rate of inflation do bonds stop hedging?"

The two are reconcilable because the level and the volatility are themselves linked --
that is Okun's result, and ``model/okun.py`` fits the kink. Composing the two gives a
level answer, but composing two fitted relationships compounds their errors. This module
takes the shorter route and regresses the realised correlation directly on the realised
inflation LEVEL, pooled across markets.

**Result.** Across the four deck markets, pooled, the fitted correlation crosses zero at
about **3.5%** inflation (90% bootstrap band roughly 2.9-4.2%). At a 2% target the
fitted correlation is about -0.07: bonds hedge. Sustained inflation in the mid-3s is
where that stops.

**A hockey stick was tested and rejected.** Searching a kink over 0.5-7% pins it at the
grid floor, which is the search telling us there is no kink to find: over the observed
range the relationship is a straight line, and the interesting parameter is where that
line crosses zero rather than where it bends. Reported as linear for that reason.

**Read the pooled number, not the per-market ones.** Individually the markets scatter
badly -- crossings run from 0.7% (UK) to 7.7% (Japan) with R-squared between 0.02 and
0.27 -- because each market sees only its own narrow slice of inflation history. The
pooled fit is identified by the cross-section, and the dispersion is a caveat to state,
not a set of four separate answers to quote.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MARKETS = ("US", "UK", "DE", "JP")
N_BOOT = 4000
SEED = 0


def build_panel(markets: tuple = MARKETS) -> pd.DataFrame:
    """Annual (market, inflation level, realised stock-bond correlation)."""
    from validation.sbc_macro import build_panel as sbc_panel
    from macro.fetch_core_cpi import core_inflation_yoy

    sbc = sbc_panel()
    level = core_inflation_yoy().resample("YS").mean()
    rows = []
    for country, grp in sbc.groupby("country"):
        if country not in markets or country not in level.columns:
            continue
        for date, corr in grp.set_index("date")["sbc"].items():
            year = pd.Timestamp(date.year, 1, 1)
            if year in level.index and np.isfinite(level.at[year, country]):
                rows.append({"country": country, "year": int(date.year),
                             "pi": float(level.at[year, country]),
                             "sbc": float(corr)})
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def fit(panel: pd.DataFrame | None = None, n_boot: int = N_BOOT,
        seed: int = SEED) -> dict:
    """Pooled linear fit, the zero crossing, and a bootstrap band for the crossing."""
    p = build_panel() if panel is None else panel
    x, y = p["pi"].to_numpy(), p["sbc"].to_numpy()
    slope, intercept = _line(x, y)
    cross = -intercept / slope if slope else float("nan")
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(p))
    draws = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        b, a = _line(x[s], y[s])
        if b:
            draws.append(-a / b)
    d = np.asarray(draws)
    d = d[(d > -5) & (d < 15)]                 # a near-zero slope throws the crossing
    lo, hi = (float(np.percentile(d, 5)), float(np.percentile(d, 95))) if len(d) else \
        (float("nan"), float("nan"))

    per = {}
    for c, g in p.groupby("country"):
        b, a = _line(g["pi"].to_numpy(), g["sbc"].to_numpy())
        per[c] = {"slope": round(b, 4),
                  "cross": round(-a / b, 2) if b else None,
                  "r2": round(float(np.corrcoef(g["pi"], g["sbc"])[0, 1] ** 2), 2),
                  "n": int(len(g))}

    return {"slope": round(slope, 4), "intercept": round(intercept, 4),
            "cross": round(cross, 2), "cross_lo": round(lo, 2), "cross_hi": round(hi, 2),
            "r2": round(r2, 3), "n": int(len(p)),
            "markets": int(p["country"].nunique()),
            "at2": round(slope * 2 + intercept, 3),
            "at4": round(slope * 4 + intercept, 3),
            "per_market": per}


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    p = build_panel()
    f = fit(p)
    print(f"\npooled, {f['markets']} markets, n={f['n']}")
    print(f"  slope {f['slope']:+.4f} per pp of inflation   R2 {f['r2']:.3f}")
    print(f"  correlation crosses zero at {f['cross']:.2f}% "
          f"(90% band {f['cross_lo']:.2f} to {f['cross_hi']:.2f})")
    print(f"  fitted correlation at 2% = {f['at2']:+.3f}   at 4% = {f['at4']:+.3f}\n")
    print("per market (quote the pooled number, not these):")
    for c, m in f["per_market"].items():
        print(f"  {c}: slope {m['slope']:+.4f}  crosses at "
              f"{m['cross']:6.2f}%  R2 {m['r2']:.2f}  n={m['n']}")


if __name__ == "__main__":
    _main()
