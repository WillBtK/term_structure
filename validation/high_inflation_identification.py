"""Can the HIGH-inflation branch of the premium relationship be identified from data
we actually trust?

Run: ``python -m validation.high_inflation_identification``

The Okun link -- inflation level breeds inflation volatility -- is estimated on core CPI
against core CPI. Both sides are observed data, so the 1970s can be used there without
reservation. The exposed link is the next one, from inflation to the TERM PREMIUM, whose
left-hand side is our own model output and whose high-inflation observations sit in the
regime the Kim filter tells us to exclude.

Three questions, in order:

A. Does shortening the EWMA memory bring the 2021-24 episode into the high-inflation
   sample, and does the upper branch change when it does? (It does, and it does.)
B. Is the upper branch identified across several markets, or off one episode?
C. Does the relationship fitted on modern data, extrapolated back to the 1970s, agree
   with the INDEPENDENT regime-switching historical exhibit? That is the one honest use
   of the 1970s: not to fit, but to test.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from macro.fetch_core_cpi import core_inflation_yoy
from macro.tp_history import _load_history, _production_monthly, spliced_tp

logger = logging.getLogger(__name__)

# Each market's regime-clean production window, per README.
CLEAN = {"US": "1987-01-01", "DE": "1984-01-01", "JP": "2000-05-01",
         "UK": "1995-11-01", "CH": "1995-01-01", "CA": "1995-01-01"}


def ols(y, cols):
    A = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    dof = max(len(y) - A.shape[1], 1)
    se = np.sqrt(np.maximum(np.diag(np.linalg.pinv(A.T @ A)) * (r @ r) / dof, 1e-18))
    return b, b / se, 1 - (r @ r) / ((y - y.mean()) ** 2).sum()


def clean_panel(halflife_years: float = 4.5, tenor: float = 10.0) -> pd.DataFrame:
    """Term premium against EWMA core inflation, each market inside its clean window."""
    infl, tp = core_inflation_yoy(), spliced_tp(tenor)
    rows = []
    for c, start in CLEAN.items():
        if c not in tp.columns:
            continue
        e = infl[c].ewm(halflife=halflife_years * 12, min_periods=24).mean()
        d = pd.DataFrame({"tp": tp[c], "pi": e}).dropna()
        d = d[d.index >= start]
        d["country"] = c
        rows.append(d)
    return pd.concat(rows)


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    print("A. memory length, and whether 2021-24 counts as high inflation")
    print("   half-life   n     n>3.5%   of which 2015+   slope 2-3.5   slope >3.5")
    for hl in (1.0, 2.0, 3.0, 4.5):
        P = clean_panel(hl)
        pi = P["pi"].values
        hi = P[P["pi"] > 3.5]
        b, _, _ = ols(P["tp"].values,
                      [np.maximum(pi - 2.0, 0), np.maximum(pi - 3.5, 0)])
        print(f"   {hl:4.1f}y     {len(P):5}  {len(hi):6}   "
              f"{int((hi.index.year >= 2015).sum()):12}   {b[1]:+.3f}       {b[1]+b[2]:+.3f}")

    print("\nB. is the upper branch identified across markets, or off one episode?")
    P = clean_panel()
    for c in CLEAN:
        h = P[(P["country"] == c) & (P["pi"] > 3.0)]
        if len(h) < 30:
            continue
        b, tt, r2 = ols(h["tp"].values, [h["pi"].values])
        yrs = sorted(set(h.index.year))
        print(f"   {c}: n={len(h):4}  {yrs[0]}-{yrs[-1]}   slope {b[1]:+.3f} "
              f"(t={tt[1]:+.1f})  R2={r2:.3f}")

    print("\nC. extrapolate the modern fit back to the 1970s and test it against the")
    print("   independent regime-switching exhibit (Germany)")
    infl = core_inflation_yoy()["DE"].ewm(halflife=54, min_periods=60).mean()
    tp = spliced_tp(10.0)["DE"]
    mod = pd.DataFrame({"tp": tp, "pi": infl}).dropna()
    mod = mod[mod.index >= CLEAN["DE"]]
    p = mod["pi"].values
    b, _, r2 = ols(mod["tp"].values, [np.maximum(p - 2.0, 0), np.maximum(p - 3.5, 0)])
    prod = _production_monthly(10.0)["DE"].dropna()
    prod.index = prod.index.to_period("M")
    hist = _load_history("DE", 10.0)
    seam = hist.index.intersection(prod.index).min()
    hist = hist + float(prod[seam] - hist[seam])          # same level match as the splice
    hist.index = hist.index.to_timestamp()
    old = pd.DataFrame({"hist": hist, "pi": infl}).dropna()
    old = old[old.index < CLEAN["DE"]]
    q = old["pi"].values
    old["pred"] = b[0] + b[1] * np.maximum(q - 2.0, 0) + b[2] * np.maximum(q - 3.5, 0)
    print(f"   fitted on DE {CLEAN['DE'][:4]}+ (n={len(mod)}), R2={r2:.3f}")
    print(f"   applied to {old.index.min().date()}..{old.index.max().date()} (n={len(old)})")
    print(f"     regime exhibit  mean {old['hist'].mean():+.2f}%  "
          f"range {old['hist'].min():+.2f} to {old['hist'].max():+.2f}")
    print(f"     extrapolation   mean {old['pred'].mean():+.2f}%  "
          f"range {old['pred'].min():+.2f} to {old['pred'].max():+.2f}")
    print(f"     correlation {old['hist'].corr(old['pred']):+.3f}   "
          f"mean error {(old['pred'] - old['hist']).mean():+.2f}pp")


if __name__ == "__main__":
    _main()
