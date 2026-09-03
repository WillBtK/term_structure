"""Monthly global term-premium factor, exported for the fiscal-sensitivity repo's
cross-market design (``WillBtK/imf_fiscal_sensitivity``, ``docs/CROSS_MARKET_DESIGN.md``).

One row per (month, country) from the **10y term-premium DFM**
(``model/global_factor.py``, ``GF_ORDER`` = US/DE/JP/UK/CA) — the model's existing
default combo (``viz/global_factor_meta.json``'s ``"default"``), re-decomposed here
rather than a fresh fit, so this file and the Atlas's "Global cycle" panel always agree.

``G`` and ``residual`` are in the model's STANDARDIZED space (``G`` ~ N(0,1) marginally;
``residual`` is the standardized local residual ``z_i,t - loading_i * G_t``). To invert
back to raw term-premium percent, this file also carries each market's ``mean_pct`` /
``sd_pct`` — the mean and SD used to standardize it:

    TP_i,t (%, raw) = mean_pct_i + sd_pct_i * (loading_i * G_t + residual_i,t)

**Switzerland is not a row in this file.** It is excluded from the DFM *by design* —
its government-bond market liquidity is too thin next to the majors, so including it
would let its own noise distort the shared factor for everyone else (see the comment on
``model.global_factor.GF_ORDER``) — not a short-history or partial-coverage case, which
is the caveat this file's request asked to flag. If a Swiss series is needed, it has to
come from the plain nominal export (``multi_market_term_premia_monthly.csv``), read on
its own rather than relative to this factor.

**Coverage the DFM's loadings/residuals ARE estimated on**, per market's own full
history (the EM step masks each market's missing months rather than requiring a
balanced panel): US/DE/JP from 1974-09 (DE/JP via the committed pre-modern-curve
backfill, one-time level-spliced onto the production series — see
``model.global_factor.splice_backfill``), UK from 1979-01, CA from 1986-01. The
loadings/residuals are still estimated over each market's own available months; only
the standardization mean/sd use the fully-balanced window (below), so a market's
earlier, non-balanced months are decomposed with the same parameters, not different
ones.

    python -m exports.global_factor_monthly
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import global_factor as gf

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ROOT / "viz" / "params" / "global_factor.json"
OUT = Path(__file__).resolve().parent / "global_factor_monthly.csv"
TENOR = 10.0
COMBO = "term_premium@10y"


def build() -> tuple[pd.DataFrame, str]:
    """Long-form (month, country) -> G, loading, residual, mean_pct, sd_pct.

    Returns the frame plus a note string (the balanced-window start) for the CLI
    summary — not part of the CSV, since it is a single fact about the fit rather than
    a per-row value.
    """
    panel = gf.load_measure_panel("term_premium", tenor=TENOR, freq="M")   # GF_ORDER order
    names = list(panel.columns)
    all_params = json.loads(PARAMS.read_text())
    if COMBO not in all_params:
        raise RuntimeError(f"{COMBO!r} not in {PARAMS.relative_to(ROOT)} — "
                           f"run `python -m viz.global_factor` first")
    p = gf.DFMParams.from_dict(all_params[COMBO])
    if p.names != names:
        raise RuntimeError(f"panel market order {names} != fitted params order {p.names} "
                           f"— GF_ORDER must have changed since this combo was fit")

    Y = panel.to_numpy(float)
    dec = gf.filter_forward(p, Y, smoothed=True)          # full-sample smoothed, for description
    G = dec["G"]
    resid_std = dec["residual"] / p.sd[None, :]            # raw residual -> standardized residual
    months = panel.index.values.astype("datetime64[M]")

    frames = []
    for i, c in enumerate(names):
        d = pd.DataFrame({
            "month": months, "country": c,
            "G": np.round(G, 5),
            "loading": round(float(p.lam[i]), 5),
            "residual": np.round(resid_std[:, i], 5),
            "mean_pct": round(float(p.mu[i]) * 100.0, 4),
            "sd_pct": round(float(p.sd[i]) * 100.0, 4),
        })
        d = d[np.isfinite(panel[c].to_numpy(float))]        # only months that market reports
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["country"] = pd.Categorical(out["country"], categories=names, ordered=True)
    out = out.sort_values(["country", "month"]).reset_index(drop=True)

    balanced_start = str(panel.index[panel.notna().all(axis=1)].min().date())
    return out, balanced_start


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    m, balanced_start = build()
    m.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(m):,} rows, "
          f"{str(m['month'].min())[:7]}..{str(m['month'].max())[:7]}, "
          f"{OUT.stat().st_size:,} bytes)")
    print(f"combo {COMBO}  markets {sorted(m['country'].unique().tolist())}  "
          f"standardization window from {balanced_start} (first fully-balanced month)")
    for c in m["country"].cat.categories:
        sub = m[m["country"] == c]
        print(f"  {c}: {len(sub):>4} rows, {str(sub['month'].min())[:7]}..{str(sub['month'].max())[:7]}, "
              f"loading {sub['loading'].iloc[0]:+.3f}, mean {sub['mean_pct'].iloc[0]:.2f}% "
              f"sd {sub['sd_pct'].iloc[0]:.2f}%")
    print("  CH: excluded by design (not in the DFM) — see the module docstring")


if __name__ == "__main__":
    _main()
