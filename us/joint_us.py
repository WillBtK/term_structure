"""US joint nominal/real AFNS spike — estimate, decompose, and validate vs DKW.

Builds a matched US nominal (GSW) + real (TIPS) monthly panel (2004+), estimates the
joint single-kernel model (`model.joint_afns`), and reports the model-consistent real
term premium, inflation-risk premium (IRP) and expected inflation against (a) the DKW
benchmark and (b) the current "two separate AFNS, subtract" numbers.

Run (slow, background):
    python -m us.joint_us            # estimate + save results to scratchpad/joint_us_*
    python -m us.joint_us --compare  # load saved results + print DKW/separate comparison
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from model import joint_afns as ja

logger = logging.getLogger("us.joint_us")
SCRATCH = Path(__file__).resolve().parent.parent / "scratchpad"
NOM_GRID = np.array([1., 2., 3., 5., 7., 10.])
REAL_GRID = np.array([2., 3., 5., 7., 10.])
START = "2004-01-01"


def build_matched_panels():
    """Matched month-end US nominal + real panels (decimal) from START."""
    from us.fetch_gsw import build_yield_panel as nom_panel
    from us.tips import build_real_panel
    nom = nom_panel(grid=NOM_GRID)
    nom = nom[nom.index >= pd.Timestamp(START)]
    nom = nom.groupby([nom.index.year, nom.index.month]).tail(1)
    real = build_real_panel(grid=REAL_GRID, freq="M", start=START)
    idx = nom.index.intersection(real.index)
    return nom.loc[idx], real.loc[idx]


def run(maxiter: int = 2500, liq: bool = False) -> dict:
    nom, real = build_matched_panels()
    Yn, Yr = nom.to_numpy(float), real.to_numpy(float)
    logger.info("joint US%s: %d months %s..%s | nom %s real %s", " +liquidity" if liq else "",
                len(nom), nom.index.min().date(), nom.index.max().date(),
                list(NOM_GRID), list(REAL_GRID))
    p, res = ja.estimate(Yn, Yr, NOM_GRID, REAL_GRID, dt=1 / 12, elb_n=-0.0035,
                         maxiter=maxiter, n_per_year=24, liq=liq)
    xf, _, ll = ja.filter_states(p, Yn, Yr, NOM_GRID, REAL_GRID, 1 / 12)
    d = ja.decompose(p, xf, NOM_GRID, REAL_GRID, mr_halflife=3.5)
    dates = nom.index
    out = {"dates": [str(x.date()) for x in dates], "loglik": float(ll), "liq": liq,
           "success": bool(res.success), "common": [float(c) for c in d["common"]],
           "kappa": p.kappa.tolist(), "theta": p.theta.tolist(), "sigma": p.sigma.tolist(),
           "lambda": float(p.lambda_), "h_n": float(p.h_n), "h_r": float(p.h_r)}
    for m in (5.0, 10.0):
        jn = int(np.where(np.isclose(NOM_GRID, m))[0][0])
        jr = int(np.where(np.isclose(REAL_GRID, m))[0][0])
        jc = int(np.where(np.isclose(np.array(d["common"]), m))[0][0])
        out[f"tp_n_{m:g}"] = (d["tp_n"][:, jn] * 100).round(4).tolist()
        out[f"tp_r_{m:g}"] = (d["tp_r"][:, jr] * 100).round(4).tolist()
        out[f"irp_{m:g}"] = (d["irp"][:, jc] * 100).round(4).tolist()
        out[f"einfl_{m:g}"] = (d["expected_inflation"][:, jc] * 100).round(4).tolist()
        out[f"liq_{m:g}"] = (d["liquidity_premium"][:, jc] * 100).round(4).tolist()
    SCRATCH.mkdir(exist_ok=True)
    fname = "joint_us_liq_results.json" if liq else "joint_us_results.json"
    (SCRATCH / fname).write_text(json.dumps(out))
    logger.info("joint US done: loglik=%.1f success=%s -> scratchpad/%s", ll, res.success, fname)
    return out


def compare(liq: bool = False) -> None:
    """Print joint vs DKW (and note vs separate-AFNS) at 5y/10y, latest + correlation."""
    fname = "joint_us_liq_results.json" if liq else "joint_us_results.json"
    res = json.loads((SCRATCH / fname).read_text())
    dates = pd.DatetimeIndex(res["dates"])
    from validation.benchmarks import load_dkw
    dkw = load_dkw()
    print(f"\nJoint US spike — {dates.min().date()}..{dates.max().date()} "
          f"(loglik {res['loglik']:.0f}, success {res['success']})\n")
    for m in (5, 10):
        js = pd.Series(res[f"irp_{m}"], index=dates)
        jr = pd.Series(res[f"tp_r_{m}"], index=dates)
        je = pd.Series(res[f"einfl_{m}"], index=dates)
        dk = dkw[dkw.maturity == m].set_index("date")
        dki = (dk["inflation_risk_premium"] * 100).reindex(dates, method="nearest")
        dkr = (dk["real_term_premium"] * 100).reindex(dates, method="nearest")
        dke = (dk["expected_inflation"] * 100).reindex(dates, method="nearest")
        def line(name, a, b):
            corr = a.corr(b)
            print(f"  {m:>2}y {name:<20} joint {a.iloc[-1]:+.2f}%  DKW {b.iloc[-1]:+.2f}%  "
                  f"| corr {corr:.2f}  mean gap {(a-b).mean():+.2f}pp")
        line("real term premium", jr, dkr)
        line("inflation-risk prem", js, dki)
        line("expected inflation", je, dke)
        if res.get("liq"):
            jl = pd.Series(res[f"liq_{m}"], index=dates)
            dkl = (dk["tips_liquidity_premium"] * 100).reindex(dates, method="nearest") \
                if "tips_liquidity_premium" in dk.columns else None
            extra = f"  DKW {dkl.iloc[-1]:+.2f}% corr {jl.corr(dkl):.2f}" if dkl is not None else ""
            print(f"  {m:>2}y {'TIPS liquidity prem':<20} joint {jl.iloc[-1]:+.2f}%{extra}")
        print()


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description="US joint nominal/real AFNS spike")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--liq", action="store_true", help="add the TIPS liquidity factor (5-factor)")
    ap.add_argument("--maxiter", type=int, default=2500)
    args = ap.parse_args()
    if args.compare:
        compare(liq=args.liq)
    else:
        run(maxiter=args.maxiter, liq=args.liq)


if __name__ == "__main__":
    _main()
