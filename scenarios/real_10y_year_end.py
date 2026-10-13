"""Implied ten-year real yields by scenario: the sheet's nominal year-end projections
(``scenarios.year_end_10y``) less the breakeven projections (``model.breakeven``).
Both are zero-coupon on the same curves (GSW nominal/TIPS, BoE nominal/real), so today's
column reproduces the market's real yield.

    python -m scenarios.real_10y_year_end
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scenarios" / "real_10y_year_end.csv"
LINKER = {"US": "TIPS", "UK": "Index-linked gilt (RPI)", "EA": "OAT€i / BTP€i (HICPxT)"}
SHEET_CODE = {"US": "US", "UK": "UK", "EA": "DE"}


def build() -> pd.DataFrame:
    from model.breakeven import year_end_table
    from scenarios.year_end_10y import evaluate
    be = year_end_table()
    nom = evaluate(["all"])
    rows = []
    for _, r in be.iterrows():
        res = nom[r["scenario"]]
        n = res["rows"][SHEET_CODE[r["market"]]]
        if r["market"] == "EA":
            continue          # the sheet projects Bunds, not OATs or BTPs
        row = {"market": r["market"], "linker": LINKER[r["market"]], "scenario": r["scenario"],
               "Today": round(n["today"] - r["Today"], 2)}
        row.update({str(y): round(v - r[str(y)], 2) for y, v in zip(res["years"], n["y"])})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = build()
    t.to_csv(OUT, index=False)
    print(t.to_string(index=False))
