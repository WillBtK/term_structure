"""Daily ten-year nominal term premia, US / UK / Germany / Japan, from 1975 -- the
series the Atlas plots and the Supabase ``term_premia`` table holds, written as a
wide CSV (percent) from the same canonical ``viz/data_daily.parquet``. Run by the
daily update after the rebuild; committed with the other exports.

    python -m exports.term_premia_10y_daily
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "viz" / "data_daily.parquet"
OUT = ROOT / "exports" / "term_premia_10y_daily.csv"
MARKETS = ("US", "UK", "DE", "JP")
START = "1975-01-01"


def build() -> pd.DataFrame:
    d = pd.read_parquet(DAILY)
    d = d[(d["instrument"] == "nominal") & (d["tenor"] == 10.0) & d["country"].isin(MARKETS)]
    w = d.pivot_table(index="date", columns="country", values="term_premium") * 100.0
    w = w.reindex(columns=list(MARKETS)).loc[START:].round(4)
    w.columns = [f"{c}_tp10" for c in w.columns]
    w.index = pd.DatetimeIndex(w.index).date
    w.index.name = "date"
    return w


if __name__ == "__main__":
    t = build()
    t.to_csv(OUT)
    print(f"{OUT.relative_to(ROOT)}: {len(t)} days, {t.index.min()}..{t.index.max()}")
