"""Ten-year nominal term premia, US / UK / Germany / Japan, from 1975, in one CSV
(percent) -- the series the Atlas plots: daily from the canonical
``viz/data_daily.parquet`` (what the Supabase ``term_premia`` table holds), and where a
market has no daily data, the Atlas's MONTH-END history (the same one-time level match,
``model.global_factor.splice_backfill``):

    Germany  month-end 1975-01 .. 1997-07, daily from 1997-08-07
    Japan    month-end 1975-01 .. 1986-06 (the history's nine-year node, as the Atlas
             uses), daily from 1986-07-05
    US daily from 1975; UK daily from 1979-01-02 (no earlier data)

Run by the daily update after the rebuild; committed with the other exports.

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
# market -> (Atlas history overlay, its node used for the ten-year)
HISTORY = {"DE": ("germany_hist_tp.json", "10"), "JP": ("japan_hist_tp.json", "9")}


def build() -> pd.DataFrame:
    d = pd.read_parquet(DAILY)
    d = d[(d["instrument"] == "nominal") & (d["tenor"] == 10.0) & d["country"].isin(MARKETS)]
    w = d.pivot_table(index="date", columns="country", values="term_premium") * 100.0
    w = add_history(w.reindex(columns=list(MARKETS)))
    w = w.loc[START:].round(4)
    w.columns = [f"{c}_tp10" for c in w.columns]
    w.index = pd.DatetimeIndex(w.index).date
    w.index.name = "date"
    return w


def add_history(w: pd.DataFrame) -> pd.DataFrame:
    """Month-end history (percent) before each market's first daily observation."""
    from model.global_factor import _load_hist_tp, splice_backfill
    w = w.copy()
    w.index = pd.DatetimeIndex(w.index)
    for c, (fname, node) in HISTORY.items():
        hist = _load_hist_tp(fname, node)
        prod = w[c].dropna()
        if hist is None or prod.empty:
            continue
        spliced = splice_backfill(prod / 100.0, hist) * 100.0
        pre = spliced[spliced.index < prod.index.min()]
        w = w.reindex(w.index.union(pre.index))
        w.loc[pre.index, c] = pre.values
    return w.sort_index()


if __name__ == "__main__":
    t = build()
    t.to_csv(OUT)
    print(f"{OUT.relative_to(ROOT)}: {len(t)} days, {t.index.min()}..{t.index.max()}")
