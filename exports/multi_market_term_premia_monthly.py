"""Monthly term premia across six markets, exported for the fiscal-sensitivity repo's
cross-market design (``WillBtK/imf_fiscal_sensitivity``, ``docs/CROSS_MARKET_DESIGN.md``).

Long format — one row per (month, country, tenor) — because the six markets do not
share a tenor grid (``us/uk/germany/japan/switzerland/canada`` config.py ``COMMON_GRID``
all differ) and forcing a common one would mean padding every market out to the union
with columns nobody asked for. Real term premia exist only for US and UK (the only two
markets with a production real curve); ``rtp`` is blank everywhere else, not a zero.

Same source and convention as ``exports/us_term_premia_monthly.csv``: calendar-month
average of daily observations from the canonical ``viz/data_daily.parquet``, in percent
(the parquet is decimal), months with fewer than ``MIN_DAYS`` observations for that
(month, tenor) dropped. **Japan's YCC-period measurement downweight is already baked
into the canonical parquet** (applied at the estimation step that produces it, in
``japan.production_tp`` / ``viz.export_daily`` — see ``japan.config.ycc_downweight``),
so this export inherits it automatically; nothing extra is applied here.

    python -m exports.multi_market_term_premia_monthly
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "viz" / "data_daily.parquet"
OUT = Path(__file__).resolve().parent / "multi_market_term_premia_monthly.csv"

# Order matches the data request. Switzerland has a production nominal curve but no
# real curve (see REAL_COUNTRIES) — it still gets a `tp` column here, just no `rtp`.
COUNTRIES = ["US", "UK", "DE", "JP", "CH", "CA"]
REAL_COUNTRIES = {"US", "UK"}          # the only two markets with a production real curve
MIN_DAYS = 5          # a (month, tenor) with fewer observations than this is dropped


def _monthly(daily: pd.DataFrame, country: str, instrument: str) -> pd.DataFrame:
    """One (month, tenor) row per average, with the trading-day count that went into it."""
    d = daily[(daily["country"] == country) & (daily["instrument"] == instrument)]
    if d.empty:
        return pd.DataFrame(columns=["month", "tenor", "value", "n_days"])
    d = d.assign(month=d["date"].values.astype("datetime64[M]"))
    g = d.groupby(["month", "tenor"])["term_premium"]
    out = pd.DataFrame({"value": g.mean() * 100.0, "n_days": g.size()}).reset_index()
    return out[out["n_days"] >= MIN_DAYS]


def build(countries: list[str] = COUNTRIES) -> pd.DataFrame:
    """Long-form (month, country, tenor) -> tp, rtp, n_days, in percent."""
    daily = pd.read_parquet(DAILY)
    frames = []
    for c in countries:
        nom = _monthly(daily, c, "nominal").rename(columns={"value": "tp"})
        if nom.empty:
            logger.warning("%s: no nominal term premium in the canonical dataset", c)
            continue
        if c in REAL_COUNTRIES:
            real = _monthly(daily, c, "real").rename(columns={"value": "rtp"})
            nom = nom.merge(real[["month", "tenor", "rtp"]], on=["month", "tenor"], how="left")
        else:
            nom["rtp"] = float("nan")
        nom.insert(1, "country", c)
        frames.append(nom)
    out = pd.concat(frames, ignore_index=True)
    out["country"] = pd.Categorical(out["country"], categories=COUNTRIES, ordered=True)
    out = out.sort_values(["country", "tenor", "month"]).reset_index(drop=True)
    out["tp"] = out["tp"].round(4)
    out["rtp"] = out["rtp"].astype(float).round(4)
    return out[["month", "country", "tenor", "tp", "rtp", "n_days"]]


def _main() -> None:
    logging.basicConfig(level=logging.WARNING)
    m = build()
    m.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(m):,} rows, "
          f"{str(m['month'].min())[:7]}..{str(m['month'].max())[:7]}, "
          f"{OUT.stat().st_size:,} bytes)")
    for c in COUNTRIES:
        sub = m[m["country"] == c]
        if sub.empty:
            print(f"  {c}: MISSING — no rows written")
            continue
        tenors = sorted(sub["tenor"].unique())
        has_rtp = sub["rtp"].notna().any()
        print(f"  {c}: {len(sub):>4} rows, tenors {tenors}, "
              f"{str(sub['month'].min())[:7]}..{str(sub['month'].max())[:7]}"
              + ("  (+real)" if has_rtp else ""))


if __name__ == "__main__":
    _main()
