"""Long MONTHLY term-premium history per market, for driver estimation.

The daily canonical panel (``viz/data_daily.parquet``) starts where each market's daily
curve starts — 1997 for Germany. That is a poor basis for estimating term-premium
drivers, because the relationships being estimated are identified by the regime change of
the 1980s-90s, and a 1997 start (which a 10-year rolling window pushes to 2007) sits
entirely on one side of it.

Germany and Japan both have a committed MONTHLY historical exhibit going back to 1972/74,
produced offline from the monthly Bundesbank/MoF curves. This module splices those onto
the daily production series so driver estimation gets the full history.

**The splice is not optional and not cosmetic.** The historical exhibits are fitted on the
full sample, whose long-run mean the 1970s-80s high rates drag upward; that biases their
expected-rate path high and their term premium LOW relative to the modern-sample
production series. Using the raw historical series alongside the others would import that
level bias straight into a pooled regression. The alignment applied here is the same
single-point, month-end level match that ``viz/thin_daily.py`` uses for the published
chart: it preserves the 1970s *shape* and removes only the *level* offset.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(__file__).resolve().parent.parent / "viz" / "history"
DAILY_PANEL = Path(__file__).resolve().parent.parent / "viz" / "data_daily.parquet"

# Markets with a committed monthly historical exhibit to splice on.
HISTORY_FILES = {"DE": "germany_hist_tp.json", "JP": "japan_hist_tp.json"}


BUNDESBANK_MONTHLY = (Path(__file__).resolve().parent.parent
                      / "germany" / "data" / "bundesbank_params.parquet")


def long_yield_monthly(tenor: float = 10.0) -> pd.DataFrame:
    """Month-end nominal par yield (percent) per market, history-extended for Germany.

    Splicing the term premium alone was not enough for Germany. Its binding constraint
    was never the premium but the YIELD: the stock-bond correlation is built from equity
    and bond returns, the daily Bund curve starts 1997-08, and a 10-year rolling window
    pushes the first usable correlation out to 2007 — inside a single regime. The
    Bundesbank MONTHLY Svensson parameters run from 1972-09, so the 10y yield is
    reconstructed from those and spliced on, which moves Germany's correlation history
    back with it.
    """
    d = pd.read_parquet(DAILY_PANEL)
    nom = d[(d["instrument"] == "nominal") & (d["tenor"] == float(tenor))]
    wide = (nom.pivot_table(index="date", columns="country", values="observed")
            .resample("MS").last() * 100.0)
    if not BUNDESBANK_MONTHLY.is_file() or "DE" not in wide.columns:
        return wide
    from us.reconstruct import svensson_yield
    p = pd.read_parquet(BUNDESBANK_MONTHLY).dropna()
    hist = pd.Series(
        svensson_yield(float(tenor), p["BETA0"].values, p["BETA1"].values,
                       p["BETA2"].values, p["BETA3"].values,
                       p["TAU1"].values, p["TAU2"].values),
        index=pd.DatetimeIndex(p.index).to_period("M"))
    prod = wide["DE"].dropna()
    prod.index = prod.index.to_period("M")
    common = hist.index.intersection(prod.index)
    if len(common):
        seam = common.min()
        # Reconstructed from the SAME curve family, so no level offset is warranted --
        # but measure the seam gap and log it, because a large one would mean the
        # monthly and daily curves are not the same object and the splice is unsafe.
        gap = float(prod[seam] - hist[seam])
        joined = pd.concat([hist[hist.index < seam], prod]).sort_index()
        logger.info("  DE @%gy yield: monthly Bundesbank history from %s spliced at %s "
                    "(seam gap %+.3fpp), %d -> %d obs", tenor, hist.index.min(), seam,
                    gap, len(prod), len(joined))
        joined.index = joined.index.to_timestamp()
        idx = wide.index.union(joined.index)        # extend backwards for the history
        wide = wide.reindex(idx)
        wide["DE"] = joined.reindex(idx)
    return wide


def _production_monthly(tenor: float) -> pd.DataFrame:
    """Month-end production term premium (percent), columns = country."""
    d = pd.read_parquet(DAILY_PANEL)
    nom = d[(d["instrument"] == "nominal") & (d["tenor"] == float(tenor))]
    wide = nom.pivot_table(index="date", columns="country", values="term_premium")
    return wide.resample("MS").last() * 100.0


def _load_history(country: str, tenor: float) -> pd.Series | None:
    name = HISTORY_FILES.get(country)
    if name is None:                    # no historical exhibit for this market
        return None
    f = HISTORY_DIR / name
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text())
    except Exception:                                               # noqa: BLE001
        logger.warning("could not read %s", f)
        return None
    tenors = d.get("tenors") or {}
    key = next((k for k in tenors if float(k) == float(tenor)), None)
    if key is None or not d.get("dates"):
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(d["dates"])).to_period("M")
    vals = [np.nan if v is None else float(v) for v in tenors[key]]
    return pd.Series(vals, index=idx).dropna()


def spliced_tp(tenor: float = 10.0) -> pd.DataFrame:
    """Monthly term premium (percent) per market, history-extended where available.

    Returns a month-start-indexed frame. Markets with no historical exhibit are returned
    as their production series alone.
    """
    prod = _production_monthly(tenor)
    out = {}
    for country in prod.columns:
        p = prod[country].dropna()
        p.index = p.index.to_period("M")
        hist = _load_history(country, tenor)
        if hist is None or p.empty:
            out[country] = p
            continue
        common = hist.index.intersection(p.index)
        if len(common) < 1:
            out[country] = p
            continue
        seam = common.min()
        offset = float(p[seam] - hist[seam])     # one-time level match at the seam
        joined = pd.concat([(hist + offset)[hist.index < seam], p]).sort_index()
        out[country] = joined
        logger.info("  %s @%gy: history from %s spliced at %s (offset %+.2fpp), "
                    "%d -> %d monthly obs", country, tenor, hist.index.min(), seam,
                    offset, len(p), len(joined))
    df = pd.DataFrame(out)
    df.index = df.index.to_timestamp()
    return df.sort_index()
