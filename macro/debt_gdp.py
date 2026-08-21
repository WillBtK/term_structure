"""Debt / GDP: the level we measure on, and the path we project it along.

Two problems, solved separately.

**The level.** Different sources mean different things by "debt". The IMF's WEO measures
GENERAL GOVERNMENT GROSS debt, which for the US is about 128% of GDP; the series that
belongs in a Treasury term-premium model is federal debt HELD BY THE PUBLIC, about 99%,
because that is the paper private investors actually have to absorb. Intragovernmental
holdings are not supply. So the history always comes from the market's own right-concept
series.

**The path.** No single official projection covers every market, and where several exist
they disagree. So the projection is taken as the average CHANGE across whatever sources a
market has, applied cumulatively to the last observed level of the right-concept series.
Taking changes rather than levels means a source measuring a different concept still
contributes its view of the trajectory without dragging the level onto its own basis.

Sources per market:

* **US** — IMF WEO. The CBO 10-year projections are the better object but cbo.gov sits
  behind a DataDome bot wall that refuses datacenter traffic, including direct file
  requests, so it cannot be fetched. Drop the workbook into ``macro/data/cbo_debt.csv``
  and it takes precedence; the file is checked for automatically.
* **DE** — IMF WEO and AMECO, averaged.
* **UK** — IMF WEO and the OBR's EFO public sector net debt.
* **JP** — IMF WEO and the Cabinet Office medium-to-long-term projections.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent

# The level we measure on: the concept that matches what private investors must hold.
HISTORY = {
    "US": "FYGFGDQ188S",      # federal debt held by the public, % GDP, quarterly 1970+
    "UK": "GGGDTAGBA188N",    # general government gross, annual 1980+
    "DE": "GGGDTADEA188N",    # annual 1991+
    "JP": "GGGDTAJPA188N",    # annual 1980+
}

# IMF WEO projections, annual, ~6 years forward.
WEO_PROJECTION = {"US": "GGGDTPUSA188N", "UK": "GGGDTPGBA188N",
                  "DE": "GGGDTPDEA188N", "JP": "GGGDTPJPA188N"}

# Optional national projections dropped in as CSV (date,value in % of GDP). Averaged
# with WEO where present.
NATIONAL_FILES = {
    "US": HERE / "data" / "cbo_debt.csv",
    "UK": HERE / "data" / "obr_debt.csv",
    "DE": HERE / "data" / "ameco_debt.csv",
    "JP": HERE / "data" / "cao_debt.csv",
}


def _annual(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.year).last()


def _read_csv(path: Path) -> "pd.Series | None":
    if not path.is_file():
        return None
    d = pd.read_csv(path)
    s = pd.Series(pd.to_numeric(d["value"], errors="coerce").values,
                  index=pd.to_datetime(d["date"]).dt.year).dropna()
    return s.sort_index()


def history(country: str) -> pd.Series:
    """Observed debt/GDP on the right concept, annual."""
    from validation.fred_api import fred_series
    return _annual(fred_series(HISTORY[country]).dropna())


def projection(country: str, years: int = 12) -> pd.Series:
    """Debt/GDP path, annual, anchored to the last observed level.

    Every source contributes its CHANGE from the last common year; the changes are
    averaged and applied to the observed level. A source on a different definitional
    basis therefore shifts the path without shifting its starting point.
    """
    from validation.fred_api import fred_series
    hist = history(country)
    base_year = int(hist.index.max())
    base = float(hist.loc[base_year])

    paths = {}
    try:
        weo = _annual(fred_series(WEO_PROJECTION[country]).dropna())
        paths["WEO"] = weo
    except Exception as exc:                                        # noqa: BLE001
        logger.warning("%s: WEO projection unavailable (%s)", country, exc)
    nat = _read_csv(NATIONAL_FILES[country])
    if nat is not None:
        paths["national"] = nat

    if not paths:
        logger.warning("%s: no projection source, holding %.1f%% flat", country, base)
        return pd.Series(base, index=range(base_year + 1, base_year + years + 1))

    deltas = []
    for name, p in paths.items():
        anchor_candidates = [y for y in p.index if y <= base_year]
        if not anchor_candidates:
            # Projection starts after our history ends: anchor on its own first year.
            anchor = int(min(p.index))
            d = p - float(p.loc[anchor])
            logger.info("  %s/%s: no overlap with history, anchored on its own %d",
                        country, name, anchor)
        else:
            anchor = max(anchor_candidates)
            d = p - float(p.loc[anchor])
        deltas.append(d[d.index > base_year])
    change = pd.concat(deltas, axis=1).mean(axis=1).dropna()
    out = (base + change).reindex(range(base_year + 1, base_year + years + 1))
    logger.info("  %s: base %.1f%% (%d) from %s; sources %s; %d -> %.1f%%",
                country, base, base_year, HISTORY[country], sorted(paths),
                int(out.dropna().index.max()) if out.notna().any() else base_year,
                out.dropna().iloc[-1] if out.notna().any() else base)
    return out.ffill()


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for c in ("US", "UK", "DE", "JP"):
        h = history(c)
        p = projection(c)
        print(f"\n{c}: history {h.index.min()}-{h.index.max()}, latest {h.iloc[-1]:.1f}%")
        print("   projected: " + "  ".join(f"{y}:{v:.1f}" for y, v in p.dropna().items()))


if __name__ == "__main__":
    _main()
